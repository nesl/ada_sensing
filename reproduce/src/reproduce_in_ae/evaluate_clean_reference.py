from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import paper_transform, rgb_loader
from .models import load_model, parameter_count
from .protocol import (
    DINOV2_HUB_REF,
    iter_specs,
    parse_model_keys,
    project_root,
    workspace_root,
)
from .replication import (
    atomic_csv_dump,
    atomic_json_dump,
    closed_class_metadata,
    default_class_index_json,
    default_reference_root,
    load_source_capture_rows,
    sha256_file,
)


EXPECTED_SAMPLE_COUNT = 5
PREDICTION_FIELDS = (
    "model",
    "sample_id",
    "source_image_path",
    "target_imagenet_index",
    "target_closed_index",
    "target_wnid",
    "target_class_name",
    "top1_imagenet_index",
    "top1_closed_index",
    "top1_wnid",
    "top1_class_name",
    "top1_confidence",
    "correct",
)


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(
        description="Evaluate the five clean source images used by replication."
    )
    parser.add_argument(
        "--manual-dataset-root",
        type=Path,
        default=workspace / "data" / "replication" / "manual_dataset",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=workspace / "data" / "replication" / "replicated_capture",
    )
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument(
        "--class-index-json", type=Path, default=default_class_index_json()
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=workspace / "replicate_result" / "comparison" / "clean_reference",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=project_root() / "checkpoints"
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2481757)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dinov2-hub-ref", default=DINOV2_HUB_REF)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clean_rows(
    manual_dataset_root: Path,
    capture_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    capture_rows = load_source_capture_rows(capture_root)
    sample_ids = tuple(dict.fromkeys(str(row["sample_id"]) for row in capture_rows))
    if len(sample_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} replication samples, found {len(sample_ids)}"
        )

    labels_path = manual_dataset_root / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV missing: {labels_path}")
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        label_rows = {str(row["sample_id"]): dict(row) for row in csv.DictReader(handle)}
    if not set(sample_ids).issubset(label_rows):
        missing = sorted(set(sample_ids) - set(label_rows))
        raise ValueError(f"Clean source labels missing for: {missing}")

    rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        label = label_rows[sample_id]
        source_path = manual_dataset_root / str(label["original_path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Clean source image missing: {source_path}")
        source_sha256 = sha256_file(source_path)
        row = {
            "sample_id": sample_id,
            "source_image_path": str(source_path.resolve()),
            "target_imagenet_index": int(label["class_index"]),
            "target_wnid": str(label["wnid"]),
            "target_class_name": str(label["class_name"]),
            "source_sha256": source_sha256,
        }
        rows.append(row)
        signature_rows.append(dict(row))

    digest = hashlib.sha256()
    digest.update(
        json.dumps(signature_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return rows, digest.hexdigest()


class CleanReferenceDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)
        self.transform = paper_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = rgb_loader(str(self.rows[index]["source_image_path"]))
        return self.transform(image), index


def _valid_result(
    path: Path,
    model_key: str,
    dataset_sha256: str,
    expected_total: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    records = payload.get("records")
    return all(
        (
            payload.get("status") == "complete",
            payload.get("model") == model_key,
            payload.get("dataset_sha256") == dataset_sha256,
            payload.get("total") == expected_total,
            isinstance(records, list),
            len(records) == expected_total if isinstance(records, list) else False,
        )
    )


def evaluate_one_model(
    model: torch.nn.Module,
    model_key: str,
    dataset: Dataset,
    source_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    workers: int,
    closed_wnids: Sequence[str],
    output_indices: Sequence[int],
    index_to_name: Mapping[int, str],
) -> tuple[list[dict[str, Any]], float]:
    closed_position = {wnid: index for index, wnid in enumerate(closed_wnids)}
    output_index_tensor = torch.tensor(output_indices, dtype=torch.long, device=device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, Any]] = []
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for images, row_indices in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise ValueError(
                    f"{model_key} returned {tuple(logits.shape)}; expected [N, 1000]"
                )
            probabilities = torch.softmax(
                logits.index_select(1, output_index_tensor), dim=1
            )
            confidences, predictions = probabilities.max(dim=1)
            for row_index, prediction, confidence in zip(
                row_indices.tolist(),
                predictions.detach().cpu().tolist(),
                confidences.detach().cpu().tolist(),
            ):
                source = source_rows[int(row_index)]
                target_closed = closed_position[str(source["target_wnid"])]
                expected_imagenet = int(output_indices[target_closed])
                target_imagenet = int(source["target_imagenet_index"])
                if target_imagenet != expected_imagenet:
                    raise ValueError(
                        f"Class-index mismatch for {source['sample_id']}: "
                        f"labels={target_imagenet}, class_index_json={expected_imagenet}"
                    )
                predicted_imagenet = int(output_indices[int(prediction)])
                record = {
                    "model": model_key,
                    "sample_id": str(source["sample_id"]),
                    "source_image_path": str(source["source_image_path"]),
                    "target_imagenet_index": target_imagenet,
                    "target_closed_index": target_closed,
                    "target_wnid": str(source["target_wnid"]),
                    "target_class_name": str(source["target_class_name"]),
                    "top1_imagenet_index": predicted_imagenet,
                    "top1_closed_index": int(prediction),
                    "top1_wnid": str(closed_wnids[int(prediction)]),
                    "top1_class_name": str(index_to_name[predicted_imagenet]),
                    "top1_confidence": float(confidence),
                    "correct": bool(int(prediction) == target_closed),
                }
                records.append({field: record[field] for field in PREDICTION_FIELDS})
    return records, time.time() - started


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device(args.device)
    model_keys = parse_model_keys(args.models)
    source_rows, dataset_sha256 = load_clean_rows(
        args.manual_dataset_root, args.capture_root
    )
    dataset = CleanReferenceDataset(source_rows)
    closed_wnids, output_indices, index_to_name = closed_class_metadata(
        args.reference_root, args.class_index_json
    )

    predictions_dir = args.result_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for spec in iter_specs(model_keys):
        json_path = predictions_dir / f"{spec.key}.json"
        csv_path = predictions_dir / f"{spec.key}.csv"
        if (
            not args.overwrite
            and _valid_result(json_path, spec.key, dataset_sha256, len(dataset))
            and csv_path.is_file()
        ):
            print(f"Skip {spec.key}: complete clean-reference result exists")
            continue

        print(f"Load {spec.key} on {device}", flush=True)
        model, provenance = load_model(
            spec.key,
            checkpoint_dir=args.checkpoint_dir,
            dinov2_hub_ref=args.dinov2_hub_ref,
        )
        model = model.to(device)
        provenance.update(
            {
                "parameter_count": str(parameter_count(model)),
                "device": str(device),
                "platform": platform.platform(),
            }
        )
        records, elapsed = evaluate_one_model(
            model=model,
            model_key=spec.key,
            dataset=dataset,
            source_rows=source_rows,
            device=device,
            batch_size=min(spec.recommended_batch_size, len(dataset)),
            workers=args.workers,
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name=index_to_name,
        )
        correct = sum(int(row["correct"]) for row in records)
        payload = {
            "status": "complete",
            "dataset": "five_clean_replication_source_images",
            "dataset_sha256": dataset_sha256,
            "model": spec.key,
            "paper_name": spec.paper_name,
            "correct": correct,
            "total": len(records),
            "top1_accuracy": 100.0 * correct / len(records),
            "protocol": {
                "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
                "resize": 256,
                "crop": 224,
                "interpolation": "PIL bilinear",
                "seed": args.seed,
            },
            "model_provenance": provenance,
            "elapsed_seconds": elapsed,
            "records": records,
        }
        atomic_csv_dump(records, csv_path)
        atomic_json_dump(payload, json_path)
        print(
            f"{spec.key}: {correct}/{len(records)} "
            f"({payload['top1_accuracy']:.2f}%)",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for spec in iter_specs(model_keys):
        payload_path = predictions_dir / f"{spec.key}.json"
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not _valid_result(payload_path, spec.key, dataset_sha256, len(dataset)):
            raise ValueError(f"Invalid completed result: {payload_path}")
        summary_rows.append(
            {
                "model": spec.key,
                "paper_name": spec.paper_name,
                "correct": int(payload["correct"]),
                "total": int(payload["total"]),
                "top1_accuracy": float(payload["top1_accuracy"]),
            }
        )
        all_records.extend(payload["records"])

    atomic_csv_dump(summary_rows, args.result_root / "summary.csv")
    atomic_csv_dump(all_records, args.result_root / "per_image_predictions.csv")
    summary = {
        "status": "complete",
        "dataset": "five_clean_replication_source_images",
        "dataset_sha256": dataset_sha256,
        "sample_count": len(source_rows),
        "sample_ids": [str(row["sample_id"]) for row in source_rows],
        "model_count": len(model_keys),
        "models": list(model_keys),
        "protocol": {
            "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
            "resize": 256,
            "crop": 224,
            "interpolation": "PIL bilinear",
        },
        "results": summary_rows,
    }
    atomic_json_dump(summary, args.result_root / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
