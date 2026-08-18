from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .datasets import paper_transform, rgb_loader
from .models import load_model, parameter_count
from .protocol import DINOV2_HUB_REF, MODEL_BY_KEY, iter_specs, parse_model_keys, project_root
from .replication import (
    atomic_csv_dump,
    atomic_json_dump,
    closed_class_metadata,
    default_class_index_json,
    default_cropped_root,
    default_reference_root,
    default_result_root,
    load_cropped_manifest,
    sha256_file,
)


PREDICTION_FIELDS = (
    "model",
    "source_manifest_index",
    "capture_key",
    "sample_id",
    "zoom_id",
    "light_id",
    "light_intensity",
    "light_percent",
    "exposure_mode",
    "parameter_key",
    "ae_shot",
    "parameter_id",
    "aperture",
    "shutter_speed",
    "iso",
    "cropped_image_path",
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


class ReplicationDataset(Dataset):
    def __init__(
        self,
        cropped_root: Path,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self.cropped_root = cropped_root
        self.rows = list(rows)
        self.transform = paper_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.cropped_root / str(row["cropped_image_path"])
        image = rgb_loader(str(path))
        return self.transform(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-image 200-way top-1 inference on the cropped replication dataset."
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--dataset-root", type=Path, default=default_cropped_root())
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument(
        "--class-index-json", type=Path, default=default_class_index_json()
    )
    parser.add_argument("--result-root", type=Path, default=default_result_root())
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=project_root() / "checkpoints"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2481757)
    parser.add_argument("--max-samples", type=int, default=0)
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


def _result_paths(result_root: Path, model_key: str, max_samples: int) -> tuple[Path, Path]:
    suffix = f"__smoke_{max_samples}" if max_samples else ""
    stem = f"{model_key}{suffix}"
    directory = result_root / ("smoke" if max_samples else "predictions")
    return directory / f"{stem}.json", directory / f"{stem}.csv"


def _valid_completed_result(
    path: Path,
    model_key: str,
    manifest_sha256: str,
    expected_total: int,
    is_smoke_test: bool,
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    records = payload.get("records")
    if not isinstance(records, list):
        return False
    keys = [str(row.get("capture_key", "")) for row in records]
    return all(
        (
            payload.get("status") == "complete",
            payload.get("model") == model_key,
            payload.get("dataset_manifest_sha256") == manifest_sha256,
            payload.get("total") == expected_total,
            len(records) == expected_total,
            len(set(keys)) == expected_total,
            payload.get("is_smoke_test") is is_smoke_test,
        )
    )


def prediction_rows_from_batch(
    model_key: str,
    source_rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    predicted_closed: Sequence[int],
    confidences: Sequence[float],
    closed_wnids: Sequence[str],
    output_indices: Sequence[int],
    index_to_name: Mapping[int, str],
) -> list[dict[str, Any]]:
    closed_position = {wnid: index for index, wnid in enumerate(closed_wnids)}
    output: list[dict[str, Any]] = []
    for row_index, predicted_position, confidence in zip(
        row_indices, predicted_closed, confidences
    ):
        source = source_rows[int(row_index)]
        target_wnid = str(source["wnid"])
        if target_wnid not in closed_position:
            raise ValueError(f"Target WNID is outside the 200-way label space: {target_wnid}")
        target_closed = closed_position[target_wnid]
        predicted_position = int(predicted_position)
        predicted_imagenet = int(output_indices[predicted_position])
        target_imagenet = int(source["class_index"])
        expected_target_imagenet = int(output_indices[target_closed])
        if target_imagenet != expected_target_imagenet:
            raise ValueError(
                f"Class-index mismatch for {source['capture_key']}: capture={target_imagenet}, "
                f"class_index_json={expected_target_imagenet}"
            )
        record = {
            "model": model_key,
            "source_manifest_index": int(source["source_manifest_index"]),
            "capture_key": str(source["capture_key"]),
            "sample_id": str(source["sample_id"]),
            "zoom_id": str(source["zoom_id"]),
            "light_id": str(source["light_id"]),
            "light_intensity": int(source["light_intensity"]),
            "light_percent": float(source["light_percent"]),
            "exposure_mode": str(source["exposure_mode"]),
            "parameter_key": str(source["parameter_key"]),
            "ae_shot": source.get("ae_shot"),
            "parameter_id": source.get("parameter_id"),
            "aperture": source.get("aperture"),
            "shutter_speed": source.get("shutter_speed"),
            "iso": source.get("iso"),
            "cropped_image_path": str(source["cropped_image_path"]),
            "target_imagenet_index": target_imagenet,
            "target_closed_index": target_closed,
            "target_wnid": target_wnid,
            "target_class_name": str(source["class_name"]),
            "top1_imagenet_index": predicted_imagenet,
            "top1_closed_index": predicted_position,
            "top1_wnid": str(closed_wnids[predicted_position]),
            "top1_class_name": str(index_to_name[predicted_imagenet]),
            "top1_confidence": float(confidence),
            "correct": bool(predicted_position == target_closed),
        }
        output.append({field: record[field] for field in PREDICTION_FIELDS})
    return output


def evaluate_model(
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    output_index_tensor = torch.tensor(output_indices, dtype=torch.long, device=device)
    records: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for images, row_indices in tqdm(loader, desc=f"replication:{model_key}"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise ValueError(
                    f"{model_key} returned {tuple(logits.shape)}; expected [N, 1000]"
                )
            closed_logits = logits.index_select(1, output_index_tensor)
            probabilities = torch.softmax(closed_logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            records.extend(
                prediction_rows_from_batch(
                    model_key=model_key,
                    source_rows=source_rows,
                    row_indices=row_indices.tolist(),
                    predicted_closed=prediction.detach().cpu().tolist(),
                    confidences=confidence.detach().cpu().tolist(),
                    closed_wnids=closed_wnids,
                    output_indices=output_indices,
                    index_to_name=index_to_name,
                )
            )
    return records, time.time() - started


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    source_rows = load_cropped_manifest(args.dataset_root)
    manifest_path = args.dataset_root / "crop_manifest.jsonl"
    manifest_sha256 = sha256_file(manifest_path)
    closed_wnids, output_indices, index_to_name = closed_class_metadata(
        args.reference_root, args.class_index_json
    )
    full_dataset = ReplicationDataset(args.dataset_root, source_rows)
    if args.max_samples:
        evaluated_total = min(args.max_samples, len(full_dataset))
        dataset: Dataset = Subset(full_dataset, range(evaluated_total))
    else:
        evaluated_total = len(full_dataset)
        dataset = full_dataset

    for spec in iter_specs(parse_model_keys(args.models)):
        json_path, csv_path = _result_paths(args.result_root, spec.key, args.max_samples)
        if not args.overwrite and _valid_completed_result(
            json_path,
            model_key=spec.key,
            manifest_sha256=manifest_sha256,
            expected_total=evaluated_total,
            is_smoke_test=bool(args.max_samples),
        ) and csv_path.is_file():
            print(f"Skip {spec.key}: complete result exists at {json_path}")
            continue

        print(f"Load {spec.key} on {device}")
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
        batch_size = args.batch_size or spec.recommended_batch_size
        records, elapsed = evaluate_model(
            model=model,
            model_key=spec.key,
            dataset=dataset,
            source_rows=source_rows,
            device=device,
            batch_size=batch_size,
            workers=args.workers,
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name=index_to_name,
        )
        correct = sum(int(row["correct"]) for row in records)
        payload = {
            "status": "complete",
            "model": spec.key,
            "paper_name": spec.paper_name,
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_manifest_sha256": manifest_sha256,
            "total": len(records),
            "correct": correct,
            "top1_accuracy": 100.0 * correct / max(1, len(records)),
            "is_smoke_test": bool(args.max_samples),
            "protocol": {
                "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
                "resize": 256,
                "crop": 224,
                "interpolation": "PIL bilinear",
                "top1_confidence": "softmax over the sliced 200 logits",
                "batch_size": batch_size,
                "workers": args.workers,
                "seed": args.seed,
            },
            "model_provenance": provenance,
            "elapsed_seconds": elapsed,
            "records": records,
        }
        atomic_csv_dump(records, csv_path)
        atomic_json_dump(payload, json_path)
        print(
            f"{spec.key}: {correct}/{len(records)} ({payload['top1_accuracy']:.2f}%) "
            f"-> {json_path}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
