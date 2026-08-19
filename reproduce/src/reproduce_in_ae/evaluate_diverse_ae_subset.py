from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .datasets import paper_transform, rgb_loader
from .evaluate_clean_reference import (
    _valid_result,
    evaluate_one_model,
    seed_everything,
)
from .models import load_model, parameter_count
from .protocol import DINOV2_HUB_REF, iter_specs, parse_model_keys, project_root, workspace_root
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
EXPECTED_LIGHT_IDS = ("l1", "l2", "l3", "l4", "l6", "l7")
EXPECTED_AE_PARAMETERS = tuple(f"param_{index}" for index in range(1, 6))


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the five replication samples across every original "
            "ImageNet-ES-Diverse auto-exposure setting."
        )
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
    parser.add_argument(
        "--ae-root",
        type=Path,
        default=(
            workspace
            / "data"
            / "ImageNet-ES-Diverse"
            / "es-diverse-test"
            / "auto_exposure"
        ),
    )
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument(
        "--class-index-json", type=Path, default=default_class_index_json()
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=(
            workspace
            / "replicate_result"
            / "comparison"
            / "diverse_ae_five_samples"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=project_root() / "checkpoints"
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2481757)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dinov2-hub-ref", default=DINOV2_HUB_REF)
    return parser.parse_args()


def load_subset_rows(
    manual_dataset_root: Path,
    capture_root: Path,
    ae_root: Path,
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
        labels = {str(row["sample_id"]): dict(row) for row in csv.DictReader(handle)}
    if not set(sample_ids).issubset(labels):
        missing = sorted(set(sample_ids) - set(labels))
        raise ValueError(f"ImageNet-ES-Diverse labels missing for: {missing}")

    light_ids = tuple(sorted(path.name for path in ae_root.iterdir() if path.is_dir()))
    if light_ids != EXPECTED_LIGHT_IDS:
        raise ValueError(
            f"Expected AE lighting environments {EXPECTED_LIGHT_IDS}, found {light_ids}"
        )

    rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for light_id in light_ids:
        light_root = ae_root / light_id
        parameter_ids = tuple(
            sorted(path.name for path in light_root.iterdir() if path.is_dir())
        )
        if parameter_ids != EXPECTED_AE_PARAMETERS:
            raise ValueError(
                f"{light_id}: expected AE parameters {EXPECTED_AE_PARAMETERS}, "
                f"found {parameter_ids}"
            )
        for parameter_id in parameter_ids:
            for sample_id in sample_ids:
                label = labels[sample_id]
                source_path = (
                    light_root
                    / parameter_id
                    / str(label["wnid"])
                    / Path(str(label["source_relative_path"])).name
                )
                if not source_path.is_file():
                    raise FileNotFoundError(f"AE subset image missing: {source_path}")
                row = {
                    "sample_id": sample_id,
                    "lighting_id": light_id,
                    "ae_parameter": parameter_id,
                    "setting": f"{light_id}/{parameter_id}",
                    "source_image_path": str(source_path.resolve()),
                    "target_imagenet_index": int(label["class_index"]),
                    "target_wnid": str(label["wnid"]),
                    "target_class_name": str(label["class_name"]),
                }
                rows.append(row)
                signature_rows.append({**row, "source_sha256": sha256_file(source_path)})

    digest = hashlib.sha256()
    digest.update(
        json.dumps(signature_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return rows, digest.hexdigest()


class DiverseAeSubsetDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)
        self.transform = paper_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = rgb_loader(str(self.rows[index]["source_image_path"]))
        return self.transform(image), index


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_lighting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_setting[str(row["setting"])].append(row)
        by_lighting[str(row["lighting_id"])].append(row)
        by_sample[str(row["sample_id"])].append(row)

    def summarize(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        output = []
        for key in sorted(groups):
            members = groups[key]
            correct = sum(int(row["correct"]) for row in members)
            output.append(
                {
                    "group": key,
                    "correct": correct,
                    "total": len(members),
                    "top1_accuracy": 100.0 * correct / len(members),
                }
            )
        return output

    return {
        "per_setting": summarize(by_setting),
        "per_lighting": summarize(by_lighting),
        "per_sample": summarize(by_sample),
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device(args.device)
    model_keys = parse_model_keys(args.models)
    source_rows, dataset_sha256 = load_subset_rows(
        args.manual_dataset_root, args.capture_root, args.ae_root
    )
    dataset = DiverseAeSubsetDataset(source_rows)
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
            print(f"Skip {spec.key}: complete Diverse AE subset result exists")
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
        base_records, elapsed = evaluate_one_model(
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
        records = []
        for source, prediction in zip(source_rows, base_records):
            records.append(
                {
                    "model": prediction["model"],
                    "sample_id": source["sample_id"],
                    "lighting_id": source["lighting_id"],
                    "ae_parameter": source["ae_parameter"],
                    "setting": source["setting"],
                    **{key: value for key, value in prediction.items() if key not in {"model", "sample_id"}},
                }
            )
        correct = sum(int(row["correct"]) for row in records)
        payload = {
            "status": "complete",
            "dataset": "imagenet_es_diverse_ae_five_replication_samples",
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
                "lighting_ids": list(EXPECTED_LIGHT_IDS),
                "ae_parameters": list(EXPECTED_AE_PARAMETERS),
                "seed": args.seed,
            },
            "model_provenance": provenance,
            "elapsed_seconds": elapsed,
            **_aggregate_records(records),
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

    macro_accuracy = sum(row["top1_accuracy"] for row in summary_rows) / len(
        summary_rows
    )
    total_correct = sum(row["correct"] for row in summary_rows)
    total_predictions = sum(row["total"] for row in summary_rows)
    atomic_csv_dump(summary_rows, args.result_root / "summary.csv")
    atomic_csv_dump(all_records, args.result_root / "per_image_predictions.csv")
    summary = {
        "status": "complete",
        "dataset": "imagenet_es_diverse_ae_five_replication_samples",
        "dataset_sha256": dataset_sha256,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "sample_ids": list(dict.fromkeys(str(row["sample_id"]) for row in source_rows)),
        "lighting_ids": list(EXPECTED_LIGHT_IDS),
        "ae_parameters": list(EXPECTED_AE_PARAMETERS),
        "settings_per_sample": len(EXPECTED_LIGHT_IDS) * len(EXPECTED_AE_PARAMETERS),
        "images_per_model": len(dataset),
        "model_count": len(model_keys),
        "models": list(model_keys),
        "protocol": {
            "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
            "resize": 256,
            "crop": 224,
            "interpolation": "PIL bilinear",
        },
        "results": summary_rows,
        "macro_mean": {
            "model": "macro_mean",
            "paper_name": f"{len(model_keys)}-model macro mean",
            "correct": total_correct,
            "total": total_predictions,
            "top1_accuracy": macro_accuracy,
        },
    }
    atomic_json_dump(summary, args.result_root / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
