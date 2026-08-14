from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .protocol import MODEL_BY_KEY, parse_model_keys, project_root
from .replication import (
    EXPECTED_CAPTURE_COUNT,
    EXPECTED_CLOSED_CLASSES,
    REPLICATION_MODEL_KEYS,
    atomic_json_dump,
    capture_key_set,
    closed_class_metadata,
    default_class_index_json,
    default_cropped_root,
    default_reference_root,
    default_result_root,
    load_cropped_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight and completeness audit for replication top-1 predictions."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_cropped_root())
    parser.add_argument("--reference-root", type=Path, default=default_reference_root())
    parser.add_argument(
        "--class-index-json", type=Path, default=default_class_index_json()
    )
    parser.add_argument("--result-root", type=Path, default=default_result_root())
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=project_root() / "checkpoints"
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--required-gpus", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _checkpoint_checks(checkpoint_dir: Path, model_keys: Sequence[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for model_key in model_keys:
        spec = MODEL_BY_KEY[model_key]
        if model_key == "resnet50_deepaugment_augmix":
            paths = [checkpoint_dir / "deepaugment_and_augmix.pth.tar"]
        elif spec.source == "torchvision":
            filename = spec.checkpoint.rsplit("/", maxsplit=1)[-1].strip()
            paths = [checkpoint_dir / "torch_hub" / "checkpoints" / filename]
        elif spec.source == "timm":
            paths = [
                checkpoint_dir
                / "huggingface"
                / "hub"
                / f"models--timm--{spec.identifier}"
            ]
        elif model_key == "dinov2_b":
            paths = [
                checkpoint_dir / "torch_hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
                checkpoint_dir / "torch_hub" / "checkpoints" / "dinov2_vitb14_linear4_head.pth",
            ]
        elif model_key == "dinov2_g":
            paths = [
                checkpoint_dir / "torch_hub" / "checkpoints" / "dinov2_vitg14_pretrain.pth",
                checkpoint_dir / "torch_hub" / "checkpoints" / "dinov2_vitg14_linear4_head.pth",
            ]
        else:
            paths = []
        checks[model_key] = {
            "paths": [str(path.resolve()) for path in paths],
            "all_present": bool(paths) and all(path.exists() for path in paths),
        }
    return checks


def run_preflight(
    dataset_root: Path,
    reference_root: Path,
    class_index_json: Path,
    checkpoint_dir: Path,
    model_keys: Sequence[str],
    required_gpus: int,
) -> dict[str, Any]:
    rows = load_cropped_manifest(dataset_root)
    closed_wnids, output_indices, _names = closed_class_metadata(
        reference_root, class_index_json
    )
    checkpoint_checks = _checkpoint_checks(checkpoint_dir, model_keys)
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    checks = {
        "dataset_count": len(rows) == EXPECTED_CAPTURE_COUNT,
        "unique_capture_keys": len(capture_key_set(rows)) == EXPECTED_CAPTURE_COUNT,
        "closed_class_count": len(closed_wnids) == EXPECTED_CLOSED_CLASSES,
        "unique_output_indices": len(set(output_indices)) == EXPECTED_CLOSED_CLASSES,
        "checkpoints_present": all(
            row["all_present"] for row in checkpoint_checks.values()
        ),
        "required_gpus_available": cuda_count >= required_gpus,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_root / "crop_manifest.jsonl"),
        "models": list(model_keys),
        "checkpoint_checks": checkpoint_checks,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": cuda_count,
            "required_device_count": required_gpus,
            "device_names": [torch.cuda.get_device_name(index) for index in range(cuda_count)],
        },
    }


def _load_prediction_payload(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file():
        return None, "missing_json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"invalid_json:{error}"
    if not isinstance(payload, dict):
        return None, "json_root_not_object"
    return payload, ""


def audit_model_result(
    model_key: str,
    predictions_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
) -> dict[str, Any]:
    json_path = predictions_dir / f"{model_key}.json"
    csv_path = predictions_dir / f"{model_key}.csv"
    payload, load_error = _load_prediction_payload(json_path)
    if payload is None:
        return {"model": model_key, "status": "incomplete", "errors": [load_error]}

    errors: list[str] = []
    records = payload.get("records")
    if not isinstance(records, list):
        records = []
        errors.append("records_not_list")
    expected_keys = [str(row["capture_key"]) for row in manifest_rows]
    actual_keys = [str(row.get("capture_key", "")) for row in records]
    if payload.get("status") != "complete":
        errors.append("status_not_complete")
    if payload.get("model") != model_key:
        errors.append("model_mismatch")
    if payload.get("is_smoke_test") is not False:
        errors.append("formal_result_marked_smoke")
    if payload.get("dataset_manifest_sha256") != manifest_sha256:
        errors.append("dataset_manifest_sha256_mismatch")
    if payload.get("total") != EXPECTED_CAPTURE_COUNT:
        errors.append("payload_total_not_600")
    if len(records) != EXPECTED_CAPTURE_COUNT:
        errors.append("record_count_not_600")
    if actual_keys != expected_keys:
        errors.append("capture_key_order_or_content_mismatch")

    if len(records) == EXPECTED_CAPTURE_COUNT:
        for index, row in enumerate(records):
            if row.get("model") != model_key:
                errors.append(f"record_{index}_model_mismatch")
                break
            predicted = row.get("top1_closed_index")
            target = row.get("target_closed_index")
            confidence = row.get("top1_confidence")
            if not isinstance(predicted, int) or not 0 <= predicted < EXPECTED_CLOSED_CLASSES:
                errors.append(f"record_{index}_invalid_top1_closed_index")
                break
            if not isinstance(target, int) or not 0 <= target < EXPECTED_CLOSED_CLASSES:
                errors.append(f"record_{index}_invalid_target_closed_index")
                break
            if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
                errors.append(f"record_{index}_invalid_confidence")
                break
            if not 0.0 <= float(confidence) <= 1.0:
                errors.append(f"record_{index}_confidence_out_of_range")
                break
            if row.get("correct") is not (predicted == target):
                errors.append(f"record_{index}_correctness_mismatch")
                break

    csv_keys: list[str] = []
    if not csv_path.is_file():
        errors.append("missing_csv")
    else:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_keys = [str(row.get("capture_key", "")) for row in csv.DictReader(handle)]
        except (OSError, csv.Error) as error:
            errors.append(f"invalid_csv:{error}")
        if csv_keys != expected_keys:
            errors.append("csv_capture_key_order_or_content_mismatch")

    correct = sum(int(bool(row.get("correct"))) for row in records)
    return {
        "model": model_key,
        "status": "complete" if not errors else "invalid",
        "errors": errors,
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "records": len(records),
        "correct": correct,
        "accuracy": 100.0 * correct / max(1, len(records)),
    }


def audit_predictions(
    dataset_root: Path,
    result_root: Path,
    model_keys: Sequence[str],
) -> dict[str, Any]:
    manifest_rows = load_cropped_manifest(dataset_root)
    manifest_sha256 = sha256_file(dataset_root / "crop_manifest.jsonl")
    predictions_dir = result_root / "predictions"
    model_results = [
        audit_model_result(
            model_key,
            predictions_dir=predictions_dir,
            manifest_rows=manifest_rows,
            manifest_sha256=manifest_sha256,
        )
        for model_key in model_keys
    ]
    complete = [row["model"] for row in model_results if row["status"] == "complete"]
    incomplete = [row["model"] for row in model_results if row["status"] != "complete"]
    return {
        "status": "complete" if not incomplete else "incomplete",
        "dataset_manifest_sha256": manifest_sha256,
        "expected_models": list(model_keys),
        "complete_models": complete,
        "incomplete_models": incomplete,
        "complete_model_count": len(complete),
        "expected_model_count": len(model_keys),
        "expected_records_per_model": EXPECTED_CAPTURE_COUNT,
        "model_results": model_results,
    }


def main() -> None:
    args = parse_args()
    model_keys = parse_model_keys(args.models)
    args.result_root.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.result_root / "diagnostics"
    preflight = run_preflight(
        dataset_root=args.dataset_root,
        reference_root=args.reference_root,
        class_index_json=args.class_index_json,
        checkpoint_dir=args.checkpoint_dir,
        model_keys=model_keys,
        required_gpus=args.required_gpus,
    )
    atomic_json_dump(preflight, diagnostics_dir / "preflight.json")
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if preflight["status"] != "pass":
        raise SystemExit(1)
    if args.preflight_only:
        return

    audit = audit_predictions(args.dataset_root, args.result_root, model_keys)
    destination = (
        diagnostics_dir / "final_audit.json"
        if audit["status"] == "complete"
        else diagnostics_dir / "progress.json"
    )
    atomic_json_dump(audit, destination)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if args.require_complete and audit["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
