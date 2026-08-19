from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .datasets import paper_transform, rgb_loader
from .evaluate_clean_reference import _valid_result, evaluate_one_model, seed_everything
from .evaluate_diverse_ae_subset import (
    EXPECTED_LIGHT_IDS,
    EXPECTED_SAMPLE_COUNT,
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


EXPECTED_MANUAL_PARAMETERS = tuple(f"param_{index}" for index in range(1, 28))
EXPECTED_CONDITIONS = EXPECTED_SAMPLE_COUNT * len(EXPECTED_LIGHT_IDS)


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all 27 manual candidates for the five replication samples "
            "in the original ImageNet-ES-Diverse dataset and combine them with "
            "the saved AE predictions to compute acquisition baselines."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "data" / "ImageNet-ES-Diverse" / "manifest_all.json",
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


def _parameter_number(value: str) -> int:
    prefix, number = value.split("_", maxsplit=1)
    if prefix != "param" or not number.isdigit():
        raise ValueError(f"Invalid parameter name: {value}")
    return int(number)


def load_manual_rows(
    manifest_path: Path,
    capture_root: Path,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    capture_rows = load_source_capture_rows(capture_root)
    sample_ids = list(dict.fromkeys(str(row["sample_id"]) for row in capture_rows))
    if len(sample_ids) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT} replication samples, found {len(sample_ids)}"
        )
    requested = set(sample_ids)

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, list):
        raise ValueError(f"Expected a JSON list: {manifest_path}")

    conditions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in manifest:
        candidates = item.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        sample_id = Path(str(candidates[0]["path"])).stem
        if sample_id not in requested:
            continue
        light_id = str(item["env"])
        key = (sample_id, light_id)
        if key in conditions:
            raise ValueError(f"Duplicate manifest condition: {key}")
        conditions[key] = item

    expected_keys = {
        (sample_id, light_id)
        for light_id in EXPECTED_LIGHT_IDS
        for sample_id in sample_ids
    }
    if set(conditions) != expected_keys:
        missing = sorted(expected_keys - set(conditions))
        extra = sorted(set(conditions) - expected_keys)
        raise ValueError(f"Manual subset condition mismatch; missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    for light_id in EXPECTED_LIGHT_IDS:
        for sample_id in sample_ids:
            item = conditions[(sample_id, light_id)]
            candidates = sorted(
                item["candidates"],
                key=lambda row: _parameter_number(str(row["meta"]["option_name"])),
            )
            parameter_names = tuple(
                str(candidate["meta"]["option_name"]) for candidate in candidates
            )
            if parameter_names != EXPECTED_MANUAL_PARAMETERS:
                raise ValueError(
                    f"{sample_id}/{light_id}: expected 27 manual parameters, "
                    f"found {parameter_names}"
                )
            for candidate in candidates:
                source_path = Path(str(candidate["path"]))
                if not source_path.is_file():
                    raise FileNotFoundError(f"Manual candidate missing: {source_path}")
                parts = source_path.parts
                try:
                    class_name = parts[-2]
                except IndexError as error:
                    raise ValueError(f"Malformed candidate path: {source_path}") from error
                parameter_key = str(candidate["meta"]["option_name"])
                row = {
                    "sample_id": sample_id,
                    "lighting_id": light_id,
                    "condition": f"{sample_id}/{light_id}",
                    "parameter_key": parameter_key,
                    "option_id": int(candidate["option_id"]),
                    "source_image_path": str(source_path.resolve()),
                    "target_imagenet_index": int(item["label"]),
                    "target_wnid": class_name,
                    "target_class_name": class_name,
                }
                rows.append(row)
                signature_rows.append({**row, "source_sha256": sha256_file(source_path)})

    digest = hashlib.sha256()
    digest.update(
        json.dumps(signature_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return rows, digest.hexdigest(), sample_ids


class DiverseManualSubsetDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)
        self.transform = paper_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = rgb_loader(str(self.rows[index]["source_image_path"]))
        return self.transform(image), index


def _accuracy(correct: float, total: int) -> float:
    return 100.0 * correct / total


def build_model_baseline(
    model_key: str,
    paper_name: str,
    ae_records: Sequence[Mapping[str, Any]],
    manual_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ae_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    manual_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ae_records:
        ae_by_condition[(str(row["sample_id"]), str(row["lighting_id"]))].append(row)
    for row in manual_records:
        manual_by_condition[(str(row["sample_id"]), str(row["lighting_id"]))].append(row)
    if set(ae_by_condition) != set(manual_by_condition) or len(ae_by_condition) != EXPECTED_CONDITIONS:
        raise ValueError(f"{model_key}: AE/manual condition grids do not match")

    ae_correct = 0
    lens_correct = 0
    oracle_s_correct = 0
    random_expected_correct = 0.0
    fixed_hits = {key: [] for key in EXPECTED_MANUAL_PARAMETERS}
    condition_rows: list[dict[str, Any]] = []
    for condition_key in sorted(ae_by_condition):
        ae = sorted(ae_by_condition[condition_key], key=lambda row: str(row["ae_parameter"]))
        manual = sorted(
            manual_by_condition[condition_key],
            key=lambda row: _parameter_number(str(row["parameter_key"])),
        )
        if len(ae) != 5 or len(manual) != len(EXPECTED_MANUAL_PARAMETERS):
            raise ValueError(f"{model_key}/{condition_key}: incomplete candidates")
        ae_hits = [int(bool(row["correct"])) for row in ae]
        manual_hits = [int(bool(row["correct"])) for row in manual]
        ae_correct += sum(ae_hits)
        for row, hit in zip(manual, manual_hits):
            fixed_hits[str(row["parameter_key"])].append(hit)

        lens_row = max(manual, key=lambda row: float(row["top1_confidence"]))
        lens_hit = int(bool(lens_row["correct"]))
        oracle_s_hit = int(any(manual_hits))
        random_mean = sum(manual_hits) / len(manual_hits)
        lens_correct += lens_hit
        oracle_s_correct += oracle_s_hit
        random_expected_correct += random_mean
        condition_rows.append(
            {
                "model": model_key,
                "paper_name": paper_name,
                "sample_id": condition_key[0],
                "lighting_id": condition_key[1],
                "ae_correct": sum(ae_hits),
                "ae_total": len(ae_hits),
                "lens_parameter_key": str(lens_row["parameter_key"]),
                "lens_correct": bool(lens_hit),
                "oracle_s_correct": bool(oracle_s_hit),
                "manual_correct_candidates": sum(manual_hits),
                "random_expected_correct": random_mean,
            }
        )

    fixed_results = [(key, sum(hits)) for key, hits in fixed_hits.items()]
    if any(len(hits) != EXPECTED_CONDITIONS for hits in fixed_hits.values()):
        raise ValueError(f"{model_key}: incomplete fixed-parameter grid")
    oracle_f_key, oracle_f_correct = min(
        fixed_results, key=lambda item: (-item[1], _parameter_number(item[0]))
    )
    ae_total = sum(len(rows) for rows in ae_by_condition.values())
    result = {
        "model": model_key,
        "paper_name": paper_name,
        "conditions": EXPECTED_CONDITIONS,
        "ae_correct": ae_correct,
        "ae_total": ae_total,
        "ae_top1_accuracy": _accuracy(ae_correct, ae_total),
        "lens_correct": lens_correct,
        "lens_total": EXPECTED_CONDITIONS,
        "lens_top1_accuracy": _accuracy(lens_correct, EXPECTED_CONDITIONS),
        "oracle_s_correct": oracle_s_correct,
        "oracle_s_total": EXPECTED_CONDITIONS,
        "oracle_s_top1_accuracy": _accuracy(oracle_s_correct, EXPECTED_CONDITIONS),
        "oracle_f_parameter_key": oracle_f_key,
        "oracle_f_correct": oracle_f_correct,
        "oracle_f_total": EXPECTED_CONDITIONS,
        "oracle_f_top1_accuracy": _accuracy(oracle_f_correct, EXPECTED_CONDITIONS),
        "random_expected_correct": random_expected_correct,
        "random_total": EXPECTED_CONDITIONS,
        "random_top1_accuracy": _accuracy(random_expected_correct, EXPECTED_CONDITIONS),
    }
    result.update(
        {
            "ae_minus_lens_pp": result["ae_top1_accuracy"] - result["lens_top1_accuracy"],
            "ae_minus_oracle_s_pp": result["ae_top1_accuracy"] - result["oracle_s_top1_accuracy"],
            "ae_minus_oracle_f_pp": result["ae_top1_accuracy"] - result["oracle_f_top1_accuracy"],
            "ae_minus_random_pp": result["ae_top1_accuracy"] - result["random_top1_accuracy"],
        }
    )
    return result, condition_rows


def build_macro_mean(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "ae_top1_accuracy",
        "lens_top1_accuracy",
        "oracle_s_top1_accuracy",
        "oracle_f_top1_accuracy",
        "random_top1_accuracy",
        "ae_minus_lens_pp",
        "ae_minus_oracle_s_pp",
        "ae_minus_oracle_f_pp",
        "ae_minus_random_pp",
    )
    macro = {
        "model": "macro_mean",
        "paper_name": f"{len(rows)}-model macro mean",
        "group_id": "imagenet_es_diverse_five_sample_reference",
        "group_label": "Original ImageNet-ES-Diverse / five-sample subset",
        "conditions_per_model": EXPECTED_CONDITIONS,
    }
    macro.update(
        {f"macro_{field}": sum(float(row[field]) for row in rows) / len(rows) for field in fields}
    )
    return macro


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    device = torch.device(args.device)
    model_keys = parse_model_keys(args.models)
    source_rows, dataset_sha256, sample_ids = load_manual_rows(
        args.manifest, args.capture_root
    )
    dataset = DiverseManualSubsetDataset(source_rows)
    closed_wnids, output_indices, index_to_name = closed_class_metadata(
        args.reference_root, args.class_index_json
    )

    predictions_dir = args.result_root / "manual_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for spec in iter_specs(model_keys):
        json_path = predictions_dir / f"{spec.key}.json"
        csv_path = predictions_dir / f"{spec.key}.csv"
        if (
            not args.overwrite
            and _valid_result(json_path, spec.key, dataset_sha256, len(dataset))
            and csv_path.is_file()
        ):
            print(f"Skip {spec.key}: complete Diverse manual subset result exists")
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
                    "condition": source["condition"],
                    "parameter_key": source["parameter_key"],
                    "option_id": source["option_id"],
                    **{
                        key: value
                        for key, value in prediction.items()
                        if key not in {"model", "sample_id"}
                    },
                }
            )
        correct = sum(int(row["correct"]) for row in records)
        payload = {
            "status": "complete",
            "dataset": "imagenet_es_diverse_manual_five_replication_samples",
            "dataset_sha256": dataset_sha256,
            "model": spec.key,
            "paper_name": spec.paper_name,
            "correct": correct,
            "total": len(records),
            "top1_accuracy": _accuracy(correct, len(records)),
            "protocol": {
                "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
                "resize": 256,
                "crop": 224,
                "interpolation": "PIL bilinear",
                "lighting_ids": list(EXPECTED_LIGHT_IDS),
                "manual_parameters": list(EXPECTED_MANUAL_PARAMETERS),
                "seed": args.seed,
            },
            "model_provenance": provenance,
            "elapsed_seconds": elapsed,
            "records": records,
        }
        atomic_csv_dump(records, csv_path)
        atomic_json_dump(payload, json_path)
        print(f"{spec.key}: {correct}/{len(records)} ({payload['top1_accuracy']:.2f}%)", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    all_manual_records: list[dict[str, Any]] = []
    for spec in iter_specs(model_keys):
        manual_path = predictions_dir / f"{spec.key}.json"
        ae_path = args.result_root / "predictions" / f"{spec.key}.json"
        with manual_path.open("r", encoding="utf-8") as handle:
            manual_payload = json.load(handle)
        with ae_path.open("r", encoding="utf-8") as handle:
            ae_payload = json.load(handle)
        if not _valid_result(manual_path, spec.key, dataset_sha256, len(dataset)):
            raise ValueError(f"Invalid completed result: {manual_path}")
        ae_records = ae_payload.get("records")
        if not isinstance(ae_records, list) or len(ae_records) != EXPECTED_CONDITIONS * 5:
            raise ValueError(f"{spec.key}: incomplete saved AE predictions")
        result, conditions = build_model_baseline(
            spec.key, spec.paper_name, ae_records, manual_payload["records"]
        )
        result_rows.append(result)
        condition_rows.extend(conditions)
        all_manual_records.extend(manual_payload["records"])

    macro_mean = build_macro_mean(result_rows)
    atomic_csv_dump(result_rows, args.result_root / "baselines.csv")
    atomic_csv_dump(condition_rows, args.result_root / "baseline_condition_details.csv")
    atomic_csv_dump(all_manual_records, args.result_root / "manual_per_image_predictions.csv")
    summary = {
        "status": "complete",
        "dataset": "imagenet_es_diverse_five_sample_acquisition_baselines",
        "manual_dataset_sha256": dataset_sha256,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "lighting_ids": list(EXPECTED_LIGHT_IDS),
        "condition_count": EXPECTED_CONDITIONS,
        "ae_images_per_condition": 5,
        "manual_candidates_per_condition": len(EXPECTED_MANUAL_PARAMETERS),
        "model_count": len(model_keys),
        "models": list(model_keys),
        "results": result_rows,
        "macro_mean": macro_mean,
    }
    atomic_json_dump(summary, args.result_root / "baselines.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_evaluation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
