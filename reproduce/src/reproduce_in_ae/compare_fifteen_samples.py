from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .protocol import workspace_root
from .replication import REPLICATION_MODEL_KEYS, atomic_csv_dump, atomic_json_dump


SAMPLE_COUNT = 15
CAPTURE_LIGHT_COUNT = 5
DIVERSE_LIGHT_COUNT = 6
ZOOM_IDS = ("z001", "z002")
METRICS = ("ae", "lens", "oracle_s", "oracle_f", "random")
GROUP_SPECS = (
    ("replicated_capture_z001", "replicated_capture / z001", "replicated_capture", "z001"),
    ("replicated_capture_z002", "replicated_capture / z002", "replicated_capture", "z002"),
    ("dpi600_z001", "dpi600 / z001", "dpi600", "z001"),
    ("dpi600_z002", "dpi600 / z002", "dpi600", "z002"),
)
DIVERSE_GROUP_ID = "imagenet_es_diverse"
DIVERSE_GROUP_LABEL = "ImageNet-ES-Diverse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the five-group 15-sample comparison.")
    parser.add_argument("--workspace-root", type=Path, default=workspace_root())
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", default=",".join(REPLICATION_MODEL_KEYS))
    return parser.parse_args()


def _model_keys(value: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if set(keys) != set(REPLICATION_MODEL_KEYS) or len(keys) != len(REPLICATION_MODEL_KEYS):
        raise ValueError("The 15-sample report requires each of the 12 replication models exactly once")
    return keys


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return bool(value)


def _parameter_sort_key(value: str) -> tuple[str, int]:
    prefix, separator, suffix = value.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    prefix, suffix = value[:1], value[1:]
    return (prefix, int(suffix)) if suffix.isdigit() else (value, 0)


def _sample_balanced_accuracy(
    condition_rows: Sequence[Mapping[str, float | str]],
    field: str,
    *,
    expected_lights: int,
) -> float:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in condition_rows:
        by_sample[str(row["sample_id"])].append(float(row[field]))
    if len(by_sample) != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} samples, found {len(by_sample)}")
    if any(len(values) != expected_lights for values in by_sample.values()):
        raise ValueError(f"Each sample must have {expected_lights} lighting results")
    sample_means = [sum(values) / len(values) for values in by_sample.values()]
    return 100.0 * sum(sample_means) / len(sample_means)


def _aggregate_model(
    *,
    model: str,
    paper_name: str,
    ae_records: Sequence[Mapping[str, Any]],
    manual_records: Sequence[Mapping[str, Any]],
    expected_lights: int,
    ae_count: int,
    manual_count: int,
    parameter_field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ae_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    manual_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ae_records:
        light = str(row.get("light_id", row.get("lighting_id")))
        ae_by_condition[(str(row["sample_id"]), light)].append(row)
    for row in manual_records:
        light = str(row.get("light_id", row.get("lighting_id")))
        manual_by_condition[(str(row["sample_id"]), light)].append(row)
    expected_conditions = SAMPLE_COUNT * expected_lights
    if set(ae_by_condition) != set(manual_by_condition) or len(ae_by_condition) != expected_conditions:
        raise ValueError(f"{model}: incomplete or mismatched sample-lighting grid")

    parameter_hits: dict[str, list[int]] = defaultdict(list)
    for key in sorted(manual_by_condition):
        manual = manual_by_condition[key]
        if len(manual) != manual_count:
            raise ValueError(f"{model}/{key}: expected {manual_count} manual candidates")
        for row in manual:
            parameter_hits[str(row[parameter_field])].append(int(_bool(row["correct"])))
    if len(parameter_hits) != manual_count or any(
        len(hits) != expected_conditions for hits in parameter_hits.values()
    ):
        raise ValueError(f"{model}: incomplete fixed-parameter grid")
    oracle_f_key = min(
        parameter_hits,
        key=lambda key: (-sum(parameter_hits[key]), _parameter_sort_key(key)),
    )

    condition_rows: list[dict[str, Any]] = []
    for sample_id, light_id in sorted(ae_by_condition):
        ae = ae_by_condition[(sample_id, light_id)]
        manual = sorted(
            manual_by_condition[(sample_id, light_id)],
            key=lambda row: _parameter_sort_key(str(row[parameter_field])),
        )
        if len(ae) != ae_count:
            raise ValueError(f"{model}/{sample_id}/{light_id}: expected {ae_count} AE captures")
        lens = max(manual, key=lambda row: float(row["top1_confidence"]))
        fixed = next(row for row in manual if str(row[parameter_field]) == oracle_f_key)
        manual_hits = [int(_bool(row["correct"])) for row in manual]
        condition_rows.append(
            {
                "sample_id": sample_id,
                "lighting_id": light_id,
                "ae": sum(int(_bool(row["correct"])) for row in ae) / len(ae),
                "lens": int(_bool(lens["correct"])),
                "oracle_s": int(any(manual_hits)),
                "oracle_f": int(_bool(fixed["correct"])),
                "random": sum(manual_hits) / len(manual_hits),
            }
        )

    result: dict[str, Any] = {
        "model": model,
        "paper_name": paper_name,
        "sample_count": SAMPLE_COUNT,
        "lighting_count": expected_lights,
        "sample_lighting_pairs": expected_conditions,
        "ae_predictions": len(ae_records),
        "manual_predictions": len(manual_records),
        "oracle_f_parameter_key": oracle_f_key,
    }
    for metric in METRICS:
        result[f"{metric}_top1_accuracy"] = _sample_balanced_accuracy(
            condition_rows, metric, expected_lights=expected_lights
        )
    return result, condition_rows


def _macro_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": "macro_mean",
        "paper_name": f"{len(rows)}-model macro mean",
        "sample_count": SAMPLE_COUNT,
        "lighting_count": rows[0]["lighting_count"],
        "sample_lighting_pairs": rows[0]["sample_lighting_pairs"],
        "ae_predictions": "",
        "manual_predictions": "",
        "oracle_f_parameter_key": "",
    }
    for metric in METRICS:
        field = f"{metric}_top1_accuracy"
        result[field] = sum(float(row[field]) for row in rows) / len(rows)
    return result


def _capture_group(
    result_root: Path,
    dataset: str,
    zoom_id: str,
    model_keys: Sequence[str],
) -> list[dict[str, Any]]:
    analysis = _read_json(result_root / dataset / "analysis" / "analysis_summary.json")
    if analysis.get("protocol", {}).get("conditions_per_zoom") != SAMPLE_COUNT * CAPTURE_LIGHT_COUNT:
        raise ValueError(f"{dataset}: analysis is not the regenerated 15-sample analysis")
    baseline_by_model = {
        str(row["model"]): row for row in analysis["downstream_baselines"][zoom_id]
    }
    rows: list[dict[str, Any]] = []
    for model in model_keys:
        payload = _read_json(result_root / dataset / "predictions" / f"{model}.json")
        records = [row for row in payload["records"] if str(row["zoom_id"]) == zoom_id]
        ae = [row for row in records if row["exposure_mode"] == "auto"]
        manual = [row for row in records if row["exposure_mode"] == "manual"]
        result, _conditions = _aggregate_model(
            model=model,
            paper_name=str(payload.get("paper_name", model)),
            ae_records=ae,
            manual_records=manual,
            expected_lights=CAPTURE_LIGHT_COUNT,
            ae_count=3,
            manual_count=27,
            parameter_field="parameter_key",
        )
        source = baseline_by_model[model]
        for metric in METRICS:
            if abs(result[f"{metric}_top1_accuracy"] - float(source[f"{metric}_top1_accuracy"])) > 1e-9:
                raise ValueError(f"{dataset}/{zoom_id}/{model}: {metric} does not reproduce analysis")
        if result["oracle_f_parameter_key"] != source["oracle_f_parameter_key"]:
            raise ValueError(f"{dataset}/{zoom_id}/{model}: Oracle-F key does not reproduce analysis")
        rows.append(result)
    return rows


def _diverse_group(result_root: Path, model_keys: Sequence[str]) -> list[dict[str, Any]]:
    root = result_root / "comparison" / "diverse_ae_fifteen_samples"
    baseline = _read_json(root / "baselines.json")
    if baseline.get("sample_count") != SAMPLE_COUNT or baseline.get("condition_count") != SAMPLE_COUNT * DIVERSE_LIGHT_COUNT:
        raise ValueError("ImageNet-ES-Diverse baselines are not complete for 15 samples")
    baseline_by_model = {str(row["model"]): row for row in baseline["results"]}
    rows: list[dict[str, Any]] = []
    for model in model_keys:
        ae_payload = _read_json(root / "predictions" / f"{model}.json")
        manual_payload = _read_json(root / "manual_predictions" / f"{model}.json")
        if len(ae_payload.get("records", [])) != 450 or len(manual_payload.get("records", [])) != 2430:
            raise ValueError(f"{model}: expected 450 AE and 2,430 manual predictions")
        result, _conditions = _aggregate_model(
            model=model,
            paper_name=str(manual_payload.get("paper_name", model)),
            ae_records=ae_payload["records"],
            manual_records=manual_payload["records"],
            expected_lights=DIVERSE_LIGHT_COUNT,
            ae_count=5,
            manual_count=27,
            parameter_field="parameter_key",
        )
        source = baseline_by_model[model]
        for metric in METRICS:
            if abs(result[f"{metric}_top1_accuracy"] - float(source[f"{metric}_top1_accuracy"])) > 1e-9:
                raise ValueError(f"ImageNet-ES-Diverse/{model}: {metric} does not reproduce baselines")
        rows.append(result)
    return rows


def _original_rows(result_root: Path, model_keys: Sequence[str]) -> list[dict[str, Any]]:
    root = result_root / "comparison" / "clean_reference_fifteen_samples"
    summary = _read_json(root / "summary.json")
    by_model = {str(row["model"]): row for row in summary["results"]}
    rows: list[dict[str, Any]] = []
    for model in model_keys:
        payload = _read_json(root / "predictions" / f"{model}.json")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != SAMPLE_COUNT:
            raise ValueError(f"Original images/{model}: expected exactly 15 predictions")
        source = by_model[model]
        correct = sum(int(_bool(row["correct"])) for row in records)
        if correct != int(source["correct"]):
            raise ValueError(f"Original images/{model}: summary does not reproduce predictions")
        rows.append(
            {
                "model": model,
                "paper_name": str(source["paper_name"]),
                "correct": correct,
                "total": SAMPLE_COUNT,
                "correct_over_15": f"{correct}/15",
                "top1_accuracy": 100.0 * correct / SAMPLE_COUNT,
            }
        )
    macro_correct = sum(float(row["correct"]) for row in rows) / len(rows)
    rows.append(
        {
            "model": "macro_mean",
            "paper_name": f"{len(rows)}-model macro mean",
            "correct": macro_correct,
            "total": SAMPLE_COUNT,
            "correct_over_15": f"{macro_correct:.4f}/15",
            "top1_accuracy": sum(float(row["top1_accuracy"]) for row in rows) / len(rows),
        }
    )
    return rows


def _resolution(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Resolution source image missing: {path}")
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def _resolution_text(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def _roi_rows(workspace: Path) -> list[dict[str, Any]]:
    labels_path = workspace / "data" / "replication" / "manual_dataset" / "labels.csv"
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    if len(labels) != SAMPLE_COUNT or len({row["sample_id"] for row in labels}) != SAMPLE_COUNT:
        raise ValueError("ROI resolution table requires exactly 15 unique labeled samples")

    pipeline_sizes: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    for _group_id, _label, dataset, zoom_id in GROUP_SPECS:
        path = workspace / "data" / "replication" / f"{dataset}_roi" / "crop_manifest.jsonl"
        sizes: dict[str, set[tuple[int, int]]] = defaultdict(set)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if str(row["zoom_id"]) == zoom_id:
                    width, height = row["cropped_image_size"]
                    sizes[str(row["sample_id"])].add((int(width), int(height)))
        if len(sizes) != SAMPLE_COUNT or any(len(values) != 1 for values in sizes.values()):
            raise ValueError(f"{dataset}/{zoom_id}: ROI size must be stable for each of 15 samples")
        pipeline_sizes[(dataset, zoom_id)] = {
            sample_id: next(iter(values)) for sample_id, values in sizes.items()
        }

    manifest_items = json.loads(
        (workspace / "data" / "ImageNet-ES-Diverse" / "manifest_all.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(manifest_items, list):
        raise ValueError("ImageNet-ES-Diverse manifest must be a JSON list")
    korean_paths: dict[str, list[Path]] = defaultdict(list)
    requested = {str(row["sample_id"]) for row in labels}
    for item in manifest_items:
        candidates = item.get("candidates") or []
        if not candidates:
            continue
        sample_id = Path(str(candidates[0]["path"])).stem
        if sample_id in requested:
            korean_paths[sample_id].extend(Path(str(row["path"])) for row in candidates)

    output: list[dict[str, Any]] = []
    manual_root = labels_path.parent
    ae_root = (
        workspace
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure"
    )
    for label in labels:
        sample_id = str(label["sample_id"])
        paths = list(korean_paths[sample_id])
        paths.extend(
            ae_root
            / light_id
            / f"param_{parameter}"
            / str(label["wnid"])
            / f"{sample_id}.JPEG"
            for light_id in ("l1", "l2", "l3", "l4", "l6", "l7")
            for parameter in range(1, 6)
        )
        if len(paths) != DIVERSE_LIGHT_COUNT * (27 + 5):
            raise ValueError(f"{sample_id}: expected 192 Korean capture images")
        korean_sizes = {_resolution(path) for path in paths}
        if len(korean_sizes) != 1:
            raise ValueError(f"{sample_id}: Korean dataset resolution is not stable")
        output.append(
            {
                "Sample ID": sample_id,
                "Original Sample from ImageNet": _resolution_text(
                    _resolution(manual_root / str(label["original_path"]))
                ),
                "korean dataset": _resolution_text(next(iter(korean_sizes))),
                "600dpi Zoom1": _resolution_text(pipeline_sizes[("dpi600", "z001")][sample_id]),
                "600dpi Zoom2": _resolution_text(pipeline_sizes[("dpi600", "z002")][sample_id]),
                "Replicated Zoom1": _resolution_text(
                    pipeline_sizes[("replicated_capture", "z001")][sample_id]
                ),
                "Replicated Zoom2": _resolution_text(
                    pipeline_sizes[("replicated_capture", "z002")][sample_id]
                ),
            }
        )
    if len(output) != SAMPLE_COUNT or any(len(row) != 7 for row in output):
        raise AssertionError("Expected a 15-row, 7-column ROI resolution table")
    return output


def _ae_frequency_rows(result_root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_id, _label, dataset, zoom_id in GROUP_SPECS:
        path = result_root / dataset / "analysis" / "auto_exposure_parameters.csv"
        grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["zoom_id"] == zoom_id:
                    grouped[(row["f_number"], row["shutter_speed"], int(row["iso"]))].append(row)
        group_total = 0
        for (aperture, shutter, iso), rows in sorted(grouped.items()):
            group_total += len(rows)
            samples = sorted({row["sample_id"] for row in rows})
            lights = sorted({row["light_id"] for row in rows})
            shots = sorted({int(row["ae_shot"]) for row in rows})
            output.append(
                {
                    "group": group_id,
                    "aperture": aperture,
                    "shutter_speed": shutter,
                    "iso": iso,
                    "frequency": len(rows),
                    "sample_count": len(samples),
                    "sample_coverage": ";".join(samples),
                    "light_count": len(lights),
                    "light_coverage": ";".join(lights),
                    "ae_shot_count": len(shots),
                    "ae_shot_coverage": ";".join(map(str, shots)),
                }
            )
        if group_total != 225:
            raise ValueError(f"{group_id}: expected 225 AE captures, found {group_total}")
    return output


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row[key]
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    groups: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]],
    original_rows: Sequence[Mapping[str, Any]],
) -> None:
    sections = [
        "# Fifteen-sample across-lighting analysis",
        "",
        "For each model, captures are averaged across all lighting levels within each sample, then the 15 sample means are averaged equally. AE pools all AE captures; sample-lighting pairs are calculation and audit units, not final conditions. The macro row is an equal mean over the 12 model-level accuracies.",
        "",
    ]
    columns = (("paper_name", "Model"),) + tuple(
        (f"{metric}_top1_accuracy", label)
        for metric, label in zip(METRICS, ("AE (%)", "Lens (%)", "Oracle-S (%)", "Oracle-F (%)", "Random (%)"))
    )
    for _group_id, label, rows in groups:
        sections.extend((f"## {label}", "", _markdown_table(rows, columns), ""))
    sections.extend(
        (
            "## Original images (15 samples)",
            "",
            _markdown_table(
                original_rows,
                (("paper_name", "Model"), ("correct_over_15", "Correct / 15"), ("top1_accuracy", "Top-1 (%)")),
            ),
            "",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(sections), encoding="utf-8")
    temporary.replace(path)


def run_comparison(
    workspace: Path,
    result_root: Path,
    output_dir: Path,
    model_keys: Sequence[str],
) -> dict[str, Any]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for group_id, label, dataset, zoom_id in GROUP_SPECS:
        model_rows = _capture_group(result_root, dataset, zoom_id, model_keys)
        groups.append((group_id, label, [*model_rows, _macro_row(model_rows)]))
    diverse_models = _diverse_group(result_root, model_keys)
    groups.append((DIVERSE_GROUP_ID, DIVERSE_GROUP_LABEL, [*diverse_models, _macro_row(diverse_models)]))

    output_dir.mkdir(parents=True, exist_ok=True)
    group_dir = output_dir / "group_results"
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_id, _label, rows in groups:
        if len(rows) != 13:
            raise AssertionError(f"{group_id}: expected 13 result rows")
        atomic_csv_dump(rows, group_dir / f"{group_id}.csv")

    original_rows = _original_rows(result_root, model_keys)
    roi_rows = _roi_rows(workspace)
    frequency_rows = _ae_frequency_rows(result_root)
    atomic_csv_dump(original_rows, output_dir / "original_images_results.csv")
    atomic_csv_dump(roi_rows, output_dir / "roi_resolution_15x4.csv")
    atomic_csv_dump(frequency_rows, output_dir / "ae_parameter_frequency_by_pipeline.csv")
    _write_report(output_dir / "report.md", groups, original_rows)

    summary = {
        "status": "complete",
        "aggregation": "mean lighting results within sample, then equal mean over 15 samples",
        "model_count": len(model_keys),
        "models": list(model_keys),
        "groups": {
            group_id: {
                "label": label,
                "models": {str(row["model"]): dict(row) for row in rows[:-1]},
                "macro_mean": dict(rows[-1]),
            }
            for group_id, label, rows in groups
        },
        "original_images": {
            "models": {str(row["model"]): dict(row) for row in original_rows[:-1]},
            "macro_mean": dict(original_rows[-1]),
        },
        "validation": {
            "group_count": len(groups),
            "rows_per_group": 13,
            "diverse_predictions_per_model": {"ae": 450, "manual": 2430},
            "original_predictions_per_model": 15,
            "roi_rows": len(roi_rows),
            "roi_columns": len(roi_rows[0]),
            "ae_captures_per_pipeline": 225,
        },
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    result_root = args.result_root.resolve() if args.result_root else workspace / "replicate_result"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else result_root / "comparison" / "fifteen_samples_analysis"
    )
    summary = run_comparison(workspace, result_root, output_dir, _model_keys(args.models))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
