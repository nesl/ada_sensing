from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import ExifTags, Image

from .audit_replication import audit_predictions
from .replication import (
    EXPECTED_CAPTURE_COUNT,
    REPLICATION_MODEL_KEYS,
    atomic_csv_dump,
    atomic_json_dump,
    default_cropped_root,
    default_result_root,
    default_source_root,
    load_cropped_manifest,
    load_source_capture_rows,
    parameter_key,
    sha256_file,
)


ZOOM_IDS = ("z001", "z002")
AE_PARAMETER_KEYS = ("ae_01", "ae_02", "ae_03")
MANUAL_PARAMETER_KEYS = tuple(f"p{index:03d}" for index in range(1, 28))
ALL_PARAMETER_KEYS = AE_PARAMETER_KEYS + MANUAL_PARAMETER_KEYS
EXPECTED_CONDITIONS_PER_ZOOM = 10
EXPECTED_AE_PER_ZOOM = 30
EXPECTED_MANUAL_PER_CONDITION = 27
EXIF_IFD_TAG = 34665


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate table-only analysis for the replicated capture experiment."
    )
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--dataset-root", type=Path, default=default_cropped_root())
    parser.add_argument("--result-root", type=Path, default=default_result_root())
    parser.add_argument("--models", default=",".join(REPLICATION_MODEL_KEYS))
    return parser.parse_args()


def _model_keys(value: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(keys) - set(REPLICATION_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown replication models: {unknown}")
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Model list must be non-empty and unique")
    return keys


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_prediction_payloads(
    result_root: Path,
    dataset_root: Path,
    model_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    audit = audit_predictions(dataset_root, result_root, model_keys)
    if audit["status"] != "complete":
        raise ValueError(
            "Replication predictions are incomplete or invalid: "
            f"{audit['incomplete_models']}"
        )
    payloads: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        payload = _read_json(result_root / "predictions" / f"{model_key}.json")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != EXPECTED_CAPTURE_COUNT:
            raise ValueError(f"{model_key}: expected 600 records")
        payloads[model_key] = payload
    return payloads


def _accuracy(correct: float, total: int) -> float:
    return 100.0 * float(correct) / total


def model_top1_rows(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    by_zoom: list[dict[str, Any]] = []
    for model_key, payload in payloads.items():
        records = list(payload["records"])
        correct = sum(int(bool(row["correct"])) for row in records)
        overall.append(
            {
                "model": model_key,
                "paper_name": str(payload.get("paper_name", model_key)),
                "correct": correct,
                "total": len(records),
                "top1_accuracy": _accuracy(correct, len(records)),
            }
        )
        for zoom_id in ZOOM_IDS:
            subset = [row for row in records if row["zoom_id"] == zoom_id]
            zoom_correct = sum(int(bool(row["correct"])) for row in subset)
            if len(subset) != EXPECTED_CAPTURE_COUNT // 2:
                raise ValueError(
                    f"{model_key}/{zoom_id}: expected 300 records, found {len(subset)}"
                )
            by_zoom.append(
                {
                    "model": model_key,
                    "paper_name": str(payload.get("paper_name", model_key)),
                    "zoom_id": zoom_id,
                    "correct": zoom_correct,
                    "total": len(subset),
                    "top1_accuracy": _accuracy(zoom_correct, len(subset)),
                }
            )
    return overall, by_zoom


def _condition_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["sample_id"]), str(row["zoom_id"]), str(row["light_id"])


def _parameter_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in records:
        key = str(row["parameter_key"])
        current = {
            "parameter_key": key,
            "exposure_mode": str(row["exposure_mode"]),
            "ae_shot": row.get("ae_shot"),
            "aperture": row.get("aperture"),
            "shutter_speed": row.get("shutter_speed"),
            "iso": row.get("iso"),
        }
        previous = metadata.get(key)
        if previous is not None and previous != current:
            raise ValueError(f"Inconsistent metadata for parameter {key}")
        metadata[key] = current
    if set(metadata) != set(ALL_PARAMETER_KEYS):
        raise ValueError(
            f"Expected 30 parameter keys, found {sorted(metadata)}"
        )
    return metadata


def parameter_condition_tables(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    score_tables = {zoom_id: [] for zoom_id in ZOOM_IDS}
    distribution_tables = {zoom_id: [] for zoom_id in ZOOM_IDS}
    for model_key, payload in payloads.items():
        records = list(payload["records"])
        metadata = _parameter_metadata(records)
        paper_name = str(payload.get("paper_name", model_key))
        for zoom_id in ZOOM_IDS:
            zoom_records = [row for row in records if row["zoom_id"] == zoom_id]
            condition_keys = {_condition_key(row) for row in zoom_records}
            if len(condition_keys) != EXPECTED_CONDITIONS_PER_ZOOM:
                raise ValueError(
                    f"{model_key}/{zoom_id}: expected 10 conditions, "
                    f"found {len(condition_keys)}"
                )
            correct_counts: Counter[int] = Counter()
            for parameter_key in ALL_PARAMETER_KEYS:
                subset = [
                    row
                    for row in zoom_records
                    if row["parameter_key"] == parameter_key
                ]
                if len(subset) != EXPECTED_CONDITIONS_PER_ZOOM:
                    raise ValueError(
                        f"{model_key}/{zoom_id}/{parameter_key}: expected 10 records, "
                        f"found {len(subset)}"
                    )
                if len({_condition_key(row) for row in subset}) != len(subset):
                    raise ValueError(
                        f"{model_key}/{zoom_id}/{parameter_key}: duplicate conditions"
                    )
                correct = sum(int(bool(row["correct"])) for row in subset)
                correct_counts[correct] += 1
                score_tables[zoom_id].append(
                    {
                        "model": model_key,
                        "paper_name": paper_name,
                        "zoom_id": zoom_id,
                        **metadata[parameter_key],
                        "correct_conditions": correct,
                        "total_conditions": EXPECTED_CONDITIONS_PER_ZOOM,
                        "top1_accuracy": _accuracy(
                            correct, EXPECTED_CONDITIONS_PER_ZOOM
                        ),
                    }
                )
            distribution_row: dict[str, Any] = {
                "model": model_key,
                "paper_name": paper_name,
                "zoom_id": zoom_id,
            }
            for correct in range(EXPECTED_CONDITIONS_PER_ZOOM + 1):
                distribution_row[f"parameters_correct_{correct}"] = correct_counts[correct]
            distribution_row["parameter_total"] = sum(correct_counts.values())
            if distribution_row["parameter_total"] != len(ALL_PARAMETER_KEYS):
                raise AssertionError("Parameter distribution does not sum to 30")
            distribution_tables[zoom_id].append(distribution_row)
    return score_tables, distribution_tables


def downstream_baseline_tables(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    summary_tables = {zoom_id: [] for zoom_id in ZOOM_IDS}
    condition_tables = {zoom_id: [] for zoom_id in ZOOM_IDS}
    for model_key, payload in payloads.items():
        records = list(payload["records"])
        paper_name = str(payload.get("paper_name", model_key))
        for zoom_id in ZOOM_IDS:
            grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
            for row in records:
                if row["zoom_id"] == zoom_id:
                    grouped[_condition_key(row)].append(row)
            if len(grouped) != EXPECTED_CONDITIONS_PER_ZOOM:
                raise ValueError(f"{model_key}/{zoom_id}: expected 10 conditions")

            ae_correct = 0
            lens_correct = 0
            oracle_s_correct = 0
            random_expected_correct = 0.0
            manual_parameter_hits: dict[str, list[int]] = {
                key: [] for key in MANUAL_PARAMETER_KEYS
            }

            for condition_key in sorted(grouped):
                candidates = grouped[condition_key]
                ae = sorted(
                    (row for row in candidates if row["exposure_mode"] == "auto"),
                    key=lambda row: str(row["parameter_key"]),
                )
                manual = sorted(
                    (row for row in candidates if row["exposure_mode"] == "manual"),
                    key=lambda row: str(row["parameter_key"]),
                )
                if [str(row["parameter_key"]) for row in ae] != list(AE_PARAMETER_KEYS):
                    raise ValueError(f"{model_key}/{condition_key}: invalid AE candidates")
                if [str(row["parameter_key"]) for row in manual] != list(
                    MANUAL_PARAMETER_KEYS
                ):
                    raise ValueError(
                        f"{model_key}/{condition_key}: invalid manual candidates"
                    )

                ae_hits = [int(bool(row["correct"])) for row in ae]
                manual_hits = [int(bool(row["correct"])) for row in manual]
                ae_correct += sum(ae_hits)
                for row, hit in zip(manual, manual_hits):
                    manual_parameter_hits[str(row["parameter_key"])].append(hit)

                # Match the reproduction acquisition baseline: Lens selects the
                # manual candidate with maximum classifier confidence. Sorting above
                # gives deterministic smallest-parameter tie breaking.
                lens_row = max(manual, key=lambda row: float(row["top1_confidence"]))
                lens_hit = int(bool(lens_row["correct"]))
                oracle_s_hit = int(any(manual_hits))
                random_mean = sum(manual_hits) / EXPECTED_MANUAL_PER_CONDITION
                lens_correct += lens_hit
                oracle_s_correct += oracle_s_hit
                random_expected_correct += random_mean

                condition_tables[zoom_id].append(
                    {
                        "model": model_key,
                        "paper_name": paper_name,
                        "zoom_id": zoom_id,
                        "sample_id": condition_key[0],
                        "light_id": condition_key[2],
                        "ae_correct_shots": sum(ae_hits),
                        "ae_total_shots": len(ae_hits),
                        "ae_mean_accuracy": _accuracy(sum(ae_hits), len(ae_hits)),
                        "lens_parameter_key": str(lens_row["parameter_key"]),
                        "lens_confidence": float(lens_row["top1_confidence"]),
                        "lens_correct": bool(lens_hit),
                        "oracle_s_correct": bool(oracle_s_hit),
                        "manual_correct_candidates": sum(manual_hits),
                        "random_expected_correct": random_mean,
                        "manual_correct_parameter_keys": ";".join(
                            str(row["parameter_key"])
                            for row in manual
                            if bool(row["correct"])
                        ),
                    }
                )

            fixed_results = [
                (key, sum(hits)) for key, hits in sorted(manual_parameter_hits.items())
            ]
            if any(len(hits) != EXPECTED_CONDITIONS_PER_ZOOM for hits in manual_parameter_hits.values()):
                raise ValueError(f"{model_key}/{zoom_id}: incomplete fixed parameters")
            oracle_f_key, oracle_f_correct = min(
                fixed_results, key=lambda item: (-item[1], item[0])
            )
            summary_tables[zoom_id].append(
                {
                    "model": model_key,
                    "paper_name": paper_name,
                    "zoom_id": zoom_id,
                    "ae_correct": ae_correct,
                    "ae_total": EXPECTED_AE_PER_ZOOM,
                    "ae_top1_accuracy": _accuracy(ae_correct, EXPECTED_AE_PER_ZOOM),
                    "lens_correct": lens_correct,
                    "lens_total": EXPECTED_CONDITIONS_PER_ZOOM,
                    "lens_top1_accuracy": _accuracy(
                        lens_correct, EXPECTED_CONDITIONS_PER_ZOOM
                    ),
                    "oracle_s_correct": oracle_s_correct,
                    "oracle_s_total": EXPECTED_CONDITIONS_PER_ZOOM,
                    "oracle_s_top1_accuracy": _accuracy(
                        oracle_s_correct, EXPECTED_CONDITIONS_PER_ZOOM
                    ),
                    "oracle_f_parameter_key": oracle_f_key,
                    "oracle_f_correct": oracle_f_correct,
                    "oracle_f_total": EXPECTED_CONDITIONS_PER_ZOOM,
                    "oracle_f_top1_accuracy": _accuracy(
                        oracle_f_correct, EXPECTED_CONDITIONS_PER_ZOOM
                    ),
                    "random_expected_correct": random_expected_correct,
                    "random_total": EXPECTED_CONDITIONS_PER_ZOOM,
                    "random_top1_accuracy": _accuracy(
                        random_expected_correct, EXPECTED_CONDITIONS_PER_ZOOM
                    ),
                }
            )
    return summary_tables, condition_tables


def _exif_value(exif: Mapping[int, Any], name: str) -> Any:
    tag = next((tag for tag, label in ExifTags.TAGS.items() if label == name), None)
    if tag is None:
        raise KeyError(f"Unknown EXIF tag name: {name}")
    return exif.get(tag)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _rational_string(value: Any) -> str:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        return f"{numerator}/{denominator}"
    return str(value)


def _shutter_label(seconds: float) -> str:
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError(f"Invalid EXIF exposure time: {seconds}")
    reciprocal = 1.0 / seconds
    rounded = round(reciprocal)
    if seconds < 1.0 and math.isclose(reciprocal, rounded, rel_tol=0.0, abs_tol=0.02):
        return f"1/{rounded}"
    return f"{seconds:.6g}s"


def extract_auto_exposure_parameters(
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = load_source_capture_rows(source_root)
    output: list[dict[str, Any]] = []
    exposure_program_names = {
        0: "Not defined",
        1: "Manual",
        2: "Normal program",
        3: "Aperture priority",
        4: "Shutter priority",
        5: "Creative program",
        6: "Action program",
        7: "Portrait mode",
        8: "Landscape mode",
    }
    exposure_mode_names = {0: "Auto exposure", 1: "Manual exposure", 2: "Auto bracket"}
    for row in source_rows:
        if row["exposure_mode"] != "auto":
            continue
        image_path = source_root / str(row["image_path"])
        with Image.open(image_path) as image:
            root_exif = image.getexif()
            exif = root_exif.get_ifd(EXIF_IFD_TAG)
        exposure_time_raw = _exif_value(exif, "ExposureTime")
        f_number = _number(_exif_value(exif, "FNumber"))
        exposure_seconds = _number(exposure_time_raw)
        iso = _exif_value(exif, "ISOSpeedRatings")
        exposure_program = _exif_value(exif, "ExposureProgram")
        actual_exposure_mode = _exif_value(exif, "ExposureMode")
        if f_number is None or exposure_seconds is None or iso is None:
            raise ValueError(f"Missing core AE EXIF settings: {image_path}")
        output.append(
            {
                "capture_key": str(row["capture_key"]),
                "sample_id": str(row["sample_id"]),
                "zoom_id": str(row["zoom_id"]),
                "light_id": str(row["light_id"]),
                "light_intensity": int(row["light_intensity"]),
                "light_percent": float(row["light_percent"]),
                "ae_shot": int(row["ae_shot"]),
                "parameter_key": parameter_key(row),
                "source_image_path": str(row["image_path"]),
                "f_number": f_number,
                "exposure_time_fraction": _rational_string(exposure_time_raw),
                "exposure_time_seconds": exposure_seconds,
                "shutter_speed": _shutter_label(exposure_seconds),
                "iso": int(iso),
                "exposure_program": int(exposure_program),
                "exposure_program_name": exposure_program_names.get(
                    int(exposure_program), "Unknown"
                ),
                "actual_exposure_mode": int(actual_exposure_mode),
                "actual_exposure_mode_name": exposure_mode_names.get(
                    int(actual_exposure_mode), "Unknown"
                ),
                "exposure_bias_ev": _number(_exif_value(exif, "ExposureBiasValue")),
                "brightness_value": _number(_exif_value(exif, "BrightnessValue")),
                "metering_mode": _exif_value(exif, "MeteringMode"),
                "focal_length_mm": _number(_exif_value(exif, "FocalLength")),
            }
        )
    if len(output) != 60:
        raise ValueError(f"Expected 60 AE captures, found {len(output)}")

    group_fields = (
        "zoom_id",
        "light_id",
        "f_number",
        "shutter_speed",
        "exposure_time_seconds",
        "iso",
        "exposure_program_name",
        "actual_exposure_mode_name",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in output:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    frequency: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        frequency.append(
            {
                **dict(zip(group_fields, key)),
                "capture_count": len(rows),
                "sample_count": len({row["sample_id"] for row in rows}),
                "samples": ";".join(sorted({str(row["sample_id"]) for row in rows})),
                "ae_shots": ";".join(
                    str(value) for value in sorted({int(row["ae_shot"]) for row in rows})
                ),
            }
        )
    return output, frequency


def _markdown_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]
) -> str:
    headers = [label for _key, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for key, _label in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    overall_rows: Sequence[Mapping[str, Any]],
    distribution_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    baseline_tables: Mapping[str, Sequence[Mapping[str, Any]]],
    ae_frequency_rows: Sequence[Mapping[str, Any]],
) -> None:
    sections = [
        "# Replicated capture analysis",
        "",
        "All accuracies use the closed 200-way label space. No plots are generated.",
        "",
        "## Overall per-model Top-1 (600 images)",
        "",
        _markdown_table(
            overall_rows,
            (
                ("paper_name", "Model"),
                ("correct", "Correct"),
                ("total", "Total"),
                ("top1_accuracy", "Top-1 (%)"),
            ),
        ),
        "",
        "## Acquisition baseline definitions",
        "",
        "- AE: mean Top-1 over all 3 AE shots per condition (30 images per zoom).",
        "- Lens: select the manual candidate with maximum closed-200 Top-1 confidence.",
        "- Oracle-S: a condition is correct if any of its 27 manual candidates is correct.",
        "- Oracle-F: best single fixed manual parameter over the 10 conditions in that zoom.",
        "- Random: expected accuracy under uniform random selection among 27 manual candidates.",
        "",
    ]
    for zoom_id in ZOOM_IDS:
        sections.extend(
            [
                f"## {zoom_id}: 30-parameter success distribution",
                "",
                _markdown_table(
                    distribution_tables[zoom_id],
                    tuple(
                        [("paper_name", "Model")]
                        + [
                            (f"parameters_correct_{correct}", str(correct))
                            for correct in range(EXPECTED_CONDITIONS_PER_ZOOM + 1)
                        ]
                        + [("parameter_total", "Total")]
                    ),
                ),
                "",
                f"## {zoom_id}: downstream Top-1 accuracy",
                "",
                _markdown_table(
                    baseline_tables[zoom_id],
                    (
                        ("paper_name", "Model"),
                        ("ae_top1_accuracy", "AE (%)"),
                        ("lens_top1_accuracy", "Lens (%)"),
                        ("oracle_s_top1_accuracy", "Oracle-S (%)"),
                        ("oracle_f_top1_accuracy", "Oracle-F (%)"),
                        ("random_top1_accuracy", "Random (%)"),
                        ("oracle_f_parameter_key", "Oracle-F parameter"),
                    ),
                ),
                "",
            ]
        )
    sections.extend(
        [
            "## Auto-exposure EXIF summary",
            "",
            f"The 60 AE captures contain {len(ae_frequency_rows)} distinct settings when grouped by zoom, lighting, aperture, shutter speed, ISO, exposure program, and exposure mode.",
            "",
            "See `auto_exposure_parameters.csv` for all captures and `auto_exposure_parameter_frequency.csv` for grouped settings.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(sections), encoding="utf-8")
    temporary.replace(path)


def run_analysis(
    source_root: Path,
    dataset_root: Path,
    result_root: Path,
    model_keys: Sequence[str],
) -> dict[str, Any]:
    payloads = load_prediction_payloads(result_root, dataset_root, model_keys)
    manifest_rows = load_cropped_manifest(dataset_root)
    overall_rows, by_zoom_rows = model_top1_rows(payloads)
    score_tables, distribution_tables = parameter_condition_tables(payloads)
    baseline_tables, condition_tables = downstream_baseline_tables(payloads)
    ae_rows, ae_frequency_rows = extract_auto_exposure_parameters(source_root)

    output_dir = result_root / "analysis"
    atomic_csv_dump(overall_rows, output_dir / "model_top1_600.csv")
    atomic_csv_dump(by_zoom_rows, output_dir / "model_top1_by_zoom.csv")
    atomic_csv_dump(ae_rows, output_dir / "auto_exposure_parameters.csv")
    atomic_csv_dump(
        ae_frequency_rows, output_dir / "auto_exposure_parameter_frequency.csv"
    )
    for zoom_id in ZOOM_IDS:
        atomic_csv_dump(
            score_tables[zoom_id],
            output_dir / f"parameter_condition_scores_{zoom_id}.csv",
        )
        atomic_csv_dump(
            distribution_tables[zoom_id],
            output_dir / f"parameter_success_distribution_{zoom_id}.csv",
        )
        atomic_csv_dump(
            baseline_tables[zoom_id],
            output_dir / f"downstream_baselines_{zoom_id}.csv",
        )
        atomic_csv_dump(
            condition_tables[zoom_id],
            output_dir / f"downstream_condition_details_{zoom_id}.csv",
        )

    summary = {
        "status": "complete",
        "protocol": {
            "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
            "conditions_per_zoom": EXPECTED_CONDITIONS_PER_ZOOM,
            "conditions": "2 samples x 5 lighting levels",
            "parameters": "3 AE shots + 27 manual settings",
            "ae": "mean Top-1 over all 3 AE shots per condition",
            "lens": "manual candidate with maximum closed-200 Top-1 confidence",
            "oracle_s": "condition correct iff any of 27 manual candidates is correct",
            "oracle_f": "best single fixed manual parameter over 10 conditions per zoom",
            "oracle_f_tie_break": "smallest parameter_key",
            "random": "expected accuracy under uniform random manual-parameter selection",
        },
        "dataset_manifest_sha256": sha256_file(dataset_root / "crop_manifest.jsonl"),
        "capture_count": len(manifest_rows),
        "model_count": len(model_keys),
        "models": list(model_keys),
        "ae_capture_count": len(ae_rows),
        "ae_grouped_setting_count": len(ae_frequency_rows),
        "overall_top1": overall_rows,
        "downstream_baselines": baseline_tables,
        "output_dir": str(output_dir.resolve()),
    }
    atomic_json_dump(summary, output_dir / "analysis_summary.json")
    write_markdown_report(
        output_dir / "report.md",
        overall_rows=overall_rows,
        distribution_tables=distribution_tables,
        baseline_tables=baseline_tables,
        ae_frequency_rows=ae_frequency_rows,
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        source_root=args.source_root,
        dataset_root=args.dataset_root,
        result_root=args.result_root,
        model_keys=_model_keys(args.models),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
