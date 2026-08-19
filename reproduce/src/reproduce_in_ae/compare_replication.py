from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .analyze_replication import extract_auto_exposure_parameters
from .audit_replication import audit_predictions
from .protocol import workspace_root
from .replication import (
    REPLICATION_MODEL_KEYS,
    atomic_csv_dump,
    atomic_json_dump,
    load_cropped_manifest,
    sha256_file,
)


ZOOM_IDS = ("z001", "z002")
EXPECTED_SAMPLE_COUNT = 5
EXPECTED_LIGHT_COUNT = 5
ACCURACY_FIELDS = (
    "ae_top1_accuracy",
    "lens_top1_accuracy",
    "oracle_s_top1_accuracy",
    "oracle_f_top1_accuracy",
    "random_top1_accuracy",
)
GAP_FIELDS = (
    "ae_minus_lens_pp",
    "ae_minus_oracle_s_pp",
    "ae_minus_oracle_f_pp",
    "ae_minus_random_pp",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    print_strategy: str
    print_strategy_label: str


DATASET_SPECS = (
    DatasetSpec(
        key="replicated_capture",
        print_strategy="scaled_5p1in",
        print_strategy_label="5.1-inch scaled print",
    ),
    DatasetSpec(
        key="dpi600",
        print_strategy="bitmap_600ppi_1to1",
        print_strategy_label="600-PPI source-pixel 1:1 print",
    ),
)
DATASET_BY_KEY = {spec.key: spec for spec in DATASET_SPECS}
GROUP_ORDER = tuple(
    f"{spec.print_strategy}_{zoom_id}"
    for spec in DATASET_SPECS
    for zoom_id in ZOOM_IDS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four print-strategy/zoom replication groups."
    )
    parser.add_argument("--workspace-root", type=Path, default=workspace_root())
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
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
    if not path.is_file():
        raise FileNotFoundError(f"Required analysis output missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _group_id(spec: DatasetSpec, zoom_id: str) -> str:
    return f"{spec.print_strategy}_{zoom_id}"


def _group_label(spec: DatasetSpec, zoom_id: str) -> str:
    return f"{spec.print_strategy_label} / {zoom_id}"


def validate_dataset_inputs(
    workspace: Path,
    result_root: Path,
    model_keys: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    summaries: dict[str, dict[str, Any]] = {}
    expected_capture_keys: tuple[str, ...] | None = None
    expected_sample_ids: tuple[str, ...] | None = None
    expected_light_ids: tuple[str, ...] | None = None

    for spec in DATASET_SPECS:
        dataset_root = workspace / "data" / "replication" / f"{spec.key}_roi"
        dataset_result_root = result_root / spec.key
        manifest_rows = load_cropped_manifest(dataset_root)
        capture_keys = tuple(str(row["capture_key"]) for row in manifest_rows)
        sample_ids = tuple(sorted({str(row["sample_id"]) for row in manifest_rows}))
        light_ids = tuple(sorted({str(row["light_id"]) for row in manifest_rows}))
        zoom_ids = tuple(sorted({str(row["zoom_id"]) for row in manifest_rows}))

        if len(sample_ids) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(
                f"{spec.key}: expected {EXPECTED_SAMPLE_COUNT} samples, "
                f"found {len(sample_ids)}"
            )
        if len(light_ids) != EXPECTED_LIGHT_COUNT:
            raise ValueError(
                f"{spec.key}: expected {EXPECTED_LIGHT_COUNT} lights, "
                f"found {len(light_ids)}"
            )
        if zoom_ids != ZOOM_IDS:
            raise ValueError(f"{spec.key}: expected zooms {ZOOM_IDS}, found {zoom_ids}")

        audit = audit_predictions(dataset_root, dataset_result_root, model_keys)
        if audit["status"] != "complete":
            raise ValueError(
                f"{spec.key}: predictions are incomplete: {audit['incomplete_models']}"
            )

        summary_path = dataset_result_root / "analysis" / "analysis_summary.json"
        summary = _read_json(summary_path)
        if summary.get("status") != "complete":
            raise ValueError(f"{spec.key}: individual analysis is not complete")
        if tuple(summary.get("models", ())) != tuple(model_keys):
            raise ValueError(
                f"{spec.key}: individual analysis model order does not match request"
            )
        current_sha256 = sha256_file(dataset_root / "crop_manifest.jsonl")
        if summary.get("dataset_manifest_sha256") != current_sha256:
            raise ValueError(f"{spec.key}: individual analysis uses a stale manifest")

        if expected_capture_keys is None:
            expected_capture_keys = capture_keys
            expected_sample_ids = sample_ids
            expected_light_ids = light_ids
        else:
            if capture_keys != expected_capture_keys:
                raise ValueError(
                    f"{spec.key}: capture grid/order differs from {DATASET_SPECS[0].key}"
                )
            if sample_ids != expected_sample_ids or light_ids != expected_light_ids:
                raise ValueError(f"{spec.key}: sample/light identities do not match")
        summaries[spec.key] = summary

    assert expected_sample_ids is not None and expected_light_ids is not None
    return summaries, expected_sample_ids, expected_light_ids


def build_baseline_rows(
    summaries: Mapping[str, Mapping[str, Any]],
    model_keys: Sequence[str],
    *,
    expected_conditions: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    paper_names: dict[str, str] = {}
    for spec in DATASET_SPECS:
        if spec.key not in summaries:
            raise ValueError(f"Missing dataset summary: {spec.key}")
        tables = summaries[spec.key].get("downstream_baselines")
        if not isinstance(tables, Mapping):
            raise ValueError(f"{spec.key}: downstream_baselines is not an object")
        if set(tables) != set(ZOOM_IDS):
            raise ValueError(
                f"{spec.key}: expected baseline zooms {ZOOM_IDS}, "
                f"found {sorted(tables)}"
            )
        for zoom_id in ZOOM_IDS:
            source_rows = tables[zoom_id]
            if not isinstance(source_rows, list):
                raise ValueError(f"{spec.key}/{zoom_id}: baseline table is not a list")
            by_model = {str(row.get("model")): row for row in source_rows}
            if set(by_model) != set(model_keys) or len(source_rows) != len(model_keys):
                raise ValueError(
                    f"{spec.key}/{zoom_id}: baseline models do not match request"
                )
            for model_key in model_keys:
                source = by_model[model_key]
                paper_name = str(source["paper_name"])
                previous_name = paper_names.setdefault(model_key, paper_name)
                if previous_name != paper_name:
                    raise ValueError(f"Inconsistent paper name for {model_key}")
                if int(source["ae_total"]) != 3 * expected_conditions:
                    raise ValueError(
                        f"{spec.key}/{zoom_id}/{model_key}: invalid AE total"
                    )
                for total_field in (
                    "lens_total",
                    "oracle_s_total",
                    "oracle_f_total",
                    "random_total",
                ):
                    if int(source[total_field]) != expected_conditions:
                        raise ValueError(
                            f"{spec.key}/{zoom_id}/{model_key}: "
                            f"invalid {total_field}"
                        )
                row = {
                    "group_id": _group_id(spec, zoom_id),
                    "group_label": _group_label(spec, zoom_id),
                    "dataset": spec.key,
                    "print_strategy": spec.print_strategy,
                    "print_strategy_label": spec.print_strategy_label,
                    "zoom_id": zoom_id,
                    "model": model_key,
                    "paper_name": paper_name,
                    "conditions": expected_conditions,
                    "ae_correct": int(source["ae_correct"]),
                    "ae_total": int(source["ae_total"]),
                    "ae_top1_accuracy": float(source["ae_top1_accuracy"]),
                    "lens_correct": int(source["lens_correct"]),
                    "lens_total": int(source["lens_total"]),
                    "lens_top1_accuracy": float(source["lens_top1_accuracy"]),
                    "oracle_s_correct": int(source["oracle_s_correct"]),
                    "oracle_s_total": int(source["oracle_s_total"]),
                    "oracle_s_top1_accuracy": float(
                        source["oracle_s_top1_accuracy"]
                    ),
                    "oracle_f_parameter_key": str(
                        source["oracle_f_parameter_key"]
                    ),
                    "oracle_f_correct": int(source["oracle_f_correct"]),
                    "oracle_f_total": int(source["oracle_f_total"]),
                    "oracle_f_top1_accuracy": float(
                        source["oracle_f_top1_accuracy"]
                    ),
                    "random_expected_correct": float(
                        source["random_expected_correct"]
                    ),
                    "random_total": int(source["random_total"]),
                    "random_top1_accuracy": float(source["random_top1_accuracy"]),
                }
                row.update(
                    {
                        "ae_minus_lens_pp": (
                            row["ae_top1_accuracy"] - row["lens_top1_accuracy"]
                        ),
                        "ae_minus_oracle_s_pp": (
                            row["ae_top1_accuracy"]
                            - row["oracle_s_top1_accuracy"]
                        ),
                        "ae_minus_oracle_f_pp": (
                            row["ae_top1_accuracy"]
                            - row["oracle_f_top1_accuracy"]
                        ),
                        "ae_minus_random_pp": (
                            row["ae_top1_accuracy"] - row["random_top1_accuracy"]
                        ),
                    }
                )
                output.append(row)
    return output


def build_overview_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    model_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        grouped[str(row["group_id"])].append(row)
    if set(grouped) != set(GROUP_ORDER):
        raise ValueError(f"Expected four groups {GROUP_ORDER}, found {sorted(grouped)}")

    output: list[dict[str, Any]] = []
    for group_id in GROUP_ORDER:
        rows = grouped[group_id]
        if [str(row["model"]) for row in rows] != list(model_keys):
            raise ValueError(f"{group_id}: unstable or incomplete model order")
        first = rows[0]
        overview = {
            "group_id": group_id,
            "group_label": str(first["group_label"]),
            "dataset": str(first["dataset"]),
            "print_strategy": str(first["print_strategy"]),
            "print_strategy_label": str(first["print_strategy_label"]),
            "zoom_id": str(first["zoom_id"]),
            "model_count": len(rows),
            "conditions_per_model": int(first["conditions"]),
        }
        for field in ACCURACY_FIELDS + GAP_FIELDS:
            overview[f"macro_{field}"] = sum(float(row[field]) for row in rows) / len(
                rows
            )
        output.append(overview)
    return output


def build_delta_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    overview_rows: Sequence[Mapping[str, Any]],
    model_keys: Sequence[str],
) -> list[dict[str, Any]]:
    detail_lookup = {
        (str(row["group_id"]), str(row["model"])): row for row in baseline_rows
    }
    overview_lookup = {str(row["group_id"]): row for row in overview_rows}
    comparisons = (
        (
            "scaled_zoom_z002_minus_z001",
            "zoom_within_print_strategy",
            "scaled_5p1in_z001",
            "scaled_5p1in_z002",
        ),
        (
            "bitmap_zoom_z002_minus_z001",
            "zoom_within_print_strategy",
            "bitmap_600ppi_1to1_z001",
            "bitmap_600ppi_1to1_z002",
        ),
        (
            "z001_bitmap_minus_scaled",
            "print_strategy_within_zoom",
            "scaled_5p1in_z001",
            "bitmap_600ppi_1to1_z001",
        ),
        (
            "z002_bitmap_minus_scaled",
            "print_strategy_within_zoom",
            "scaled_5p1in_z002",
            "bitmap_600ppi_1to1_z002",
        ),
    )
    output: list[dict[str, Any]] = []
    for comparison_id, comparison_type, left_group, right_group in comparisons:
        for model_key in (*model_keys, "macro_mean"):
            if model_key == "macro_mean":
                left = overview_lookup[left_group]
                right = overview_lookup[right_group]
                paper_name = f"{len(model_keys)}-model macro mean"
                value = lambda row, field: float(row[f"macro_{field}"])
            else:
                left = detail_lookup[(left_group, model_key)]
                right = detail_lookup[(right_group, model_key)]
                paper_name = str(left["paper_name"])
                value = lambda row, field: float(row[field])
            row = {
                "comparison_id": comparison_id,
                "comparison_type": comparison_type,
                "direction": "right_minus_left",
                "left_group": left_group,
                "right_group": right_group,
                "model": model_key,
                "paper_name": paper_name,
            }
            for field in ACCURACY_FIELDS:
                row[f"delta_{field}_pp"] = value(right, field) - value(left, field)
            for field in GAP_FIELDS:
                row[f"delta_{field}"] = value(right, field) - value(left, field)
            output.append(row)
    return output


def _load_print_dimensions(
    workspace: Path, sample_ids: Sequence[str]
) -> dict[str, dict[str, tuple[float, float]]]:
    manifests = {
        "replicated_capture": (
            workspace
            / "reproduce"
            / "results"
            / "print_pdf"
            / "imagenet_letter_print.manifest.csv",
            "relative_path",
        ),
        "dpi600": (
            workspace
            / "data"
            / "replication"
            / "manual_dataset"
            / "600dpi"
            / "imagenet_letter_600dpi_1to1.manifest.csv",
            "sample_id",
        ),
    }
    requested = set(sample_ids)
    output: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset, (path, id_field) in manifests.items():
        if not path.is_file():
            raise FileNotFoundError(f"Print manifest missing: {path}")
        dimensions: dict[str, tuple[float, float]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = (
                    Path(str(row[id_field])).stem
                    if id_field == "relative_path"
                    else str(row[id_field])
                )
                if sample_id in requested:
                    dimensions[sample_id] = (
                        float(row["placed_width_in"]),
                        float(row["placed_height_in"]),
                    )
        if set(dimensions) != requested:
            missing = sorted(requested - set(dimensions))
            raise ValueError(f"{dataset}: print dimensions missing for {missing}")
        output[dataset] = dimensions
    return output


def build_protocol_rows(
    workspace: Path,
    sample_ids: Sequence[str],
    expected_conditions: int,
) -> list[dict[str, Any]]:
    print_dimensions = _load_print_dimensions(workspace, sample_ids)
    focal_by_dataset: dict[str, dict[tuple[str, str], float]] = {}
    for spec in DATASET_SPECS:
        ae_rows, _frequency = extract_auto_exposure_parameters(
            workspace / "data" / "replication" / spec.key
        )
        grouped: dict[tuple[str, str], set[float]] = defaultdict(set)
        for row in ae_rows:
            grouped[(str(row["sample_id"]), str(row["zoom_id"]))].add(
                float(row["focal_length_mm"])
            )
        focal_map: dict[tuple[str, str], float] = {}
        for key, values in grouped.items():
            if len(values) != 1:
                raise ValueError(f"{spec.key}/{key}: focal length changed within group")
            focal_map[key] = next(iter(values))
        expected_keys = {(sample_id, zoom_id) for sample_id in sample_ids for zoom_id in ZOOM_IDS}
        if set(focal_map) != expected_keys:
            raise ValueError(f"{spec.key}: incomplete AE focal-length metadata")
        focal_by_dataset[spec.key] = focal_map

    output: list[dict[str, Any]] = []
    for spec in DATASET_SPECS:
        unique_dimensions = sorted(set(print_dimensions[spec.key].values()))
        dimension_text = "; ".join(
            f"{width:.6g}x{height:.6g}" for width, height in unique_dimensions
        )
        for zoom_id in ZOOM_IDS:
            by_focal: dict[float, list[str]] = defaultdict(list)
            for sample_id in sample_ids:
                by_focal[focal_by_dataset[spec.key][(sample_id, zoom_id)]].append(
                    sample_id
                )
            focal_text = "; ".join(
                f"{focal:g} mm ({len(samples)}/{len(sample_ids)} samples)"
                for focal, samples in sorted(by_focal.items())
            )
            modal_focal = min(
                by_focal,
                key=lambda focal: (-len(by_focal[focal]), focal),
            )
            exceptions = [
                f"{sample_id}={focal:g} mm"
                for focal, samples in sorted(by_focal.items())
                for sample_id in samples
                if focal != modal_focal
            ]
            output.append(
                {
                    "group_id": _group_id(spec, zoom_id),
                    "group_label": _group_label(spec, zoom_id),
                    "dataset": spec.key,
                    "print_strategy": spec.print_strategy,
                    "zoom_id": zoom_id,
                    "sample_count": len(sample_ids),
                    "conditions": expected_conditions,
                    "unique_print_dimensions_in": dimension_text,
                    "focal_length_summary": focal_text,
                    "focal_length_exceptions": "; ".join(exceptions),
                }
            )
    return output


def load_clean_reference_rows(
    output_dir: Path,
    model_keys: Sequence[str],
    sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    summary_path = output_dir / "clean_reference" / "summary.json"
    if not summary_path.is_file():
        return []
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Five-image clean-reference evaluation is not complete")
    if tuple(summary.get("models", ())) != tuple(model_keys):
        raise ValueError("Clean-reference model order does not match comparison")
    clean_sample_ids = tuple(str(value) for value in summary.get("sample_ids", ()))
    if (
        len(clean_sample_ids) != len(sample_ids)
        or set(clean_sample_ids) != set(sample_ids)
    ):
        raise ValueError("Clean-reference sample IDs do not match comparison")
    source_rows = summary.get("results")
    if not isinstance(source_rows, list) or len(source_rows) != len(model_keys):
        raise ValueError("Clean-reference summary has an invalid result table")
    rows: list[dict[str, Any]] = []
    for model_key, source in zip(model_keys, source_rows):
        if source.get("model") != model_key:
            raise ValueError("Clean-reference result model order is unstable")
        total = int(source["total"])
        correct = int(source["correct"])
        accuracy = float(source["top1_accuracy"])
        if total != len(sample_ids) or not 0 <= correct <= total:
            raise ValueError(f"{model_key}: invalid clean-reference counts")
        if not abs(accuracy - 100.0 * correct / total) < 1e-9:
            raise ValueError(f"{model_key}: inconsistent clean-reference accuracy")
        rows.append(
            {
                "model": model_key,
                "paper_name": str(source["paper_name"]),
                "correct": correct,
                "total": total,
                "correct_total": f"{correct}/{total}",
                "top1_accuracy": accuracy,
            }
        )
    return rows


def load_diverse_ae_reference_rows(
    output_dir: Path,
    model_keys: Sequence[str],
    sample_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    summary_path = output_dir / "diverse_ae_five_samples" / "baselines.json"
    if not summary_path.is_file():
        return [], None
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("Five-sample ImageNet-ES-Diverse baselines are not complete")
    if tuple(summary.get("models", ())) != tuple(model_keys):
        raise ValueError("Diverse reference model order does not match comparison")
    reference_sample_ids = tuple(
        str(value) for value in summary.get("sample_ids", ())
    )
    if (
        len(reference_sample_ids) != len(sample_ids)
        or set(reference_sample_ids) != set(sample_ids)
    ):
        raise ValueError("Diverse reference sample IDs do not match comparison")

    expected_conditions = len(sample_ids) * 6
    if int(summary.get("condition_count", 0)) != expected_conditions:
        raise ValueError("Diverse reference does not contain the expected conditions")
    source_rows = summary.get("results")
    if not isinstance(source_rows, list) or len(source_rows) != len(model_keys):
        raise ValueError("Diverse reference summary has an invalid result table")

    rows: list[dict[str, Any]] = []
    for model_key, source in zip(model_keys, source_rows):
        if source.get("model") != model_key:
            raise ValueError("Diverse reference result model order is unstable")
        row = dict(source)
        for prefix, expected_total in (("ae", 5 * expected_conditions),) + tuple(
            (prefix, expected_conditions)
            for prefix in ("lens", "oracle_s", "oracle_f", "random")
        ):
            total = int(row[f"{prefix}_total"])
            if total != expected_total:
                raise ValueError(f"{model_key}: invalid {prefix} total")
            accuracy = float(row[f"{prefix}_top1_accuracy"])
            correct = float(
                row[
                    "random_expected_correct"
                    if prefix == "random"
                    else f"{prefix}_correct"
                ]
            )
            if not abs(accuracy - 100.0 * correct / total) < 1e-9:
                raise ValueError(f"{model_key}: inconsistent {prefix} accuracy")
        rows.append(row)

    saved_macro = summary.get("macro_mean")
    if not isinstance(saved_macro, Mapping):
        raise ValueError("Diverse reference macro mean is missing")
    macro_row = dict(saved_macro)
    for field in ACCURACY_FIELDS + GAP_FIELDS:
        expected = sum(float(row[field]) for row in rows) / len(rows)
        if not abs(float(macro_row[f"macro_{field}"]) - expected) < 1e-9:
            raise ValueError(f"Diverse reference macro {field} is inconsistent")
    return rows, macro_row


def _markdown_table(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]
) -> str:
    headers = [label for _field, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for field, _label in columns:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    protocol_rows: Sequence[Mapping[str, Any]],
    clean_reference_rows: Sequence[Mapping[str, Any]],
    diverse_ae_reference_rows: Sequence[Mapping[str, Any]],
    diverse_ae_macro_row: Mapping[str, Any] | None,
    overview_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
) -> None:
    sections = [
        "# Four-group auto-exposure/manual baseline comparison",
        "",
        "All values are closed-200 Top-1 percentages. The overview is a macro mean across the 12 models; models are not treated as independent statistical samples.",
        "",
        "## Capture groups",
        "",
        _markdown_table(
            protocol_rows,
            (
                ("group_label", "Group"),
                ("unique_print_dimensions_in", "Printed W x H (in)"),
                ("focal_length_summary", "Observed focal length"),
                ("conditions", "Conditions"),
            ),
        ),
        "",
        "The scaled-print z002 group contains one focal-length exception: `ILSVRC2012_val_00048338` was captured at 39 mm; its other four samples were captured at 70 mm. The 600-PPI z002 group used 70 mm for all five samples.",
        "",
        *(
            [
                "## Five-image clean-source reference",
                "",
                "These are the exact five original JPEGs used to create the printed targets, evaluated with the same closed-200 label space and Resize(256) / CenterCrop(224) preprocessing. This is the paired clean-source baseline for the capture groups.",
                "",
                _markdown_table(
                    clean_reference_rows,
                    (
                        ("paper_name", "Model"),
                        ("correct_total", "Correct / 5"),
                        ("top1_accuracy", "Top-1 (%)"),
                    ),
                ),
                "",
            ]
            if clean_reference_rows
            else []
        ),
        *(
            [
                "## Five-sample ImageNet-ES-Diverse acquisition baselines",
                "",
                "These are the same five samples in the original ImageNet-ES-Diverse dataset, evaluated over all six available lighting environments (`l1`, `l2`, `l3`, `l4`, `l6`, `l7`). Each model sees 30 conditions, with five AE captures and 27 manual candidates per condition, using the same closed-200 label space and Resize(256) / CenterCrop(224) preprocessing.",
                "",
                _markdown_table(
                    diverse_ae_reference_rows,
                    (
                        ("paper_name", "Model"),
                        ("ae_top1_accuracy", "AE"),
                        ("lens_top1_accuracy", "Lens"),
                        ("oracle_s_top1_accuracy", "Oracle-S"),
                        ("oracle_f_top1_accuracy", "Oracle-F"),
                        ("oracle_f_parameter_key", "Oracle-F parameter"),
                        ("random_top1_accuracy", "Random"),
                        ("ae_minus_lens_pp", "AE-Lens"),
                        ("ae_minus_oracle_s_pp", "AE-Oracle-S"),
                        ("ae_minus_oracle_f_pp", "AE-Oracle-F"),
                        ("ae_minus_random_pp", "AE-Random"),
                    ),
                ),
                "",
            ]
            if diverse_ae_reference_rows and diverse_ae_macro_row is not None
            else []
        ),
        "## Four-group overview (12-model macro mean)",
        "",
        "The first four rows are the replication groups; the fifth row is the matched five-sample reference from the original ImageNet-ES-Diverse dataset.",
        "",
        _markdown_table(
            [*overview_rows, diverse_ae_macro_row]
            if diverse_ae_macro_row is not None
            else overview_rows,
            (
                ("group_label", "Group"),
                ("macro_ae_top1_accuracy", "AE"),
                ("macro_lens_top1_accuracy", "Lens"),
                ("macro_oracle_s_top1_accuracy", "Oracle-S"),
                ("macro_oracle_f_top1_accuracy", "Oracle-F"),
                ("macro_random_top1_accuracy", "Random"),
                ("macro_ae_minus_lens_pp", "AE-Lens"),
                ("macro_ae_minus_oracle_s_pp", "AE-Oracle-S"),
                ("macro_ae_minus_oracle_f_pp", "AE-Oracle-F"),
                ("macro_ae_minus_random_pp", "AE-Random"),
            ),
        ),
        "",
        "## Macro deltas (right minus left, percentage points)",
        "",
        _markdown_table(
            [row for row in delta_rows if row["model"] == "macro_mean"],
            (
                ("comparison_id", "Comparison"),
                ("delta_ae_top1_accuracy_pp", "AE"),
                ("delta_lens_top1_accuracy_pp", "Lens"),
                ("delta_oracle_s_top1_accuracy_pp", "Oracle-S"),
                ("delta_oracle_f_top1_accuracy_pp", "Oracle-F"),
                ("delta_random_top1_accuracy_pp", "Random"),
            ),
        ),
        "",
    ]
    for group_id in GROUP_ORDER:
        rows = [row for row in baseline_rows if row["group_id"] == group_id]
        sections.extend(
            [
                f"## {rows[0]['group_label']}: per-model results",
                "",
                _markdown_table(
                    rows,
                    (
                        ("paper_name", "Model"),
                        ("ae_top1_accuracy", "AE"),
                        ("lens_top1_accuracy", "Lens"),
                        ("oracle_s_top1_accuracy", "Oracle-S"),
                        ("oracle_f_top1_accuracy", "Oracle-F"),
                        ("oracle_f_parameter_key", "Oracle-F parameter"),
                        ("random_top1_accuracy", "Random"),
                        ("ae_minus_lens_pp", "AE-Lens"),
                        ("ae_minus_oracle_s_pp", "AE-Oracle-S"),
                        ("ae_minus_oracle_f_pp", "AE-Oracle-F"),
                        ("ae_minus_random_pp", "AE-Random"),
                    ),
                ),
                "",
            ]
        )
    sections.extend(
        [
            "## Baseline definitions",
            "",
            "- AE: mean Top-1 over the three auto-exposure captures per replication condition; the original-dataset reference uses its five AE captures per condition.",
            "- Lens: the manual candidate with maximum closed-200 classifier confidence.",
            "- Oracle-S: a condition is correct if any of its 27 manual candidates is correct.",
            "- Oracle-F: the best single fixed manual parameter within that print/zoom group; ties use the smallest parameter key.",
            "- Random: expected accuracy under uniform random selection among the 27 manual candidates.",
            "",
            "The four-group comparison is descriptive and does not claim causal identification.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(sections), encoding="utf-8")
    temporary.replace(path)


def run_comparison(
    workspace: Path,
    result_root: Path,
    output_dir: Path,
    model_keys: Sequence[str],
) -> dict[str, Any]:
    summaries, sample_ids, light_ids = validate_dataset_inputs(
        workspace, result_root, model_keys
    )
    expected_conditions = len(sample_ids) * len(light_ids)
    baseline_rows = build_baseline_rows(
        summaries, model_keys, expected_conditions=expected_conditions
    )
    overview_rows = build_overview_rows(baseline_rows, model_keys)
    delta_rows = build_delta_rows(baseline_rows, overview_rows, model_keys)
    protocol_rows = build_protocol_rows(workspace, sample_ids, expected_conditions)
    clean_reference_rows = load_clean_reference_rows(
        output_dir, model_keys, sample_ids
    )
    diverse_ae_reference_rows, diverse_ae_macro_row = load_diverse_ae_reference_rows(
        output_dir, model_keys, sample_ids
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "four_group_baselines.csv"
    overview_path = output_dir / "four_group_overview.csv"
    delta_path = output_dir / "four_group_deltas.csv"
    report_path = output_dir / "report.md"
    summary_path = output_dir / "summary.json"
    atomic_csv_dump(baseline_rows, baseline_path)
    atomic_csv_dump(overview_rows, overview_path)
    atomic_csv_dump(delta_rows, delta_path)
    write_report(
        report_path,
        protocol_rows,
        clean_reference_rows,
        diverse_ae_reference_rows,
        diverse_ae_macro_row,
        overview_rows,
        baseline_rows,
        delta_rows,
    )

    summary = {
        "status": "complete",
        "label_space": "closed 200-way sorted ImageNet-ES WNIDs",
        "model_count": len(model_keys),
        "models": list(model_keys),
        "group_count": len(GROUP_ORDER),
        "groups": list(GROUP_ORDER),
        "sample_count": len(sample_ids),
        "light_count": len(light_ids),
        "conditions_per_group": expected_conditions,
        "parameters_per_condition": "3 AE + 27 manual",
        "protocol_groups": protocol_rows,
        "clean_reference": clean_reference_rows,
        "diverse_ae_reference": {
            "results": diverse_ae_reference_rows,
            "macro_mean": diverse_ae_macro_row,
        },
        "overview": overview_rows,
        "macro_deltas": [
            row for row in delta_rows if row["model"] == "macro_mean"
        ],
        "outputs": {
            "baselines_csv": str(baseline_path.resolve()),
            "overview_csv": str(overview_path.resolve()),
            "deltas_csv": str(delta_path.resolve()),
            "report": str(report_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    atomic_json_dump(summary, summary_path)
    return summary


def main() -> None:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    result_root = (
        args.result_root.resolve()
        if args.result_root is not None
        else workspace / "replicate_result"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_root / "comparison"
    )
    summary = run_comparison(
        workspace=workspace,
        result_root=result_root,
        output_dir=output_dir,
        model_keys=_model_keys(args.models),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
