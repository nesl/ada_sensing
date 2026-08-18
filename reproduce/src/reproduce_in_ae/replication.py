from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import MODEL_SPECS, project_root, workspace_root


EXPECTED_CAPTURE_COUNT = 600
EXPECTED_AE_COUNT = 60
EXPECTED_MANUAL_COUNT = 540
EXPECTED_CLOSED_CLASSES = 200
EXPECTED_ZOOM_IDS = ("z001", "z002")
EXPECTED_LIGHT_IDS = ("b010", "b200", "b500", "b700", "b1000")
EXPECTED_PARAMETER_KEYS = tuple(f"ae_{index:02d}" for index in range(1, 4)) + tuple(
    f"p{index:03d}" for index in range(1, 28)
)
REPLICATION_MODEL_KEYS = tuple(spec.key for spec in MODEL_SPECS)


def default_source_root() -> Path:
    return workspace_root() / "data" / "replication" / "replicated_capture"


def default_cropped_root() -> Path:
    return workspace_root() / "data" / "replication" / "replicated_capture_roi"


def default_result_root() -> Path:
    return workspace_root() / "replicate_result"


def default_roi_config() -> Path:
    return project_root() / "configs" / "replication_roi.json"


def default_reference_root() -> Path:
    return (
        workspace_root()
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "sampled_tin_no_resize2"
    )


def default_class_index_json() -> Path:
    return workspace_root() / "data" / "ImageNet-ES-Diverse" / "imagenet_class_index.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl_dump(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv_dump(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def parameter_key(row: Mapping[str, Any]) -> str:
    mode = str(row.get("exposure_mode"))
    if mode == "auto":
        shot = row.get("ae_shot")
        if not isinstance(shot, int) or shot not in (1, 2, 3):
            raise ValueError(f"Invalid AE shot in {row.get('capture_key')}: {shot}")
        return f"ae_{shot:02d}"
    if mode == "manual":
        value = str(row.get("parameter_id", ""))
        if len(value) != 4 or not value.startswith("p") or not value[1:].isdigit():
            raise ValueError(f"Invalid manual parameter in {row.get('capture_key')}: {value}")
        return value
    raise ValueError(f"Unknown exposure_mode in {row.get('capture_key')}: {mode}")


def deduplicate_capture_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the final successful attempt for each key without changing key order."""
    output: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    duplicates = 0
    for source in rows:
        row = dict(source)
        key = str(row.get("capture_key", ""))
        if not key:
            raise ValueError("Every capture must contain capture_key")
        if key in positions:
            output[positions[key]] = row
            duplicates += 1
        else:
            positions[key] = len(output)
            output.append(row)
    return output, duplicates


def validate_capture_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_count: int | None = None,
) -> dict[str, Any]:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} captures, found {len(rows)}")
    capture_keys = [str(row.get("capture_key", "")) for row in rows]
    if any(not key for key in capture_keys):
        raise ValueError("Every capture must contain capture_key")
    if len(set(capture_keys)) != len(capture_keys):
        duplicates = sorted(key for key, count in Counter(capture_keys).items() if count > 1)
        raise ValueError(f"Duplicate capture keys: {duplicates[:5]}")
    statuses = Counter(str(row.get("capture_status")) for row in rows)
    if statuses != {"captured": len(rows)}:
        raise ValueError(f"Unexpected capture statuses: {dict(statuses)}")
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    zoom_ids = sorted({str(row["zoom_id"]) for row in rows})
    light_ids = sorted({str(row["light_id"]) for row in rows})
    if set(zoom_ids) != set(EXPECTED_ZOOM_IDS):
        raise ValueError(f"Unexpected zoom IDs: {zoom_ids}")
    if set(light_ids) != set(EXPECTED_LIGHT_IDS):
        raise ValueError(f"Unexpected light IDs: {light_ids}")

    condition_parameters: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        condition = (str(row["sample_id"]), str(row["zoom_id"]), str(row["light_id"]))
        condition_parameters.setdefault(condition, set()).add(parameter_key(row))
    expected_conditions = len(sample_ids) * len(zoom_ids) * len(light_ids)
    if len(condition_parameters) != expected_conditions:
        raise ValueError(
            f"Expected {expected_conditions} sample/zoom/light conditions, "
            f"found {len(condition_parameters)}"
        )
    expected_parameters = set(EXPECTED_PARAMETER_KEYS)
    incomplete = [
        condition
        for condition, parameters in condition_parameters.items()
        if parameters != expected_parameters
    ]
    if incomplete:
        raise ValueError(f"Incomplete parameter grids: {sorted(incomplete)[:5]}")

    inferred_count = expected_conditions * len(EXPECTED_PARAMETER_KEYS)
    if len(rows) != inferred_count:
        raise ValueError(f"Expected complete grid of {inferred_count} captures, found {len(rows)}")
    mode_counts = Counter(str(row.get("exposure_mode")) for row in rows)
    expected_mode_counts = {
        "auto": expected_conditions * 3,
        "manual": expected_conditions * 27,
    }
    if mode_counts != expected_mode_counts:
        raise ValueError(f"Unexpected exposure-mode counts: {dict(mode_counts)}")
    return {
        "captures": len(rows),
        "unique_capture_keys": len(set(capture_keys)),
        "samples": len(sample_ids),
        "zooms": len(zoom_ids),
        "lights": len(light_ids),
        "conditions": expected_conditions,
        "parameters_per_condition": len(EXPECTED_PARAMETER_KEYS),
        "exposure_mode_counts": dict(sorted(mode_counts.items())),
        "sample_counts": dict(sorted(Counter(str(row["sample_id"]) for row in rows).items())),
        "zoom_counts": dict(sorted(Counter(str(row["zoom_id"]) for row in rows).items())),
        "light_counts": dict(sorted(Counter(str(row["light_id"]) for row in rows).items())),
    }


def load_source_capture_rows(source_root: Path) -> list[dict[str, Any]]:
    manifest = source_root / "captures.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Source capture manifest missing: {manifest}")
    raw_rows = load_jsonl(manifest)
    rows, _duplicate_count = deduplicate_capture_rows(raw_rows)
    validate_capture_rows(rows)
    for row in rows:
        path = source_root / str(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Captured image missing: {path}")
    return rows


def load_cropped_manifest(cropped_root: Path) -> list[dict[str, Any]]:
    manifest = cropped_root / "crop_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Cropped manifest missing: {manifest}. Run prepare_replication first."
        )
    rows = load_jsonl(manifest)
    validate_capture_rows(rows)
    for row in rows:
        path = cropped_root / str(row["cropped_image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Cropped image missing: {path}")
    return rows


def closed_class_metadata(
    reference_root: Path,
    class_index_json: Path,
) -> tuple[list[str], list[int], dict[int, str]]:
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root missing: {reference_root}")
    wnids = sorted(path.name for path in reference_root.iterdir() if path.is_dir())
    if len(wnids) != EXPECTED_CLOSED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLOSED_CLASSES} closed-set WNIDs, found {len(wnids)}"
        )
    with class_index_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    wnid_to_index = {str(row[0]): int(index) for index, row in payload.items()}
    index_to_name = {int(index): str(row[1]) for index, row in payload.items()}
    missing = sorted(set(wnids) - set(wnid_to_index))
    if missing:
        raise ValueError(f"Closed-set WNIDs absent from ImageNet class index: {missing}")
    indices = [wnid_to_index[wnid] for wnid in wnids]
    return wnids, indices, index_to_name


def capture_key_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["capture_key"]) for row in rows}
