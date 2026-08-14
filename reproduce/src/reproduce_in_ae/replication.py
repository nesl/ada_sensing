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


def validate_capture_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_count: int = EXPECTED_CAPTURE_COUNT,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} captures, found {len(rows)}")
    capture_keys = [str(row.get("capture_key", "")) for row in rows]
    if any(not key for key in capture_keys):
        raise ValueError("Every capture must contain capture_key")
    if len(set(capture_keys)) != len(capture_keys):
        duplicates = sorted(key for key, count in Counter(capture_keys).items() if count > 1)
        raise ValueError(f"Duplicate capture keys: {duplicates[:5]}")
    statuses = Counter(str(row.get("capture_status")) for row in rows)
    if statuses != {"captured": expected_count}:
        raise ValueError(f"Unexpected capture statuses: {dict(statuses)}")
    mode_counts = Counter(str(row.get("exposure_mode")) for row in rows)
    if expected_count == EXPECTED_CAPTURE_COUNT and mode_counts != {
        "auto": EXPECTED_AE_COUNT,
        "manual": EXPECTED_MANUAL_COUNT,
    }:
        raise ValueError(f"Unexpected exposure-mode counts: {dict(mode_counts)}")
    for row in rows:
        parameter_key(row)
    return {
        "captures": len(rows),
        "unique_capture_keys": len(set(capture_keys)),
        "exposure_mode_counts": dict(sorted(mode_counts.items())),
        "sample_counts": dict(sorted(Counter(str(row["sample_id"]) for row in rows).items())),
        "zoom_counts": dict(sorted(Counter(str(row["zoom_id"]) for row in rows).items())),
        "light_counts": dict(sorted(Counter(str(row["light_id"]) for row in rows).items())),
    }


def load_source_capture_rows(source_root: Path) -> list[dict[str, Any]]:
    manifest = source_root / "captures.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Source capture manifest missing: {manifest}")
    rows = load_jsonl(manifest)
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
