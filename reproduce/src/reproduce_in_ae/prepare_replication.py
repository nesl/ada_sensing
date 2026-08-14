from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps
from tqdm import tqdm

from .preview_replication_roi import EXPECTED_IMAGE_SIZE, RoiSpec, load_config, validate_box
from .replication import (
    atomic_json_dump,
    atomic_jsonl_dump,
    default_cropped_root,
    default_roi_config,
    default_source_root,
    load_source_capture_rows,
    parameter_key,
    sha256_file,
    validate_capture_rows,
)


JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the approved 600-image cropped replication dataset."
    )
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--output-root", type=Path, default=default_cropped_root())
    parser.add_argument("--roi-config", type=Path, default=default_roi_config())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _roi_map(specs: list[RoiSpec]) -> dict[tuple[str, str], RoiSpec]:
    mapping = {(spec.sample_id, spec.zoom_id): spec for spec in specs}
    if len(mapping) != len(specs):
        raise ValueError("ROI config contains duplicate sample/zoom keys")
    return mapping


def _atomic_save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.jpg")
    image.save(
        temporary,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False,
        progressive=False,
    )
    os.replace(temporary, path)


def _validate_existing_crop(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == expected_size and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def _crop_record(
    source_row: Mapping[str, Any],
    source_index: int,
    source_root: Path,
    output_root: Path,
    spec: RoiSpec,
    overwrite: bool,
) -> dict[str, Any]:
    relative_path = Path(str(source_row["image_path"]))
    source_path = source_root / relative_path
    output_path = output_root / relative_path
    expected_crop_size = (spec.box[2] - spec.box[0], spec.box[3] - spec.box[1])

    if overwrite or not _validate_existing_crop(output_path, expected_crop_size):
        if output_path.exists() and not overwrite:
            raise ValueError(
                f"Existing crop is invalid: {output_path}. Pass --overwrite to replace it."
            )
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                f"Unexpected source size for {source_path}: {image.size}; "
                f"expected {EXPECTED_IMAGE_SIZE}"
            )
        validate_box(spec, image.size, source_path)
        _atomic_save_jpeg(image.crop(spec.box), output_path)

    row = dict(source_row)
    row.update(
        {
            "source_manifest_index": source_index,
            "parameter_key": parameter_key(source_row),
            "source_image_path": relative_path.as_posix(),
            "cropped_image_path": relative_path.as_posix(),
            "roi_box": list(spec.box),
            "source_image_size": list(EXPECTED_IMAGE_SIZE),
            "cropped_image_size": list(expected_crop_size),
            "source_size_bytes": source_path.stat().st_size,
            "cropped_size_bytes": output_path.stat().st_size,
            "source_sha256": sha256_file(source_path),
            "cropped_sha256": sha256_file(output_path),
            "crop_encoding": {
                "format": "JPEG",
                "quality": JPEG_QUALITY,
                "subsampling": JPEG_SUBSAMPLING,
                "optimize": False,
                "progressive": False,
            },
        }
    )
    return row


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    roi_config: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_rows = load_source_capture_rows(source_root)
    _representative, specs, config_payload = load_config(roi_config)
    roi_by_key = _roi_map(specs)
    observed_keys = {(str(row["sample_id"]), str(row["zoom_id"])) for row in source_rows}
    if observed_keys != set(roi_by_key):
        missing = sorted(observed_keys - set(roi_by_key))
        extra = sorted(set(roi_by_key) - observed_keys)
        raise ValueError(f"ROI key mismatch; missing={missing}, extra={extra}")

    crop_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(tqdm(source_rows, desc="Crop replication ROI")):
        key = (str(source_row["sample_id"]), str(source_row["zoom_id"]))
        crop_rows.append(
            _crop_record(
                source_row=source_row,
                source_index=index,
                source_root=source_root,
                output_root=output_root,
                spec=roi_by_key[key],
                overwrite=overwrite,
            )
        )

    counts = validate_capture_rows(crop_rows)
    manifest_path = output_root / "crop_manifest.jsonl"
    atomic_jsonl_dump(crop_rows, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    roi_config_copy = output_root / "roi_config.json"
    atomic_json_dump(config_payload, roi_config_copy)
    summary = {
        "status": "complete",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_manifest_sha256": sha256_file(source_root / "captures.jsonl"),
        "crop_manifest_sha256": manifest_sha256,
        "roi_config_sha256": sha256_file(roi_config),
        "counts": counts,
        "parameter_counts": dict(
            sorted(Counter(str(row["parameter_key"]) for row in crop_rows).items())
        ),
        "jpeg_encoding": {
            "quality": JPEG_QUALITY,
            "subsampling": JPEG_SUBSAMPLING,
            "optimize": False,
            "progressive": False,
        },
    }
    atomic_json_dump(summary, output_root / "dataset_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        roi_config=args.roi_config,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
