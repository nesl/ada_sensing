from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageOps

from .protocol import project_root, workspace_root


EXPECTED_IMAGE_SIZE = (6528, 4352)
OVERVIEW_TILE_SIZE = (960, 700)
OVERVIEW_COLUMNS = 2


@dataclass(frozen=True)
class RoiSpec:
    sample_id: str
    zoom_id: str
    box: tuple[int, int, int, int]

    @property
    def key(self) -> str:
        return f"{self.sample_id}/{self.zoom_id}"

    @property
    def output_stem(self) -> str:
        return f"{self.sample_id}__{self.zoom_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate review-only ROI overlays and crop previews for replicated capture. "
            "This command never creates the batch-cropped dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=workspace_root() / "data" / "replication" / "replicated_capture",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "replication_roi.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "results" / "replication_roi" / "roi_review",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[str, list[RoiSpec], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported ROI config schema in {path}")
    representative = str(payload.get("representative_relative_path", "")).strip()
    if not representative:
        raise ValueError("representative_relative_path must be non-empty")

    specs: list[RoiSpec] = []
    seen: set[tuple[str, str]] = set()
    for row in payload.get("rois", []):
        sample_id = str(row["sample_id"])
        zoom_id = str(row["zoom_id"])
        raw_box = row["box"]
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError(f"ROI {sample_id}/{zoom_id} must contain four coordinates")
        box = tuple(int(value) for value in raw_box)
        left, top, right, bottom = box
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValueError(f"Invalid ROI box for {sample_id}/{zoom_id}: {box}")
        key = (sample_id, zoom_id)
        if key in seen:
            raise ValueError(f"Duplicate ROI key: {sample_id}/{zoom_id}")
        seen.add(key)
        specs.append(RoiSpec(sample_id, zoom_id, box))
    if not specs:
        raise ValueError("ROI config contains no entries")
    return representative, specs, payload


def representative_path(source_root: Path, spec: RoiSpec, relative: str) -> Path:
    return source_root / spec.sample_id / spec.zoom_id / relative


def validate_box(spec: RoiSpec, image_size: tuple[int, int], path: Path) -> None:
    width, height = image_size
    left, top, right, bottom = spec.box
    if right > width or bottom > height:
        raise ValueError(
            f"ROI {spec.key} {spec.box} exceeds {path} dimensions {image_size}"
        )


def _label(spec: RoiSpec, crop_size: tuple[int, int]) -> str:
    short_sample = spec.sample_id.replace("ILSVRC2012_val_000", "")
    return f"{short_sample} {spec.zoom_id}  box={spec.box}  crop={crop_size}"


def _headered_tile(image: Image.Image, label: str) -> Image.Image:
    content_height = OVERVIEW_TILE_SIZE[1] - 42
    content = ImageOps.contain(image, (OVERVIEW_TILE_SIZE[0], content_height))
    tile = Image.new("RGB", OVERVIEW_TILE_SIZE, "white")
    x = (OVERVIEW_TILE_SIZE[0] - content.width) // 2
    y = 42 + (content_height - content.height) // 2
    tile.paste(content, (x, y))
    ImageDraw.Draw(tile).text((12, 12), label, fill="black")
    return tile


def _overlay(image: Image.Image, spec: RoiSpec) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    line_width = max(8, round(max(image.size) / 500))
    draw.rectangle(spec.box, outline=(255, 0, 0), width=line_width)
    return overlay


def _overview(tiles: Sequence[Image.Image]) -> Image.Image:
    rows = (len(tiles) + OVERVIEW_COLUMNS - 1) // OVERVIEW_COLUMNS
    canvas = Image.new(
        "RGB",
        (OVERVIEW_COLUMNS * OVERVIEW_TILE_SIZE[0], rows * OVERVIEW_TILE_SIZE[1]),
        "white",
    )
    for index, tile in enumerate(tiles):
        x = (index % OVERVIEW_COLUMNS) * OVERVIEW_TILE_SIZE[0]
        y = (index // OVERVIEW_COLUMNS) * OVERVIEW_TILE_SIZE[1]
        canvas.paste(tile, (x, y))
    return canvas


def _assert_writable_targets(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        shown = "\n".join(str(path) for path in existing[:5])
        raise FileExistsError(
            "ROI review outputs already exist; pass --overwrite to replace them:\n"
            + shown
        )


def generate_previews(
    source_root: Path,
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> list[Path]:
    representative, specs, config_payload = load_config(config_path)
    individual_paths = [
        output_dir / f"{spec.output_stem}__{kind}.jpg"
        for spec in specs
        for kind in ("box", "crop")
    ]
    overview_paths = [
        output_dir / "roi_boxes_overview.jpg",
        output_dir / "roi_crops_overview.jpg",
        output_dir / "roi_config.review.json",
    ]
    targets = individual_paths + overview_paths
    _assert_writable_targets(targets, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_tiles: list[Image.Image] = []
    crop_tiles: list[Image.Image] = []
    review_rows: list[dict[str, Any]] = []
    written: list[Path] = []

    for spec in specs:
        source_path = representative_path(source_root, spec, representative)
        if not source_path.is_file():
            raise FileNotFoundError(f"Representative image missing: {source_path}")
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                f"Unexpected source dimensions for {source_path}: {image.size}; "
                f"expected {EXPECTED_IMAGE_SIZE}"
            )
        validate_box(spec, image.size, source_path)
        crop = image.crop(spec.box)
        label = _label(spec, crop.size)
        overlay = _overlay(image, spec)

        box_path = output_dir / f"{spec.output_stem}__box.jpg"
        crop_path = output_dir / f"{spec.output_stem}__crop.jpg"
        overlay.save(box_path, quality=92, subsampling=0)
        crop.save(crop_path, quality=95, subsampling=0)
        written.extend((box_path, crop_path))
        overlay_tiles.append(_headered_tile(overlay, label))
        crop_tiles.append(_headered_tile(crop, label))
        review_rows.append(
            {
                "sample_id": spec.sample_id,
                "zoom_id": spec.zoom_id,
                "box": list(spec.box),
                "source_path": str(source_path.resolve()),
                "source_size": list(image.size),
                "crop_size": list(crop.size),
                "box_preview": box_path.name,
                "crop_preview": crop_path.name,
            }
        )

    boxes_overview = output_dir / "roi_boxes_overview.jpg"
    crops_overview = output_dir / "roi_crops_overview.jpg"
    _overview(overlay_tiles).save(boxes_overview, quality=95, subsampling=0)
    _overview(crop_tiles).save(crops_overview, quality=95, subsampling=0)
    written.extend((boxes_overview, crops_overview))

    review_payload = {
        **config_payload,
        "status": "review_only_not_approved",
        "source_root": str(source_root.resolve()),
        "review_rows": review_rows,
        "safety": "No batch-cropped dataset was created by this command.",
    }
    review_config = output_dir / "roi_config.review.json"
    with review_config.open("w", encoding="utf-8") as handle:
        json.dump(review_payload, handle, indent=2)
        handle.write("\n")
    written.append(review_config)
    return written


def main() -> None:
    args = parse_args()
    written = generate_previews(
        source_root=args.source_root,
        config_path=args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("ROI review-only outputs:")
    for path in written:
        print(path.resolve())
    print("No batch crop or model inference was run.")


if __name__ == "__main__":
    main()
