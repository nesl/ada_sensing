from __future__ import annotations

import argparse
import csv
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .protocol import workspace_root


POINTS_PER_INCH = 72.0
PAGE_WIDTH_PT, PAGE_HEIGHT_PT = letter
PAGE_WIDTH_IN = PAGE_WIDTH_PT / POINTS_PER_INCH
PAGE_HEIGHT_IN = PAGE_HEIGHT_PT / POINTS_PER_INCH
DEFAULT_DPI = 600
LABEL_FONT_SIZE_PT = 14.0
LABEL_RIGHT_MARGIN_IN = 0.60
LABEL_BASELINE_FROM_TOP_IN = 0.81
REQUIRED_SELECTION_FIELDS = (
    "sample_id",
    "original_path",
    "class_index",
    "wnid",
    "class_name",
    "source_relative_path",
)
DEFAULT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def default_manual_dataset_root() -> Path:
    return workspace_root() / "data" / "replication" / "manual_dataset"


def default_labels_csv() -> Path:
    return default_manual_dataset_root() / "labels.csv"


def default_output_pdf() -> Path:
    return (
        default_manual_dataset_root()
        / "600dpi"
        / "imagenet_letter_600dpi_1to1.pdf"
    )


@dataclass(frozen=True)
class Selection:
    metadata: Mapping[str, str]
    source_path: Path


@dataclass(frozen=True)
class BitmapPdfResult:
    pdf_path: Path
    manifest_path: Path
    pages: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letter_canvas_size(dpi: int) -> tuple[int, int]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    return round(PAGE_WIDTH_IN * dpi), round(PAGE_HEIGHT_IN * dpi)


def centered_paste_box(
    canvas_size: tuple[int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    canvas_width, canvas_height = canvas_size
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("source image dimensions must be positive")
    if image_width > canvas_width or image_height > canvas_height:
        raise ValueError(
            f"source image {image_width}x{image_height} exceeds "
            f"canvas {canvas_width}x{canvas_height}; resizing is forbidden"
        )
    left = (canvas_width - image_width) // 2
    top = (canvas_height - image_height) // 2
    return left, top, left + image_width, top + image_height


def load_selections(labels_csv: Path) -> list[Selection]:
    labels_csv = Path(labels_csv).resolve()
    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels CSV does not exist: {labels_csv}")
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            field
            for field in REQUIRED_SELECTION_FIELDS
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"labels CSV is missing required fields: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("labels CSV contains no samples")

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("labels CSV contains duplicate sample_id values")

    selections: list[Selection] = []
    for row in rows:
        source_path = (labels_csv.parent / row["original_path"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"source image for {row['sample_id']} does not exist: {source_path}"
            )
        selections.append(Selection(metadata=row, source_path=source_path))
    return selections


def resolve_font_path(font_path: Path | None) -> Path:
    if font_path is not None:
        resolved = Path(font_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"font file does not exist: {resolved}")
        return resolved
    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No Arial-compatible TrueType font found; pass --font-path explicitly"
    )


def _temporary_path(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-", suffix=suffix, dir=target.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _source_dpi_text(image: Image.Image) -> str:
    dpi = image.info.get("dpi")
    if not dpi:
        return ""
    try:
        return f"{float(dpi[0]):g}x{float(dpi[1]):g}"
    except (TypeError, ValueError, IndexError):
        return str(dpi)


def _load_strict_rgb_jpeg(path: Path) -> tuple[Image.Image, dict[str, object]]:
    with Image.open(path) as opened:
        opened.load()
        orientation = opened.getexif().get(274)
        details: dict[str, object] = {
            "format": opened.format or "",
            "mode": opened.mode,
            "orientation": "" if orientation is None else orientation,
            "dpi_metadata": _source_dpi_text(opened),
        }
        if opened.format != "JPEG":
            raise ValueError(f"source must be JPEG, got {opened.format!r}: {path}")
        if opened.mode != "RGB":
            raise ValueError(
                f"source must decode directly to RGB without conversion, "
                f"got {opened.mode!r}: {path}"
            )
        if orientation not in {None, 0, 1}:
            raise ValueError(
                f"source has EXIF orientation {orientation}; EXIF transpose is forbidden: {path}"
            )
        return opened.copy(), details


def _draw_label(
    page: Image.Image,
    text: str,
    *,
    dpi: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    x = page.width - round(LABEL_RIGHT_MARGIN_IN * dpi)
    y = round(LABEL_BASELINE_FROM_TOP_IN * dpi)
    draw = ImageDraw.Draw(page)
    bounds = draw.textbbox((x, y), text, font=font, anchor="rs")
    if (
        bounds[0] < 0
        or bounds[1] < 0
        or bounds[2] > page.width
        or bounds[3] > page.height
    ):
        raise ValueError(f"label does not fit on page at fixed paper-faithful size: {text}")
    draw.text((x, y), text, font=font, fill="black", anchor="rs")


def render_bitmap_page(
    source: Image.Image,
    label_text: str,
    *,
    dpi: int,
    font: ImageFont.FreeTypeFont,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if source.mode != "RGB":
        raise ValueError(f"source image must be RGB, got {source.mode!r}")
    size = letter_canvas_size(dpi)
    paste_box = centered_paste_box(size, source.size)
    page = Image.new("RGB", size, "white")
    page.paste(source, paste_box[:2])
    _draw_label(page, label_text, dpi=dpi, font=font)
    return page, paste_box


def generate_bitmap_print_pdf(
    labels_csv: Path,
    output_pdf: Path,
    *,
    dpi: int = DEFAULT_DPI,
    font_path: Path | None = None,
    overwrite: bool = False,
) -> BitmapPdfResult:
    labels_csv = Path(labels_csv).resolve()
    output_pdf = Path(output_pdf).resolve()
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("output_pdf must end in .pdf")
    manifest_path = output_pdf.with_suffix(".manifest.csv")
    collisions = [path for path in (output_pdf, manifest_path) if path.exists()]
    if collisions and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(f"Output already exists; pass --overwrite:\n{formatted}")

    selections = load_selections(labels_csv)
    resolved_font = resolve_font_path(font_path)
    font_size_px = round(LABEL_FONT_SIZE_PT * dpi / POINTS_PER_INCH)
    font = ImageFont.truetype(str(resolved_font), font_size_px)
    canvas_width, canvas_height = letter_canvas_size(dpi)

    temporary_pdf = _temporary_path(output_pdf, ".pdf")
    temporary_manifest = _temporary_path(manifest_path, ".csv")
    rows: list[dict[str, object]] = []
    try:
        document = canvas.Canvas(
            str(temporary_pdf), pagesize=letter, pageCompression=1
        )
        document.setTitle("ImageNet Letter 600-PPI 1:1 Bitmap Banners")
        document.setSubject(
            "Letter bitmap pages; source pixels placed 1:1 on a 600-PPI canvas"
        )
        for page_index, selection in enumerate(selections, start=1):
            source, details = _load_strict_rgb_jpeg(selection.source_path)
            try:
                label_text = selection.metadata["source_relative_path"]
                bitmap_page, paste_box = render_bitmap_page(
                    source, label_text, dpi=dpi, font=font
                )
            finally:
                source.close()
            try:
                document.drawImage(
                    ImageReader(bitmap_page),
                    0,
                    0,
                    width=PAGE_WIDTH_PT,
                    height=PAGE_HEIGHT_PT,
                    preserveAspectRatio=False,
                    mask=None,
                )
            finally:
                bitmap_page.close()
            document.showPage()

            left, top, right, bottom = paste_box
            metadata = selection.metadata
            rows.append(
                {
                    "pdf_file": output_pdf.name,
                    "pdf_page": page_index,
                    "sample_id": metadata["sample_id"],
                    "source_path": metadata["original_path"],
                    "source_relative_path": metadata["source_relative_path"],
                    "class_index": metadata["class_index"],
                    "wnid": metadata["wnid"],
                    "class_name": metadata["class_name"],
                    "source_width_px": right - left,
                    "source_height_px": bottom - top,
                    "source_mode": details["mode"],
                    "source_format": details["format"],
                    "source_orientation": details["orientation"],
                    "source_dpi_metadata_ignored": details["dpi_metadata"],
                    "source_sha256": sha256_file(selection.source_path),
                    "canvas_width_px": canvas_width,
                    "canvas_height_px": canvas_height,
                    "canvas_dpi": dpi,
                    "paste_left_px": left,
                    "paste_top_px": top,
                    "paste_right_px": right,
                    "paste_bottom_px": bottom,
                    "placed_width_in": f"{(right - left) / dpi:.6f}",
                    "placed_height_in": f"{(bottom - top) / dpi:.6f}",
                    "label_text": label_text,
                    "font_path": str(resolved_font),
                    "label_font_size_pt": f"{LABEL_FONT_SIZE_PT:g}",
                    "label_font_size_px": font_size_px,
                    "label_right_margin_in": f"{LABEL_RIGHT_MARGIN_IN:.2f}",
                    "label_baseline_from_top_in": f"{LABEL_BASELINE_FROM_TOP_IN:.2f}",
                    "points_per_canvas_pixel": f"{POINTS_PER_INCH / dpi:.6f}",
                }
            )
            print(f"Rendered {page_index}/{len(selections)} bitmap pages", flush=True)
        document.save()

        fieldnames = list(rows[0])
        with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_pdf, output_pdf)
        output_pdf.chmod(0o644)
        os.replace(temporary_manifest, manifest_path)
        manifest_path.chmod(0o644)
    except Exception:
        for path in (temporary_pdf, temporary_manifest):
            if path.exists():
                path.unlink()
        raise

    return BitmapPdfResult(output_pdf, manifest_path, len(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Letter bitmap banner pages at 600 PPI, paste source JPEG pixels "
            "1:1 without resize, and wrap the lossless pages in a PDF."
        )
    )
    parser.add_argument("--labels-csv", type=Path, default=default_labels_csv())
    parser.add_argument("--output-pdf", type=Path, default=default_output_pdf())
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = generate_bitmap_print_pdf(
        args.labels_csv,
        args.output_pdf,
        dpi=args.dpi,
        font_path=args.font_path,
        overwrite=args.overwrite,
    )
    print(f"PDF: {result.pdf_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Pages: {result.pages}")


if __name__ == "__main__":
    main()
