from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from .protocol import workspace_root


POINTS_PER_INCH = 72.0
PAGE_WIDTH, PAGE_HEIGHT = letter
DEFAULT_IMAGE_WIDTH_RATIO = 0.60
DEFAULT_LABEL_FONT_SIZE = 8.0
MIN_LABEL_FONT_SIZE = 5.0
LABEL_SIDE_MARGIN = 0.5 * POINTS_PER_INCH
LABEL_BASELINE_FROM_TOP = 0.45 * POINTS_PER_INCH
MAX_IMAGE_HEIGHT_WITH_LABEL = 9.5 * POINTS_PER_INCH
MAX_IMAGE_HEIGHT_WITHOUT_LABEL = 10.0 * POINTS_PER_INCH
SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def default_input_dir() -> Path:
    return (
        workspace_root()
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "sampled_tin_no_resize2"
    )


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    width: int | None
    height: int | None
    mode: str | None
    orientation: int | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Placement:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class GenerationResult:
    pdf_paths: tuple[Path, ...]
    manifest_path: Path
    discovered: int
    rendered: int
    skipped: int


def discover_images(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )


def _oriented_size(width: int, height: int, orientation: int | None) -> tuple[int, int]:
    if orientation in {5, 6, 7, 8}:
        return height, width
    return width, height


def inspect_image(path: Path, input_dir: Path) -> ImageRecord:
    relative_path = path.relative_to(input_dir).as_posix()
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            orientation = int(image.getexif().get(274, 1))
            width, height = _oriented_size(*image.size, orientation)
            mode = image.mode
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions {width}x{height}")
        return ImageRecord(path, relative_path, width, height, mode, orientation)
    except Exception as exc:
        return ImageRecord(path, relative_path, None, None, None, None, str(exc))


def inspect_images(
    paths: Sequence[Path], input_dir: Path, *, skip_invalid: bool
) -> list[ImageRecord]:
    records = [inspect_image(path, input_dir) for path in paths]
    invalid = [record for record in records if not record.valid]
    if invalid and not skip_invalid:
        details = "\n".join(
            f"  - {record.relative_path}: {record.error}" for record in invalid[:10]
        )
        suffix = "" if len(invalid) <= 10 else f"\n  ... and {len(invalid) - 10} more"
        raise ValueError(f"Found {len(invalid)} invalid image(s):\n{details}{suffix}")
    return records


def image_placement(
    width_px: int,
    height_px: int,
    *,
    image_width_ratio: float = DEFAULT_IMAGE_WIDTH_RATIO,
    show_label: bool = True,
) -> Placement:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 0 < image_width_ratio <= 1:
        raise ValueError("image_width_ratio must be in (0, 1]")

    max_width = PAGE_WIDTH * image_width_ratio
    max_height = (
        MAX_IMAGE_HEIGHT_WITH_LABEL if show_label else MAX_IMAGE_HEIGHT_WITHOUT_LABEL
    )
    scale = min(max_width / width_px, max_height / height_px)
    width = width_px * scale
    height = height_px * scale
    return Placement(
        x=(PAGE_WIDTH - width) / 2.0,
        y=(PAGE_HEIGHT - height) / 2.0,
        width=width,
        height=height,
    )


def fit_label(
    text: str,
    *,
    max_width: float = PAGE_WIDTH - 2 * LABEL_SIDE_MARGIN,
    font_name: str = "Helvetica",
) -> tuple[str, float]:
    for size in (
        DEFAULT_LABEL_FONT_SIZE,
        7.5,
        7.0,
        6.5,
        6.0,
        5.5,
        MIN_LABEL_FONT_SIZE,
    ):
        if stringWidth(text, font_name, size) <= max_width:
            return text, size

    ellipsis = "..."
    left = 0
    right = len(text)
    best = ellipsis
    while left <= right:
        keep = (left + right) // 2
        head = (keep + 1) // 2
        tail = keep // 2
        candidate = text[:head] + ellipsis + (text[-tail:] if tail else "")
        if stringWidth(candidate, font_name, MIN_LABEL_FONT_SIZE) <= max_width:
            best = candidate
            left = keep + 1
        else:
            right = keep - 1
    return best, MIN_LABEL_FONT_SIZE


def _label_text(record: ImageRecord, label: str) -> str | None:
    if label == "none":
        return None
    if label == "basename":
        return record.path.name
    if label == "relative":
        return record.relative_path
    raise ValueError(f"Unknown label mode: {label}")


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or "transparency" in image.info


def _prepared_image(record: ImageRecord):
    """Return a ReportLab source while avoiding JPEG recompression when possible."""
    orientation = record.orientation or 1
    suffix = record.path.suffix.lower()
    if suffix in {".jpg", ".jpeg"} and orientation == 1 and record.mode in {
        "RGB",
        "L",
        "CMYK",
    }:
        return str(record.path), None

    original = Image.open(record.path)
    image = ImageOps.exif_transpose(original)
    if image is not original:
        original.close()
    if _has_transparency(image):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        rgba.close()
        image.close()
        image = background
    elif image.mode not in {"RGB", "L", "CMYK"}:
        converted = image.convert("RGB")
        image.close()
        image = converted
    return ImageReader(image), image


def _part_paths(output_pdf: Path, rendered_count: int, max_pages: int) -> list[Path]:
    if max_pages < 0:
        raise ValueError("max_pages_per_pdf cannot be negative")
    if max_pages == 0 or rendered_count <= max_pages:
        return [output_pdf]
    count = math.ceil(rendered_count / max_pages)
    return [
        output_pdf.with_name(
            f"{output_pdf.stem}-part-{index:04d}-of-{count:04d}{output_pdf.suffix}"
        )
        for index in range(1, count + 1)
    ]


def _temporary_path(target: Path, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-", suffix=suffix, dir=target.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _manifest_rows(
    records: Sequence[ImageRecord], rendered_rows: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.valid:
            rows.append(rendered_rows[record.relative_path])
        else:
            rows.append(
                {
                    "status": "skipped",
                    "pdf_file": "",
                    "pdf_page": "",
                    "global_page": "",
                    "relative_path": record.relative_path,
                    "source_width_px": "",
                    "source_height_px": "",
                    "source_mode": "",
                    "placed_width_in": "",
                    "placed_height_in": "",
                    "error": record.error or "invalid image",
                }
            )
    return rows


def generate_print_pdfs(
    input_dir: Path,
    output_pdf: Path,
    *,
    manifest_path: Path | None = None,
    label: str = "relative",
    image_width_ratio: float = DEFAULT_IMAGE_WIDTH_RATIO,
    skip_invalid: bool = False,
    max_pages_per_pdf: int = 0,
    overwrite: bool = False,
) -> GenerationResult:
    input_dir = Path(input_dir).resolve()
    output_pdf = Path(output_pdf).resolve()
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("output_pdf must end in .pdf")
    manifest_path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else output_pdf.with_suffix(".manifest.csv")
    )
    if manifest_path == output_pdf:
        raise ValueError("manifest_path and output_pdf must be different files")
    if label not in {"relative", "basename", "none"}:
        raise ValueError("label must be relative, basename, or none")

    paths = discover_images(input_dir)
    if not paths:
        raise ValueError(f"No supported images found under {input_dir}")
    records = inspect_images(paths, input_dir, skip_invalid=skip_invalid)
    valid_records = [record for record in records if record.valid]
    if not valid_records:
        raise ValueError("No valid images remain after validation")

    pdf_paths = _part_paths(output_pdf, len(valid_records), max_pages_per_pdf)
    collisions = [path for path in [*pdf_paths, manifest_path] if path.exists()]
    if collisions and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(f"Output already exists; pass --overwrite:\n{formatted}")

    per_part = max_pages_per_pdf or len(valid_records)
    temp_pdfs: list[Path] = []
    rendered_rows: dict[str, dict[str, object]] = {}
    try:
        for part_index, pdf_path in enumerate(pdf_paths):
            part_records = valid_records[
                part_index * per_part : min((part_index + 1) * per_part, len(valid_records))
            ]
            temp_pdf = _temporary_path(pdf_path, ".pdf")
            temp_pdfs.append(temp_pdf)
            document = canvas.Canvas(str(temp_pdf), pagesize=letter, pageCompression=1)
            document.setTitle("ImageNet Letter Print Pages")
            document.setSubject(
                "US Letter, actual-size printing; use printer 600 DPI quality"
            )
            for page_index, record in enumerate(part_records, start=1):
                assert record.width is not None and record.height is not None
                placement = image_placement(
                    record.width,
                    record.height,
                    image_width_ratio=image_width_ratio,
                    show_label=label != "none",
                )
                source, opened_image = _prepared_image(record)
                try:
                    document.drawImage(
                        source,
                        placement.x,
                        placement.y,
                        width=placement.width,
                        height=placement.height,
                        preserveAspectRatio=True,
                        anchor="c",
                        mask="auto",
                    )
                finally:
                    if opened_image is not None:
                        opened_image.close()

                text = _label_text(record, label)
                if text is not None:
                    fitted, font_size = fit_label(text)
                    document.setFont("Helvetica", font_size)
                    document.drawCentredString(
                        PAGE_WIDTH / 2.0,
                        PAGE_HEIGHT - LABEL_BASELINE_FROM_TOP,
                        fitted,
                    )
                document.showPage()

                global_page = part_index * per_part + page_index
                rendered_rows[record.relative_path] = {
                    "status": "rendered",
                    "pdf_file": pdf_path.name,
                    "pdf_page": page_index,
                    "global_page": global_page,
                    "relative_path": record.relative_path,
                    "source_width_px": record.width,
                    "source_height_px": record.height,
                    "source_mode": record.mode or "",
                    "placed_width_in": f"{placement.width / POINTS_PER_INCH:.6f}",
                    "placed_height_in": f"{placement.height / POINTS_PER_INCH:.6f}",
                    "error": "",
                }
                if global_page % 100 == 0 or global_page == len(valid_records):
                    print(f"Rendered {global_page}/{len(valid_records)} images", flush=True)
            document.save()

        rows = _manifest_rows(records, rendered_rows)
        temp_manifest = _temporary_path(manifest_path, ".csv")
        try:
            fieldnames = [
                "status",
                "pdf_file",
                "pdf_page",
                "global_page",
                "relative_path",
                "source_width_px",
                "source_height_px",
                "source_mode",
                "placed_width_in",
                "placed_height_in",
                "error",
            ]
            with temp_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            for temp_pdf, pdf_path in zip(temp_pdfs, pdf_paths):
                os.replace(temp_pdf, pdf_path)
                pdf_path.chmod(0o644)
            os.replace(temp_manifest, manifest_path)
            manifest_path.chmod(0o644)
        finally:
            if temp_manifest.exists():
                temp_manifest.unlink()
    except Exception:
        for path in temp_pdfs:
            if path.exists():
                path.unlink()
        raise

    return GenerationResult(
        pdf_paths=tuple(pdf_paths),
        manifest_path=manifest_path,
        discovered=len(records),
        rendered=len(valid_records),
        skipped=len(records) - len(valid_records),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ImageNet images into directly printable US Letter PDF pages. "
            "Print at Actual Size (100%), with Fit-to-page disabled and 600 DPI quality."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument(
        "--output-pdf", type=Path, default=Path("imagenet_letter_print.pdf")
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--label",
        choices=("relative", "basename", "none"),
        default="relative",
        help="Top-of-page sample label (default: relative path)",
    )
    parser.add_argument(
        "--no-filename",
        dest="label",
        action="store_const",
        const="none",
        help="Alias for --label none",
    )
    parser.add_argument(
        "--image-width-ratio",
        type=float,
        default=DEFAULT_IMAGE_WIDTH_RATIO,
        help="Maximum image width as a fraction of Letter width (default: 0.60)",
    )
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=0,
        help="Split into parts after this many valid pages; 0 disables splitting",
    )
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = generate_print_pdfs(
        args.input_dir,
        args.output_pdf,
        manifest_path=args.manifest,
        label=args.label,
        image_width_ratio=args.image_width_ratio,
        skip_invalid=args.skip_invalid,
        max_pages_per_pdf=args.max_pages_per_pdf,
        overwrite=args.overwrite,
    )
    print(f"PDF files: {len(result.pdf_paths)}")
    for path in result.pdf_paths:
        print(f"  {path}")
    print(f"Manifest: {result.manifest_path}")
    print(
        f"Discovered: {result.discovered}; rendered: {result.rendered}; "
        f"skipped: {result.skipped}"
    )


if __name__ == "__main__":
    main()
