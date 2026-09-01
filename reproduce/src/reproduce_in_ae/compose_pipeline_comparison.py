from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .protocol import workspace_root
from .replication import atomic_csv_dump


LOCAL_LIGHT_ID = "b200"
AE_SHOT = 2
DIVERSE_LIGHT_ID = "l1"
DIVERSE_PARAMETER = "param_2"
COLUMNS = (
    ("original", "Original ImageNet"),
    ("dpi600_z001", "600dpi Zoom1"),
    ("dpi600_z002", "600dpi Zoom2"),
    ("replicated_z001", "Replicated Zoom1"),
    ("replicated_z002", "Replicated Zoom2"),
    ("imagenet_es_diverse", "ImageNet-ES-Diverse"),
)


@dataclass(frozen=True)
class SampleComparison:
    sample_id: str
    class_name: str
    sources: Mapping[str, Path]


def default_labels_csv() -> Path:
    return workspace_root() / "data" / "replication" / "manual_dataset" / "labels.csv"


def default_output_dir() -> Path:
    return (
        workspace_root()
        / "replicate_result"
        / "comparison"
        / "fifteen_samples_analysis"
        / "visualization"
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for root in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
    ):
        path = root / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 15 or len({row["sample_id"] for row in rows}) != 15:
        raise ValueError("Pipeline visualization requires exactly 15 unique samples")
    return rows


def build_samples(labels_csv: Path) -> list[SampleComparison]:
    workspace = workspace_root()
    labels_csv = labels_csv.resolve()
    samples: list[SampleComparison] = []
    for row in _load_labels(labels_csv):
        sample_id = str(row["sample_id"])
        capture_relative = (
            Path(sample_id)
            / "{zoom_id}"
            / LOCAL_LIGHT_ID
            / "ae"
            / f"ae_{AE_SHOT:02d}.jpg"
        )
        diverse_path = (
            workspace
            / "data"
            / "ImageNet-ES-Diverse"
            / "es-diverse-test"
            / "auto_exposure"
            / DIVERSE_LIGHT_ID
            / DIVERSE_PARAMETER
            / str(row["wnid"])
            / f"{sample_id}.JPEG"
        )
        sources = {
            "original": labels_csv.parent / str(row["original_path"]),
            "dpi600_z001": workspace
            / "data"
            / "replication"
            / "dpi600_roi"
            / Path(str(capture_relative).format(zoom_id="z001")),
            "dpi600_z002": workspace
            / "data"
            / "replication"
            / "dpi600_roi"
            / Path(str(capture_relative).format(zoom_id="z002")),
            "replicated_z001": workspace
            / "data"
            / "replication"
            / "replicated_capture_roi"
            / Path(str(capture_relative).format(zoom_id="z001")),
            "replicated_z002": workspace
            / "data"
            / "replication"
            / "replicated_capture_roi"
            / Path(str(capture_relative).format(zoom_id="z002")),
            "imagenet_es_diverse": diverse_path,
        }
        missing = [path for path in sources.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{sample_id}: missing visualization sources: "
                + ", ".join(str(path) for path in missing)
            )
        samples.append(
            SampleComparison(
                sample_id=sample_id,
                class_name=str(row["class_name"]).replace("_", " "),
                sources=sources,
            )
        )
    return samples


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=3)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=3,
    )


def _render_source_cell(
    source_path: Path,
    *,
    width: int,
    height: int,
    annotation_height: int,
) -> Image.Image:
    cell = Image.new("RGB", (width, height), (70, 70, 70))
    with Image.open(source_path) as source:
        source_size = source.size
        rendered = ImageOps.contain(
            source.convert("RGB"),
            (width - 16, height - annotation_height - 16),
            method=Image.Resampling.LANCZOS,
        )
    cell.paste(
        rendered,
        ((width - rendered.width) // 2, (height - annotation_height - rendered.height) // 2),
    )
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, height - annotation_height, width, height), fill=(42, 42, 42))
    _centered_text(
        draw,
        (0, height - annotation_height, width, height),
        f"source: {source_size[0]}x{source_size[1]}",
        font=_font(16),
        fill=(225, 225, 225),
    )
    return cell


def _draw_headers(
    canvas: Image.Image,
    *,
    x_offset: int,
    y_offset: int,
    cell_width: int,
    header_height: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    for index, (_key, label) in enumerate(COLUMNS):
        x0 = x_offset + index * cell_width
        _centered_text(
            draw,
            (x0, y_offset, x0 + cell_width, y_offset + header_height),
            label,
            font=_font(19, bold=True),
            fill=(245, 245, 245),
        )


def compose_sample_row(sample: SampleComparison, output: Path) -> Path:
    cell_width = 340
    cell_height = 290
    title_height = 75
    header_height = 65
    canvas = Image.new(
        "RGB",
        (len(COLUMNS) * cell_width, title_height + header_height + cell_height),
        (28, 28, 28),
    )
    draw = ImageDraw.Draw(canvas)
    _centered_text(
        draw,
        (0, 0, canvas.width, title_height),
        f"{sample.sample_id}  |  {sample.class_name}",
        font=_font(27, bold=True),
        fill=(250, 250, 250),
    )
    _draw_headers(
        canvas,
        x_offset=0,
        y_offset=title_height,
        cell_width=cell_width,
        header_height=header_height,
    )
    y0 = title_height + header_height
    for index, (key, _label) in enumerate(COLUMNS):
        x0 = index * cell_width
        cell = _render_source_cell(
            sample.sources[key], width=cell_width, height=cell_height, annotation_height=34
        )
        canvas.paste(cell, (x0, y0))
        draw.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=(145, 145, 145),
            width=1,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output.resolve()


def compose_overview(samples: Sequence[SampleComparison], output: Path) -> Path:
    label_width = 350
    cell_width = 300
    cell_height = 235
    title_height = 80
    header_height = 65
    canvas = Image.new(
        "RGB",
        (
            label_width + len(COLUMNS) * cell_width,
            title_height + header_height + len(samples) * cell_height,
        ),
        (28, 28, 28),
    )
    draw = ImageDraw.Draw(canvas)
    _centered_text(
        draw,
        (0, 0, canvas.width, title_height),
        "15-sample pipeline image comparison",
        font=_font(30, bold=True),
        fill=(250, 250, 250),
    )
    _draw_headers(
        canvas,
        x_offset=label_width,
        y_offset=title_height,
        cell_width=cell_width,
        header_height=header_height,
    )
    for row_index, sample in enumerate(samples):
        y0 = title_height + header_height + row_index * cell_height
        _centered_text(
            draw,
            (12, y0, label_width - 12, y0 + cell_height),
            f"{sample.sample_id}\n{sample.class_name}",
            font=_font(18, bold=True),
            fill=(240, 240, 240),
        )
        for column_index, (key, _label) in enumerate(COLUMNS):
            x0 = label_width + column_index * cell_width
            cell = _render_source_cell(
                sample.sources[key],
                width=cell_width,
                height=cell_height,
                annotation_height=30,
            )
            canvas.paste(cell, (x0, y0))
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
                outline=(145, 145, 145),
                width=1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output.resolve()


def compose_all(labels_csv: Path, output_dir: Path) -> list[Path]:
    samples = build_samples(labels_csv)
    sample_dir = output_dir / "sample_rows"
    outputs = [
        compose_sample_row(sample, sample_dir / f"{sample.sample_id}.png")
        for sample in samples
    ]
    outputs.append(compose_overview(samples, output_dir / "all_15_samples_comparison.png"))

    manifest_rows = []
    for sample in samples:
        row: dict[str, str] = {
            "sample_id": sample.sample_id,
            "class_name": sample.class_name,
            "local_light_id": LOCAL_LIGHT_ID,
            "local_ae_shot": f"ae_{AE_SHOT:02d}",
            "diverse_light_id": DIVERSE_LIGHT_ID,
            "diverse_parameter": DIVERSE_PARAMETER,
        }
        for key, _label in COLUMNS:
            with Image.open(sample.sources[key]) as image:
                row[f"{key}_resolution"] = f"{image.width}x{image.height}"
            row[f"{key}_source"] = str(sample.sources[key].resolve())
        manifest_rows.append(row)
    atomic_csv_dump(manifest_rows, output_dir / "source_manifest.csv")

    readme = output_dir / "README.md"
    readme.write_text(
        "# Fifteen-sample pipeline visualization\n\n"
        "Each row shows the unmodified source content, resized with aspect ratio preserved and no crop or contrast enhancement. "
        "The annotation below each cell is its source resolution.\n\n"
        f"- Local capture pipelines: `{LOCAL_LIGHT_ID} / ae_{AE_SHOT:02d}`\n"
        f"- ImageNet-ES-Diverse: `{DIVERSE_LIGHT_ID} / {DIVERSE_PARAMETER}`\n"
        "- Column order: Original ImageNet, 600dpi Zoom1, 600dpi Zoom2, Replicated Zoom1, Replicated Zoom2, ImageNet-ES-Diverse.\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose 15-sample six-source image comparisons.")
    parser.add_argument("--labels-csv", type=Path, default=default_labels_csv())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = compose_all(args.labels_csv, args.output_dir)
    print(f"Wrote {len(outputs) - 1} sample rows and one overview under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
