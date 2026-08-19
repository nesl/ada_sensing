from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .protocol import workspace_root


LIGHT_IDS = ("l1", "l2", "l3", "l4", "l6", "l7")
AE_PARAMETER = "param_2"
SAMPLE_IDS = (
    "ILSVRC2012_val_00011526",
    "ILSVRC2012_val_00011857",
    "ILSVRC2012_val_00034232",
    "ILSVRC2012_val_00038504",
    "ILSVRC2012_val_00048338",
)


def default_labels_csv() -> Path:
    return workspace_root() / "data" / "replication" / "manual_dataset" / "labels.csv"


def default_ae_root() -> Path:
    return (
        workspace_root()
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure"
    )


def default_output() -> Path:
    return (
        workspace_root()
        / "replicate_result"
        / "comparison"
        / "ae_visual"
        / "all_samples_all_lighting_param2.png"
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_labels(labels_csv: Path) -> dict[str, dict[str, str]]:
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = {str(row["sample_id"]): dict(row) for row in csv.DictReader(handle)}
    missing = sorted(set(SAMPLE_IDS) - set(rows))
    if missing:
        raise ValueError(f"Labels are missing for samples: {missing}")
    return rows


def compose_grid(
    output: Path,
    *,
    labels_csv: Path,
    ae_root: Path,
    overwrite: bool = False,
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")
    labels = _load_labels(labels_csv)

    row_label_width = 350
    cell_width = 300
    cell_height = 270
    title_height = 80
    header_height = 70
    width = row_label_width + len(LIGHT_IDS) * cell_width
    height = title_height + header_height + len(SAMPLE_IDS) * cell_height
    background = (32, 32, 32)
    cell_background = (96, 96, 96)
    grid_color = (150, 150, 150)
    text_color = (245, 245, 245)

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    header_font = _font(26, bold=True)
    sample_font = _font(20, bold=True)
    class_font = _font(18)
    title = "ImageNet-ES-Diverse Auto Exposure — param_2"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) / 2, 23),
        title,
        fill=text_color,
        font=title_font,
    )

    for column, light_id in enumerate(LIGHT_IDS):
        x0 = row_label_width + column * cell_width
        box = draw.textbbox((0, 0), light_id, font=header_font)
        draw.text(
            (x0 + (cell_width - (box[2] - box[0])) / 2, title_height + 18),
            light_id,
            fill=text_color,
            font=header_font,
        )

    for row_index, sample_id in enumerate(SAMPLE_IDS):
        metadata = labels[sample_id]
        y0 = title_height + header_height + row_index * cell_height
        sample_box = draw.textbbox((0, 0), sample_id, font=sample_font)
        class_name = str(metadata["class_name"]).replace("_", " ")
        class_box = draw.textbbox((0, 0), class_name, font=class_font)
        label_center = y0 + cell_height / 2
        draw.text(
            (20, label_center - (sample_box[3] - sample_box[1]) - 8),
            sample_id,
            fill=text_color,
            font=sample_font,
        )
        draw.text(
            (20, label_center + 8),
            class_name,
            fill=(205, 205, 205),
            font=class_font,
        )

        for column, light_id in enumerate(LIGHT_IDS):
            x0 = row_label_width + column * cell_width
            image_path = (
                ae_root
                / light_id
                / AE_PARAMETER
                / str(metadata["wnid"])
                / Path(str(metadata["source_relative_path"])).name
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"AE image missing: {image_path}")
            cell = Image.new("RGB", (cell_width, cell_height), cell_background)
            with Image.open(image_path) as source:
                rendered = ImageOps.contain(
                    source.convert("RGB"),
                    (cell_width - 16, cell_height - 16),
                    method=Image.Resampling.LANCZOS,
                )
            paste_x = (cell_width - rendered.width) // 2
            paste_y = (cell_height - rendered.height) // 2
            cell.paste(rendered, (paste_x, paste_y))
            canvas.paste(cell, (x0, y0))
            draw.rectangle(
                (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
                outline=grid_color,
                width=1,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose a five-sample by six-lighting param_2 AE contact sheet."
    )
    parser.add_argument("--labels-csv", type=Path, default=default_labels_csv())
    parser.add_argument("--ae-root", type=Path, default=default_ae_root())
    parser.add_argument("--output", type=Path, default=default_output())
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = compose_grid(
        args.output,
        labels_csv=args.labels_csv,
        ae_root=args.ae_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
