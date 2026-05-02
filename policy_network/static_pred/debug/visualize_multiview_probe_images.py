from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "data" / "ImageNet-ES-Diverse" / "manifest_all.json"
DEFAULT_PROBES = (
    ROOT / "policy_network" / "results_hist3_multiview" / "brightness_histogram_probes.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "policy_network" / "results_hist3_multiview" / "probe_visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the selected multiview probe images under each lighting condition. "
            "Each output image uses one representative ImageNet sample that exists in all envs."
        )
    )
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--probe_json", type=str, default=str(DEFAULT_PROBES))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--envs", type=str, default="l1,l2,l3,l4,l6,l7")
    parser.add_argument("--image_size", type=int, default=180)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def parse_sample_id(sample_id: str) -> Tuple[str, str, str]:
    env, class_id, stem = sample_id.split("__", 2)
    return env, class_id, stem


def choose_representative_key(
    manifest_items: List[Dict[str, Any]],
    envs: List[str],
) -> Tuple[str, str]:
    key_to_envs: Dict[Tuple[str, str], set[str]] = {}
    for item in manifest_items:
        env, class_id, stem = parse_sample_id(str(item["id"]))
        key_to_envs.setdefault((class_id, stem), set()).add(env)

    required_envs = set(envs)
    for key in sorted(key_to_envs):
        if required_envs.issubset(key_to_envs[key]):
            return key

    raise ValueError(f"Could not find one sample present in all envs: {envs}")


def build_manifest_lookup(
    manifest_items: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for item in manifest_items:
        env, class_id, stem = parse_sample_id(str(item["id"]))
        lookup[(env, class_id, stem)] = item
    return lookup


def option_path_and_name(item: Dict[str, Any], option_id: int) -> Tuple[str, str]:
    for candidate in item["candidates"]:
        if int(candidate["option_id"]) == int(option_id):
            option_name = str(candidate.get("meta", {}).get("option_name", ""))
            return str(candidate["path"]), option_name
    raise KeyError(f"option_id={option_id} not found for sample_id={item['id']}")


def load_panel_image(path: str, image_size: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((image_size, image_size))
        canvas = Image.new("RGB", (image_size, image_size), "white")
        x = (image_size - img.width) // 2
        y = (image_size - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - text_w) // 2
    y = y0 + (y1 - y0 - text_h) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=3, align="center")


def make_grid(
    output_path: Path,
    title: str,
    option_ids: List[int],
    manifest_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    representative_key: Tuple[str, str],
    envs: List[str],
    image_size: int,
) -> None:
    class_id, stem = representative_key
    left_label_w = 90
    header_h = 70
    title_h = 44
    pad = 12
    cell_w = image_size
    cell_h = image_size + 38
    width = left_label_w + len(option_ids) * (cell_w + pad) + pad
    height = title_h + header_h + len(envs) * (cell_h + pad) + pad

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, width, title_h), fill=(245, 245, 245))
    draw_centered_text(
        draw,
        (0, 0, width, title_h),
        f"{title} | sample={class_id}/{stem}",
        font,
    )

    for col, option_id in enumerate(option_ids):
        item = manifest_lookup[(envs[0], class_id, stem)]
        _, option_name = option_path_and_name(item, option_id)
        x0 = left_label_w + pad + col * (cell_w + pad)
        draw_centered_text(
            draw,
            (x0, title_h, x0 + cell_w, title_h + header_h),
            f"option {option_id}\n{option_name}",
            font,
        )

    for row, env in enumerate(envs):
        y0 = title_h + header_h + pad + row * (cell_h + pad)
        draw_centered_text(draw, (0, y0, left_label_w, y0 + cell_h), env, font)
        item = manifest_lookup[(env, class_id, stem)]
        for col, option_id in enumerate(option_ids):
            path, option_name = option_path_and_name(item, option_id)
            image = load_panel_image(path, image_size)
            x0 = left_label_w + pad + col * (cell_w + pad)
            canvas.paste(image, (x0, y0))
            draw.rectangle((x0, y0, x0 + cell_w, y0 + image_size), outline=(210, 210, 210))
            draw_centered_text(
                draw,
                (x0, y0 + image_size, x0 + cell_w, y0 + cell_h),
                option_name,
                font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    manifest_items = load_json(args.manifest)
    probe_payload = load_json(args.probe_json)
    envs = [env.strip() for env in args.envs.split(",") if env.strip()]
    representative_key = choose_representative_key(manifest_items, envs)
    manifest_lookup = build_manifest_lookup(manifest_items)

    selections = {
        "hist3": probe_payload["hist3"]["selected_option_ids"],
    }
    for seed, selection in sorted(probe_payload.get("random3", {}).items()):
        selections[f"random3_seed{seed}"] = selection["selected_option_ids"]

    output_dir = Path(args.output_dir)
    for name, option_ids in selections.items():
        output_path = output_dir / f"{name}_selected_images_by_lighting.png"
        make_grid(
            output_path=output_path,
            title=name,
            option_ids=[int(option_id) for option_id in option_ids],
            manifest_lookup=manifest_lookup,
            representative_key=representative_key,
            envs=envs,
            image_size=args.image_size,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
