from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]
LENS_DIR = ROOT / "lens"
POLICY_DIR = ROOT / "policy_network" / "static_pred"

for extra_path in (ROOT, LENS_DIR, POLICY_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model
from policy_dataset import PolicyDataset
from policy_model import (
    SensorPolicyNetwork,
    infer_backbone_name_from_checkpoint,
    infer_input_mode_from_checkpoint,
    infer_num_input_views_from_checkpoint,
    normalize_policy_checkpoint_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a few policy-network test cases by showing the policy input image, "
            "the policy-selected candidate sent to the downstream classifier, the Lens-selected "
            "candidate, and the target-best candidate."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--predictions_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument(
        "--filter_mode",
        type=str,
        choices=[
            "all",
            "policy_wrong",
            "policy_wrong_lens_right",
            "policy_right_lens_wrong",
            "different_choice",
        ],
        default="all",
    )
    parser.add_argument("--sample_ids", type=str, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_side", type=int, default=320)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    has_prediction_file = args.predictions_json is not None
    has_checkpoint = args.checkpoint is not None

    if has_prediction_file and has_checkpoint:
        raise ValueError("Use either --predictions_json or --checkpoint, not both.")
    if not has_prediction_file and not has_checkpoint:
        raise ValueError("Provide either --predictions_json or --checkpoint.")
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1.")


def load_prediction_records(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    if args.predictions_json is not None:
        with open(args.predictions_json, "r") as f:
            payload = json.load(f)
        return payload["records"]

    checkpoint = torch.load(args.checkpoint, map_location=device)
    transform = imagenet_preprocess(args.image_size)
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    dataset = PolicyDataset(
        args.data_json,
        transform=transform,
        manifest_path=args.manifest,
        input_mode=input_mode,
        env_option_id=checkpoint.get("env_option_id"),
        env_option_ids=checkpoint.get("env_option_ids"),
        include_ae_input=bool(checkpoint.get("include_ae_input", False)),
        input_variant=checkpoint.get("input_variant") or "real",
        noise_seed=checkpoint.get("noise_seed", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    backbone_name = infer_backbone_name_from_checkpoint(checkpoint)
    state_dict = normalize_policy_checkpoint_state_dict(
        checkpoint["model_state_dict"],
        backbone_name,
    )
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
        backbone_name=backbone_name,
        input_mode=input_mode,
        num_input_views=infer_num_input_views_from_checkpoint(checkpoint),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Policy inference"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            top1_confs, top1_preds = torch.max(probs, dim=-1)

            for sample_id, pred, conf in zip(batch["sample_id"], top1_preds, top1_confs):
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "pred_best_index": int(pred.item()),
                        "top1_confidence": float(conf.item()),
                    }
                )

    return records


def build_manifest_index(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    dataset = ManifestLensDataset(manifest_path)
    manifest_index: Dict[str, Dict[str, Any]] = {}
    for item in dataset.items:
        sample_id = item.get("id")
        if sample_id is None:
            raise KeyError("Manifest item is missing required key 'id'.")
        manifest_index[str(sample_id)] = item
    return manifest_index


def build_data_index(data_json_path: str) -> Dict[str, Dict[str, Any]]:
    with open(data_json_path, "r") as f:
        items = json.load(f)
    return {str(item["sample_id"]): item for item in items}


def build_candidate_tensor(
    candidates: List[Dict[str, Any]],
    transform,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    option_id_to_pos = {
        int(candidate["option_id"]): pos for pos, candidate in enumerate(candidates)
    }
    return torch.stack(images, dim=0), option_id_to_pos


def resize_for_panel(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(max_side / max(1, w), max_side / max(1, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def pad_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_text_lines(title: str, meta: Dict[str, Any]) -> List[str]:
    lines = [title]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    lines: List[str],
    font: ImageFont.ImageFont,
    max_width: int,
    fill: Tuple[int, int, int],
) -> int:
    x, y = xy
    line_height = 0
    for raw_line in lines:
        words = str(raw_line).split(" ")
        current = ""
        wrapped = []
        for word in words:
            candidate = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        if current:
            wrapped.append(current)

        for line in wrapped:
            draw.text((x, y), line, font=font, fill=fill)
            bbox = draw.textbbox((x, y), line, font=font)
            h = bbox[3] - bbox[1]
            y += h + 4
            line_height += h + 4
    return line_height


def create_case_figure(
    output_path: str,
    sample_id: str,
    case_title: str,
    panels: List[Tuple[Image.Image, List[str]]],
    footer_lines: List[str],
    max_side: int,
) -> None:
    font = ImageFont.load_default()
    panel_images = [resize_for_panel(image, max_side=max_side) for image, _ in panels]
    panel_w = max(image.width for image in panel_images)
    panel_h = max(image.height for image in panel_images)
    text_w = panel_w
    inner_pad = 14
    top_pad = 20
    title_h = 28
    text_block_h = 120
    footer_h = 90
    panel_total_h = top_pad + title_h + panel_h + 10 + text_block_h + 12
    total_w = len(panels) * (panel_w + inner_pad) + inner_pad
    total_h = panel_total_h + footer_h

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((inner_pad, 8), f"{sample_id} | {case_title}", font=font, fill=(0, 0, 0))

    x = inner_pad
    for (image, text_lines), panel_image in zip(panels, panel_images):
        draw.rounded_rectangle(
            (x - 4, top_pad - 4, x + panel_w + 4, panel_total_h - 10),
            radius=8,
            outline=(220, 220, 220),
            width=1,
            fill=(252, 252, 252),
        )
        padded = pad_panel(panel_image, panel_w, panel_h)
        canvas.paste(padded, (x, top_pad + title_h))
        draw_wrapped_text(
            draw=draw,
            xy=(x, top_pad + title_h + panel_h + 10),
            lines=text_lines,
            font=font,
            max_width=text_w,
            fill=(15, 15, 15),
        )
        x += panel_w + inner_pad

    footer_y = panel_total_h + 6
    draw.line((inner_pad, footer_y, total_w - inner_pad, footer_y), fill=(225, 225, 225), width=1)
    draw_wrapped_text(
        draw=draw,
        xy=(inner_pad, footer_y + 10),
        lines=footer_lines,
        font=font,
        max_width=total_w - 2 * inner_pad,
        fill=(0, 0, 0),
    )
    canvas.save(output_path)


def choose_target_option(item: Dict[str, Any]) -> int:
    if "best_option_id" in item:
        return int(item["best_option_id"])
    raise KeyError("Data item is missing 'best_option_id'.")


def matches_filter(case_record: Dict[str, Any], filter_mode: str) -> bool:
    if filter_mode == "all":
        return True
    if filter_mode == "policy_wrong":
        return not case_record["policy_class_correct"]
    if filter_mode == "policy_wrong_lens_right":
        return (not case_record["policy_class_correct"]) and case_record["lens_class_correct"]
    if filter_mode == "policy_right_lens_wrong":
        return case_record["policy_class_correct"] and (not case_record["lens_class_correct"])
    if filter_mode == "different_choice":
        return case_record["policy_pred_best_index"] != case_record["lens_best_index"]
    raise ValueError(f"Unsupported filter_mode={filter_mode}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)

    prediction_records = load_prediction_records(args, device)
    manifest_index = build_manifest_index(args.manifest)
    data_index = build_data_index(args.data_json)
    checkpoint_env_option_id = None
    if args.checkpoint is not None:
        checkpoint_meta = torch.load(args.checkpoint, map_location="cpu")
        checkpoint_env_option_id = checkpoint_meta.get("env_option_id")

    if args.sample_ids:
        wanted_ids = {str(sample_id) for sample_id in args.sample_ids}
        prediction_records = [
            record for record in prediction_records if str(record["sample_id"]) in wanted_ids
        ]

    evaluated_cases: List[Dict[str, Any]] = []

    for record in tqdm(prediction_records, desc="Collect visualization cases"):
        sample_id = str(record["sample_id"])
        manifest_item = manifest_index.get(sample_id)
        data_item = data_index.get(sample_id)
        if manifest_item is None or data_item is None:
            continue

        label = int(manifest_item["label"])
        candidates = manifest_item["candidates"]
        candidate_tensor, option_id_to_pos = build_candidate_tensor(candidates, transform)

        pred_best_index = int(record["pred_best_index"])
        if pred_best_index not in option_id_to_pos:
            continue

        target_option_id = choose_target_option(data_item)
        if target_option_id not in option_id_to_pos:
            continue
        env_input_path = data_item["baseline_path"]
        if checkpoint_env_option_id is not None and checkpoint_env_option_id in option_id_to_pos:
            env_input_path = candidates[option_id_to_pos[checkpoint_env_option_id]]["path"]

        with torch.no_grad():
            logits = classifier(candidate_tensor.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values

        policy_pos = option_id_to_pos[pred_best_index]
        lens_pos = int(torch.argmax(conf).item())
        lens_option_id = int(candidates[lens_pos]["option_id"])
        target_pos = option_id_to_pos[target_option_id]

        policy_pred_label = int(torch.argmax(logits[policy_pos]).item())
        lens_pred_label = int(torch.argmax(logits[lens_pos]).item())
        target_pred_label = int(torch.argmax(logits[target_pos]).item())

        case_record = {
            "sample_id": sample_id,
            "label": label,
            "policy_pred_best_index": pred_best_index,
            "policy_pred_confidence": float(record.get("top1_confidence", 0.0)),
            "policy_class_prediction": policy_pred_label,
            "policy_class_correct": policy_pred_label == label,
            "lens_best_index": lens_option_id,
            "lens_confidence": float(conf[lens_pos].item()),
            "lens_class_prediction": lens_pred_label,
            "lens_class_correct": lens_pred_label == label,
            "target_best_index": target_option_id,
            "target_class_prediction": target_pred_label,
            "target_class_correct": target_pred_label == label,
            "baseline_path": data_item["baseline_path"],
            "env_input_path": env_input_path,
            "policy_path": candidates[policy_pos]["path"],
            "lens_path": candidates[lens_pos]["path"],
            "target_path": data_item.get("best_path", candidates[target_pos]["path"]),
        }
        if matches_filter(case_record, args.filter_mode):
            evaluated_cases.append(case_record)

    if not evaluated_cases:
        raise RuntimeError("No cases matched the requested filters.")

    if args.sample_ids:
        selected_cases = evaluated_cases[: args.num_samples]
    else:
        rng.shuffle(evaluated_cases)
        selected_cases = evaluated_cases[: args.num_samples]

    summary_records: List[Dict[str, Any]] = []
    for idx, case in enumerate(selected_cases, start=1):
        sample_id = case["sample_id"]
        title = (
            f"policy_idx={case['policy_pred_best_index']} | "
            f"lens_idx={case['lens_best_index']} | "
            f"target_idx={case['target_best_index']}"
        )
        panels = [
            (
                load_image_rgb(case["baseline_path"]),
                make_text_lines(
                    "Policy input",
                    {
                        "path": case["baseline_path"],
                    },
                ),
            ),
        ]
        if case.get("env_input_path") and case["env_input_path"] != case["baseline_path"]:
            panels.append(
                (
                    load_image_rgb(case["env_input_path"]),
                    make_text_lines(
                        "Env input",
                        {
                            "path": case["env_input_path"],
                        },
                    ),
                )
            )
        panels.extend([
            (
                load_image_rgb(case["policy_path"]),
                make_text_lines(
                    "Policy -> downstream",
                    {
                        "option_id": case["policy_pred_best_index"],
                        "policy_conf": f"{case['policy_pred_confidence']:.3f}",
                        "cls_pred": case["policy_class_prediction"],
                        "correct": case["policy_class_correct"],
                    },
                ),
            ),
            (
                load_image_rgb(case["lens_path"]),
                make_text_lines(
                    "Lens -> downstream",
                    {
                        "option_id": case["lens_best_index"],
                        "lens_conf": f"{case['lens_confidence']:.3f}",
                        "cls_pred": case["lens_class_prediction"],
                        "correct": case["lens_class_correct"],
                    },
                ),
            ),
            (
                load_image_rgb(case["target_path"]),
                make_text_lines(
                    "Target best",
                    {
                        "option_id": case["target_best_index"],
                        "cls_pred": case["target_class_prediction"],
                        "correct": case["target_class_correct"],
                    },
                ),
            ),
        ])
        footer_lines = [
            f"GT label: {case['label']}",
            f"baseline_path: {case['baseline_path']}",
            f"env_input_path: {case['env_input_path']}",
            f"policy_path: {case['policy_path']}",
            f"lens_path: {case['lens_path']}",
            f"target_path: {case['target_path']}",
        ]
        output_name = f"{idx:02d}_{sample_id.replace('/', '_')}.png"
        output_path = os.path.join(args.output_dir, output_name)
        create_case_figure(
            output_path=output_path,
            sample_id=sample_id,
            case_title=title,
            panels=panels,
            footer_lines=footer_lines,
            max_side=args.max_side,
        )
        case["output_png"] = output_name
        summary_records.append(case)

    summary = {
        "config": {
            "manifest": args.manifest,
            "data_json": args.data_json,
            "predictions_json": args.predictions_json,
            "checkpoint": args.checkpoint,
            "model": args.model,
            "filter_mode": args.filter_mode,
            "num_samples": len(summary_records),
        },
        "records": summary_records,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(summary_records)} visualizations to {args.output_dir}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
