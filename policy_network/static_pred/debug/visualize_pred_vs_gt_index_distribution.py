from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "policy_network" / "static_pred"
DEFAULT_DATA_JSON = (
    ROOT
    / "data"
    / "ImageNet-ES-Diverse"
    / "oracle_policy_labels"
    / "oracle_policy_test_labels.json"
)
DEFAULT_MANIFEST = ROOT / "data" / "ImageNet-ES-Diverse" / "manifest_all.json"
DEFAULT_OUTPUT_PNG = (
    ROOT / "policy_network" / "vis_results" / "pred_vs_gt_index_distribution.png"
)
DEFAULT_OUTPUT_JSON = (
    ROOT / "policy_network" / "results_debug" / "pred_vs_gt_index_distribution.json"
)

for extra_path in (ROOT, POLICY_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from policy_dataset import PolicyDataset
from policy_model import (
    SensorPolicyNetwork,
    infer_backbone_name_from_checkpoint,
    infer_input_mode_from_checkpoint,
    normalize_policy_checkpoint_state_dict,
)
from utils import imagenet_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a policy checkpoint's predicted test index distribution "
            "against the ground-truth best_option_id distribution."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_json", type=str, default=str(DEFAULT_DATA_JSON))
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output_png", type=str, default=str(DEFAULT_OUTPUT_PNG))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_candidates", type=int, default=27)
    return parser.parse_args()


def infer_run_name(checkpoint_path: Path) -> str:
    parent = checkpoint_path.parent
    grandparent = parent.parent
    if grandparent.name.startswith("fixed_k_"):
        return f"{grandparent.name}/{parent.name}/{checkpoint_path.name}"
    return f"{parent.name}/{checkpoint_path.name}"


def build_loader(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> DataLoader:
    transform = imagenet_preprocess(args.image_size)
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    dataset = PolicyDataset(
        args.data_json,
        transform=transform,
        manifest_path=args.manifest,
        input_mode=input_mode,
        env_option_id=checkpoint.get("env_option_id"),
        input_variant=checkpoint.get("input_variant") or "real",
        noise_seed=checkpoint.get("noise_seed", 0),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def build_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> SensorPolicyNetwork:
    backbone_name = infer_backbone_name_from_checkpoint(checkpoint)
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    state_dict = normalize_policy_checkpoint_state_dict(
        checkpoint["model_state_dict"],
        backbone_name,
    )
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
        backbone_name=backbone_name,
        input_mode=input_mode,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_predictions(
    model: SensorPolicyNetwork,
    loader: DataLoader,
    device: torch.device,
    num_candidates: int,
) -> Tuple[List[int], List[int], List[Dict[str, Any]], List[int], List[int]]:
    gt_indices: List[int] = []
    pred_indices: List[int] = []
    confidences: List[float] = []
    records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            for sample_id, target, pred, conf in zip(
                batch["sample_id"],
                targets,
                preds,
                confs,
            ):
                target_int = int(target.item())
                pred_int = int(pred.item())
                conf_float = float(conf.item())
                gt_indices.append(target_int)
                pred_indices.append(pred_int)
                confidences.append(conf_float)
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "target_best_index": target_int,
                        "pred_best_index": pred_int,
                        "top1_confidence": conf_float,
                        "top1_hit": target_int == pred_int,
                    }
                )

    gt_counts = counter_to_list(Counter(gt_indices), num_candidates)
    pred_counts = counter_to_list(Counter(pred_indices), num_candidates)
    return gt_indices, pred_indices, records, gt_counts, pred_counts


def counter_to_list(counter: Counter, num_candidates: int) -> List[int]:
    return [int(counter.get(option_id, 0)) for option_id in range(num_candidates)]


def load_option_names(data_json: Path, num_candidates: int) -> List[str]:
    with open(data_json, "r") as f:
        items = json.load(f)
    option_names = ["" for _ in range(num_candidates)]
    for item in items:
        if "best_option_id" not in item or "best_option_name" not in item:
            continue
        option_id = int(item["best_option_id"])
        if 0 <= option_id < num_candidates:
            option_names[option_id] = str(item["best_option_name"])
    return option_names


def save_json(
    output_json: Path,
    args: argparse.Namespace,
    checkpoint: Dict[str, Any],
    gt_counts: List[int],
    pred_counts: List[int],
    records: List[Dict[str, Any]],
) -> None:
    total = len(records)
    top1_correct = sum(int(record["top1_hit"]) for record in records)
    payload = {
        "checkpoint": args.checkpoint,
        "data_json": args.data_json,
        "manifest": args.manifest,
        "checkpoint_metadata": {
            "backbone_name": infer_backbone_name_from_checkpoint(checkpoint),
            "input_mode": infer_input_mode_from_checkpoint(checkpoint),
            "env_option_id": checkpoint.get("env_option_id"),
            "input_variant": checkpoint.get("input_variant") or "real",
            "loss_type": checkpoint.get("loss_type"),
            "trainable_scope": checkpoint.get("trainable_scope"),
            "effective_trainable_scope": checkpoint.get("effective_trainable_scope"),
            "epoch": checkpoint.get("epoch"),
            "best_val_acc": checkpoint.get("best_val_acc"),
        },
        "summary": {
            "total": total,
            "top1_correct": top1_correct,
            "top1_acc": 100.0 * top1_correct / max(1, total),
            "gt_num_nonzero_indices": sum(int(count > 0) for count in gt_counts),
            "pred_num_nonzero_indices": sum(int(count > 0) for count in pred_counts),
        },
        "gt_best_index_distribution": gt_counts,
        "pred_best_index_distribution": pred_counts,
        "records": records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)


def plot_distribution(
    output_png: Path,
    checkpoint_path: Path,
    data_json: Path,
    gt_counts: List[int],
    pred_counts: List[int],
    option_names: List[str],
) -> None:
    option_ids = np.arange(len(gt_counts))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    ax.bar(option_ids - width / 2, gt_counts, width=width, label="GT test")
    ax.bar(option_ids + width / 2, pred_counts, width=width, label="Pred test")
    ax.set_title(
        "Predicted vs Ground-Truth Test Index Distribution\n"
        f"{infer_run_name(checkpoint_path)}"
    )
    ax.set_xlabel("option_id")
    ax.set_ylabel("count")
    ax.set_xticks(option_ids)
    ax.set_xticklabels(
        [f"{option_id}\n{option_names[option_id]}" for option_id in option_ids],
        fontsize=8,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.text(0.01, 0.01, str(data_json), fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    data_json = Path(args.data_json)
    output_png = Path(args.output_png)
    output_json = Path(args.output_json)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_candidates = int(checkpoint.get("num_candidates", args.num_candidates))
    loader = build_loader(args, checkpoint)
    model = build_model(checkpoint, device)

    _, _, records, gt_counts, pred_counts = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        num_candidates=num_candidates,
    )
    option_names = load_option_names(data_json, num_candidates)

    save_json(
        output_json=output_json,
        args=args,
        checkpoint=checkpoint,
        gt_counts=gt_counts,
        pred_counts=pred_counts,
        records=records,
    )
    plot_distribution(
        output_png=output_png,
        checkpoint_path=checkpoint_path,
        data_json=data_json,
        gt_counts=gt_counts,
        pred_counts=pred_counts,
        option_names=option_names,
    )

    print(f"Saved plot to {output_png}")
    print(f"Saved analysis to {output_json}")


if __name__ == "__main__":
    main()
