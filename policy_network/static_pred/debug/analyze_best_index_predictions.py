from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from policy_dataset import PolicyDataset
from policy_model import SensorPolicyNetwork
from utils import imagenet_preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_json", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def evaluate_predictions(model, loader, device: torch.device) -> Dict[str, Any]:
    model.eval()

    records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            sample_ids = batch["sample_id"]
            for sample_id, target, pred, conf in zip(sample_ids, targets, preds, confs):
                target_int = int(target.item())
                pred_int = int(pred.item())
                conf_float = float(conf.item())
                y_true.append(target_int)
                y_pred.append(pred_int)
                confidences.append(conf_float)
                records.append({
                    "sample_id": sample_id,
                    "target_best_index": target_int,
                    "pred_best_index": pred_int,
                    "top1_confidence": conf_float,
                })

    total = len(records)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    majority_label, majority_count = true_counter.most_common(1)[0]
    sorted_conf = sorted(confidences)

    return {
        "summary": {
            "total": total,
            "correct": correct,
            "acc": 100.0 * correct / max(1, total),
            "majority_label": majority_label,
            "majority_baseline_acc": 100.0 * majority_count / max(1, total),
            "num_true_classes": len(true_counter),
            "num_pred_classes": len(pred_counter),
            "mean_confidence": sum(confidences) / max(1, len(confidences)),
            "median_confidence": sorted_conf[len(sorted_conf) // 2] if sorted_conf else 0.0,
            "high_conf_ratio_0_9": sum(c > 0.9 for c in confidences) / max(1, len(confidences)),
        },
        "true_best_index_distribution": dict(sorted(true_counter.items())),
        "pred_best_index_distribution": dict(sorted(pred_counter.items())),
        "top10_true_best_indices": true_counter.most_common(10),
        "top10_pred_best_indices": pred_counter.most_common(10),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)
    dataset = PolicyDataset(args.data_json, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    result = evaluate_predictions(model, loader, device)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    print("Top predicted best indices:", result["top10_pred_best_indices"])
    print(f"Saved analysis to {args.output_json}")


if __name__ == "__main__":
    main()
