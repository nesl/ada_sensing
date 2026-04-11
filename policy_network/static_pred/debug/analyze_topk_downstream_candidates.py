from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze downstream classifier accuracy for the policy network's top-k "
            "predicted indices. Useful for checking whether top-1 is wrong while "
            "other high-confidence policy candidates are still downstream-correct."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--predictions_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_json", type=str, default=None)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    has_prediction_file = args.predictions_json is not None
    has_checkpoint_inputs = args.checkpoint is not None or args.data_json is not None

    if has_prediction_file and has_checkpoint_inputs:
        raise ValueError("Use either --predictions_json or (--checkpoint and --data_json), not both.")

    if not has_prediction_file and not (args.checkpoint and args.data_json):
        raise ValueError(
            "Provide --predictions_json, or provide both --checkpoint and --data_json."
        )

    if args.topk < 1:
        raise ValueError("--topk must be >= 1.")


def normalize_checkpoint_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Support both the current policy checkpoint layout and an older layout used by
    experiment A, where the whole MobileNet was stored under:
    - backbone.features.*
    - backbone.classifier.0 / backbone.classifier.1 / backbone.classifier.2
    - backbone.classifier.3  (policy head)
    """
    if any(key.startswith("backbone.features.") for key in raw_state_dict):
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in raw_state_dict.items():
            if key.startswith("backbone.features."):
                new_key = "backbone." + key[len("backbone.features."):]
            elif key.startswith("backbone.classifier.0"):
                new_key = "feature_proj.0" + key[len("backbone.classifier.0"):]
            elif key.startswith("backbone.classifier.1"):
                new_key = "feature_proj.1" + key[len("backbone.classifier.1"):]
            elif key.startswith("backbone.classifier.2"):
                new_key = "feature_proj.2" + key[len("backbone.classifier.2"):]
            elif key.startswith("backbone.classifier.3"):
                new_key = "policy_head" + key[len("backbone.classifier.3"):]
            else:
                new_key = key
            normalized[new_key] = value
        return normalized

    return raw_state_dict


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
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    state_dict = normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    backbone_name = infer_backbone_name_from_checkpoint(checkpoint)
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
        backbone_name=backbone_name,
        input_mode=input_mode,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Policy inference"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            topk = min(args.topk, probs.shape[-1])
            topk_confs, topk_preds = torch.topk(probs, k=topk, dim=-1)
            top1_confs, top1_preds = torch.max(probs, dim=-1)

            for sample_id, pred, conf, topk_pred, topk_conf in zip(
                batch["sample_id"], top1_preds, top1_confs, topk_preds, topk_confs
            ):
                records.append(
                    {
                        "sample_id": sample_id,
                        "pred_best_index": int(pred.item()),
                        "top1_confidence": float(conf.item()),
                        "top5_pred_indices": [int(x) for x in topk_pred.tolist()],
                        "top5_confidences": [float(x) for x in topk_conf.tolist()],
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


def build_candidate_tensor(
    candidates: List[Dict[str, Any]],
    transform,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    option_id_to_pos = {
        int(candidate["option_id"]): pos for pos, candidate in enumerate(candidates)
    }
    return torch.stack(images, dim=0), option_id_to_pos


def accuracy_payload(correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "acc": 100.0 * correct / max(1, total),
    }


def mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)

    prediction_records = load_prediction_records(args, device)
    manifest_index = build_manifest_index(args.manifest)

    rank_correct = [0 for _ in range(args.topk)]
    cumulative_correct = [0 for _ in range(args.topk)]
    top1_wrong_but_topk_fixable = [0 for _ in range(args.topk)]
    rank_conf_sums = [0.0 for _ in range(args.topk)]
    rank_conf_counts = [0 for _ in range(args.topk)]
    per_sample: List[Dict[str, Any]] = []
    missing_samples: List[str] = []
    total = 0

    for record in tqdm(prediction_records, desc="Analyze top-k downstream"):
        sample_id = str(record["sample_id"])
        manifest_item = manifest_index.get(sample_id)
        if manifest_item is None:
            missing_samples.append(sample_id)
            continue

        label = int(manifest_item["label"])
        candidates = manifest_item["candidates"]
        candidate_tensor, option_id_to_pos = build_candidate_tensor(candidates, transform)

        with torch.no_grad():
            logits = classifier(candidate_tensor.to(device, non_blocking=True))
            class_probs = torch.softmax(logits, dim=-1)
            candidate_pred_labels = torch.argmax(logits, dim=-1)
            candidate_pred_confs = class_probs.max(dim=-1).values

        policy_top_indices = record.get("top5_pred_indices")
        policy_top_confs = record.get("top5_confidences")
        if not policy_top_indices or not policy_top_confs:
            policy_top_indices = [int(record["pred_best_index"])]
            policy_top_confs = [float(record["top1_confidence"])]

        max_k = min(args.topk, len(policy_top_indices))
        if max_k == 0:
            raise ValueError(f"No top-k predictions found for sample {sample_id}.")

        sample_rank_records: List[Dict[str, Any]] = []
        prefix_has_correct = False
        top1_is_correct = False

        for rank in range(max_k):
            option_id = int(policy_top_indices[rank])
            policy_conf = float(policy_top_confs[rank])
            rank_conf_sums[rank] += policy_conf
            rank_conf_counts[rank] += 1
            if option_id not in option_id_to_pos:
                raise ValueError(
                    f"Predicted option_id {option_id} not found in manifest candidates for {sample_id}."
                )

            pos = option_id_to_pos[option_id]
            downstream_pred = int(candidate_pred_labels[pos].item())
            downstream_conf = float(candidate_pred_confs[pos].item())
            downstream_correct = downstream_pred == label

            rank_correct[rank] += int(downstream_correct)
            prefix_has_correct = prefix_has_correct or downstream_correct
            cumulative_correct[rank] += int(prefix_has_correct)

            if rank == 0:
                top1_is_correct = downstream_correct

            if not top1_is_correct and prefix_has_correct:
                top1_wrong_but_topk_fixable[rank] += 1

            sample_rank_records.append(
                {
                    "rank": rank + 1,
                    "option_id": option_id,
                    "policy_confidence": policy_conf,
                    "downstream_pred_label": downstream_pred,
                    "downstream_pred_confidence": downstream_conf,
                    "downstream_correct": downstream_correct,
                }
            )

        per_sample.append(
            {
                "sample_id": sample_id,
                "label": label,
                "policy_topk": sample_rank_records,
                "top1_downstream_correct": top1_is_correct,
                "topk_contains_downstream_correct": prefix_has_correct,
            }
        )
        total += 1

    rank_summary = {
        f"rank_{rank + 1}": accuracy_payload(rank_correct[rank], total)
        for rank in range(args.topk)
    }
    cumulative_summary = {
        f"top_{rank + 1}_contains_correct": accuracy_payload(cumulative_correct[rank], total)
        for rank in range(args.topk)
    }
    fixable_summary = {
        f"top1_wrong_but_top_{rank + 1}_contains_correct": accuracy_payload(
            top1_wrong_but_topk_fixable[rank], total
        )
        for rank in range(args.topk)
    }
    average_rank_confidence = {
        f"rank_{rank + 1}": (
            rank_conf_sums[rank] / rank_conf_counts[rank] if rank_conf_counts[rank] else 0.0
        )
        for rank in range(args.topk)
    }

    result = {
        "config": {
            "manifest": args.manifest,
            "predictions_json": args.predictions_json,
            "checkpoint": args.checkpoint,
            "data_json": args.data_json,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
            "topk": args.topk,
        },
        "summary": {
            "evaluated_samples": total,
            "missing_manifest_samples": len(missing_samples),
            "average_policy_confidence_by_rank": average_rank_confidence,
            "rankwise_downstream_acc": rank_summary, # use only image of index k for downstream acc
            "cumulative_topk_contains_correct": cumulative_summary, # acc of at least one of topk hit the gt
            "top1_wrong_but_recoverable": fixable_summary, # ratio of wrong top1 but can be fixed by looking at more candidates
        },
        "missing_samples": missing_samples,
        "records": per_sample,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved analysis to {args.output_json}")


if __name__ == "__main__":
    main()
