from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]
LENS_DIR = ROOT / "lens"

for extra_path in (ROOT, LENS_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute oracle upper bound on a manifest: a sample counts as correct "
            "if any candidate image is classified correctly."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_oracle_upper_bound.json",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json",
        help="Optional policy label json used to filter the manifest to a specific split.",
    )
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_candidate_tensor(candidates: List[Dict[str, Any]], transform) -> torch.Tensor:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    return torch.stack(images, dim=0)


def load_allowed_sample_ids(data_json: Optional[str]) -> Optional[Set[str]]:
    if data_json is None:
        return None

    with open(data_json, "r") as f:
        records: List[Dict[str, Any]] = json.load(f)
    return {str(record["sample_id"]) for record in records}


def filter_manifest_items(
    items: List[Dict[str, Any]],
    allowed_sample_ids: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    if allowed_sample_ids is None:
        return items
    return [item for item in items if str(item.get("id")) in allowed_sample_ids]


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ManifestLensDataset(args.manifest)
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)
    allowed_sample_ids = load_allowed_sample_ids(args.data_json)
    items = filter_manifest_items(dataset.items, allowed_sample_ids)

    total = 0
    oracle_correct = 0
    per_sample: List[Dict[str, Any]] = []

    for item in tqdm(items, desc="Oracle upper bound"):
        sample_id = item.get("id")
        label = int(item["label"])
        candidates = item["candidates"]
        candidate_tensor = build_candidate_tensor(candidates, transform)

        with torch.no_grad():
            logits = classifier(candidate_tensor.to(device, non_blocking=True))
            preds = torch.argmax(logits, dim=-1)

        correct_positions = (preds == label).nonzero(as_tuple=False).flatten().tolist()
        hit = len(correct_positions) > 0
        oracle_correct += int(hit)
        total += 1

        correct_option_ids = [int(candidates[pos]["option_id"]) for pos in correct_positions]
        per_sample.append(
            {
                "sample_id": sample_id,
                "label": label,
                "num_candidates": len(candidates),
                "oracle_hit": hit,
                "num_correct_candidates": len(correct_positions),
                "correct_candidate_positions": correct_positions,
                "correct_option_ids": correct_option_ids,
            }
        )

    result = {
        "manifest": args.manifest,
        "data_json": args.data_json,
        "model": args.model,
        "evaluated_samples": total,
        "oracle_upper_bound": {
            "correct": oracle_correct,
            "total": total,
            "acc": 100.0 * oracle_correct / max(1, total),
        },
        "per_sample": per_sample,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["oracle_upper_bound"], indent=2))
    print(f"Saved oracle evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
