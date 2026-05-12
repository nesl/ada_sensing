from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

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
        description="Precompute downstream classifier correctness for real candidates."
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_eval_sample_ids(data_json: str) -> Set[str]:
    with open(data_json, "r") as f:
        items = json.load(f)
    return {str(item["sample_id"]) for item in items}


def build_manifest_index(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    dataset = ManifestLensDataset(manifest_path)
    manifest_index: Dict[str, Dict[str, Any]] = {}
    for item in dataset.items:
        sample_id = item.get("id")
        if sample_id is None:
            raise KeyError("Manifest item is missing required key 'id'.")
        manifest_index[str(sample_id)] = item
    return manifest_index


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)
    classifier.eval()

    eval_sample_ids = load_eval_sample_ids(args.data_json)
    manifest_index = build_manifest_index(args.manifest)

    records: Dict[str, Any] = {}
    missing_samples: List[str] = []
    with torch.no_grad():
        for sample_id in tqdm(sorted(eval_sample_ids), desc="Precompute downstream cache"):
            manifest_item = manifest_index.get(sample_id)
            if manifest_item is None:
                missing_samples.append(sample_id)
                continue

            label = int(manifest_item["label"])
            candidates = manifest_item["candidates"]
            images = [
                transform(load_image_rgb(candidate["path"]))
                for candidate in candidates
            ]
            candidate_tensor = torch.stack(images, dim=0).to(device, non_blocking=True)
            logits = classifier(candidate_tensor)
            class_probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(logits, dim=-1)
            pred_confs = class_probs.max(dim=-1).values

            candidate_records: Dict[str, Any] = {}
            for candidate, pred_label, pred_conf in zip(
                candidates,
                pred_labels,
                pred_confs,
            ):
                option_id = int(candidate["option_id"])
                pred_label_int = int(pred_label.item())
                candidate_records[str(option_id)] = {
                    "pred_label": pred_label_int,
                    "pred_confidence": float(pred_conf.item()),
                    "correct": pred_label_int == label,
                }

            records[sample_id] = {
                "label": label,
                "candidates": candidate_records,
            }

    output = {
        "config": {
            "manifest": args.manifest,
            "data_json": args.data_json,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
        },
        "missing_samples": missing_samples,
        "records": records,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f)

    print(f"Wrote downstream cache for {len(records)} samples to {args.output_json}")
    if missing_samples:
        print(f"Missing {len(missing_samples)} samples from manifest.")


if __name__ == "__main__":
    main()
