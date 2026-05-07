from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LENS_DIR = ROOT / "lens"
OPENCLIP_DIR = ROOT / "openclip_ds_policy"

for extra_path in (ROOT, LENS_DIR, OPENCLIP_DIR):
    path = str(extra_path)
    if path not in sys.path:
        sys.path.insert(0, path)

from lens.data_utils import ManifestLensDataset, load_image_rgb
from openclip200 import (
    DEFAULT_PROMPT_TEMPLATE,
    OpenCLIP200Classifier,
    filter_manifest_items,
    get_subset_label_ids,
    parse_class_id,
    parse_group_id,
    resolve_ae_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenCLIP cosine soft labels for policy training.")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/manifest_all.json"))
    parser.add_argument("--class_index_json", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/imagenet_class_index.json"))
    parser.add_argument("--split_dir", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/policy_labels"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "openclip_ds_policy/labels/tau_0p05"))
    parser.add_argument("--openclip_model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip_pretrained", type=str, default="openai")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--num_candidates", type=int, default=27)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_candidate_tensor(classifier: OpenCLIP200Classifier, candidates: List[Dict[str, Any]]) -> torch.Tensor:
    return classifier.preprocess_images(load_image_rgb(candidate["path"]) for candidate in candidates)


def build_record(
    item: Dict[str, Any],
    classifier: OpenCLIP200Classifier,
    tau: float,
    num_candidates: int,
) -> Dict[str, Any]:
    sample_id = str(item["id"])
    raw_label = int(item["label"])
    candidates = item["candidates"]
    candidate_tensor = build_candidate_tensor(classifier, candidates)
    similarity = classifier.similarity_from_images(candidate_tensor).detach().cpu()
    gt_idx = classifier.gt_subset_index(raw_label)
    gt_scores = similarity[:, gt_idx]
    soft_by_position = torch.softmax(gt_scores / tau, dim=0)

    soft_target = torch.zeros(num_candidates, dtype=torch.float32)
    for pos, candidate in enumerate(candidates):
        soft_target[int(candidate["option_id"])] = soft_by_position[pos]

    best_pos = int(torch.argmax(soft_by_position).item())
    best_candidate = candidates[best_pos]
    candidate_pred_raw = classifier.pred_raw_labels(similarity).tolist()
    candidate_correct = [int(pred) == raw_label for pred in candidate_pred_raw]
    ae_path = resolve_ae_path(ROOT, sample_id, candidates[0]["path"])

    return {
        "sample_id": sample_id,
        "group_id": parse_group_id(sample_id),
        "class_id": parse_class_id(sample_id),
        "env": item["env"],
        "label": raw_label,
        "gt_subset_index": gt_idx,
        "baseline_path": ae_path,
        "best_idx_in_candidates": best_pos,
        "best_option_id": int(best_candidate["option_id"]),
        "best_option_name": best_candidate["meta"]["option_name"],
        "best_path": best_candidate["path"],
        "soft_target": [float(value) for value in soft_target.tolist()],
        "sample_weight": 1.0,
        "openclip_gt_scores_by_position": [float(value) for value in gt_scores.tolist()],
        "openclip_candidate_pred_labels": [int(value) for value in candidate_pred_raw],
        "openclip_candidate_correct": candidate_correct,
        "openclip_num_correct_candidates": int(sum(candidate_correct)),
    }


def main() -> None:
    args = parse_args()
    if args.tau <= 0:
        raise ValueError("--tau must be positive.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ManifestLensDataset(args.manifest)
    all_items = dataset.items
    label_ids = get_subset_label_ids(all_items)
    classifier = OpenCLIP200Classifier(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
        class_index_json=args.class_index_json,
        label_ids=label_ids,
        device=device,
        prompt_template=args.prompt_template,
    )

    split_to_file = {
        "train": "policy_train_labels.json",
        "val": "policy_val_labels.json",
        "test": "policy_test_labels.json",
    }
    summary: Dict[str, Any] = {
        "manifest": args.manifest,
        "split_dir": args.split_dir,
        "class_index_json": args.class_index_json,
        "openclip_model": args.openclip_model,
        "openclip_pretrained": args.openclip_pretrained,
        "prompt_template": args.prompt_template,
        "tau": args.tau,
        "num_subset_classes": len(label_ids),
        "subset_label_ids": label_ids,
        "splits": {},
    }

    for split_name, split_file in split_to_file.items():
        split_json = Path(args.split_dir) / split_file
        items = filter_manifest_items(all_items, split_json)
        records = [
            build_record(
                item=item,
                classifier=classifier,
                tau=args.tau,
                num_candidates=args.num_candidates,
            )
            for item in tqdm(items, desc=f"Generate OpenCLIP soft labels/{split_name}")
        ]
        output_path = Path(args.output_dir) / f"openclip_soft_{split_name}_labels.json"
        save_json(output_path, records)
        summary["splits"][split_name] = {
            "source_json": str(split_json),
            "output_json": str(output_path),
            "num_samples": len(records),
            "avg_num_correct_candidates": (
                sum(record["openclip_num_correct_candidates"] for record in records)
                / max(1, len(records))
            ),
        }

    save_json(Path(args.output_dir) / "openclip_soft_label_summary.json", summary)
    print(summary)
    print(f"Saved OpenCLIP soft labels to {args.output_dir}")


if __name__ == "__main__":
    main()
