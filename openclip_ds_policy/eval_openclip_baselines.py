from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
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
    build_option_name_map,
    filter_manifest_items,
    get_subset_label_ids,
    resolve_ae_path,
    save_json,
    summarize_binary_hits,
    summarize_float_hits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenCLIP-200 AE/Random/Oracle-S/Oracle-F/Lens baselines.")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/manifest_all.json"))
    parser.add_argument("--data_json", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json"))
    parser.add_argument("--class_index_json", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/imagenet_class_index.json"))
    parser.add_argument("--output_json", type=str, default=str(ROOT / "openclip_ds_policy/results/baselines/openclip_baselines_test.json"))
    parser.add_argument("--openclip_model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip_pretrained", type=str, default="openai")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def candidate_tensor(classifier: OpenCLIP200Classifier, candidates: List[Dict[str, Any]]) -> Tuple[torch.Tensor, Dict[int, int]]:
    images = [load_image_rgb(candidate["path"]) for candidate in candidates]
    option_id_to_pos = {
        int(candidate["option_id"]): pos for pos, candidate in enumerate(candidates)
    }
    return classifier.preprocess_images(images), option_id_to_pos


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = ManifestLensDataset(args.manifest)
    all_items = dataset.items
    items = filter_manifest_items(all_items, args.data_json)
    label_ids = get_subset_label_ids(all_items)
    option_id_to_name = build_option_name_map(items)
    classifier = OpenCLIP200Classifier(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
        class_index_json=args.class_index_json,
        label_ids=label_ids,
        device=device,
        prompt_template=args.prompt_template,
    )

    ae_hits: List[int] = []
    lens_hits: List[int] = []
    oracle_s_hits: List[int] = []
    random_mean_hits: List[float] = []
    option_id_to_hits: Dict[int, List[int]] = {}
    per_sample: List[Dict[str, Any]] = []

    for item in tqdm(items, desc="OpenCLIP baselines"):
        sample_id = str(item["id"])
        raw_label = int(item["label"])
        gt_idx = classifier.gt_subset_index(raw_label)
        candidates = item["candidates"]
        cand_tensor, _ = candidate_tensor(classifier, candidates)
        ae_path = resolve_ae_path(ROOT, sample_id, candidates[0]["path"])
        ae_tensor = classifier.preprocess_images([load_image_rgb(ae_path)])

        cand_sim = classifier.similarity_from_images(cand_tensor)
        ae_sim = classifier.similarity_from_images(ae_tensor)

        cand_pred_raw = classifier.pred_raw_labels(cand_sim)
        cand_correct = (cand_pred_raw == raw_label).to(torch.int64)
        ae_pred_raw = int(classifier.pred_raw_labels(ae_sim)[0].item())
        ae_hit = int(ae_pred_raw == raw_label)

        cand_probs = F.softmax(cand_sim, dim=-1)
        cand_conf = cand_probs.max(dim=-1).values
        lens_pos = int(torch.argmax(cand_conf).item())
        lens_option_id = int(candidates[lens_pos]["option_id"])
        lens_pred_raw = int(cand_pred_raw[lens_pos].item())
        lens_hit = int(lens_pred_raw == raw_label)

        correct_positions = (cand_correct == 1).nonzero(as_tuple=False).flatten().tolist()
        oracle_s_hit = int(len(correct_positions) > 0)
        gt_scores = cand_sim[:, gt_idx]
        if oracle_s_hit:
            correct_scores = gt_scores[correct_positions]
            oracle_s_pos = int(correct_positions[int(torch.argmax(correct_scores).item())])
        else:
            oracle_s_pos = int(torch.argmax(gt_scores).item())
        oracle_s_option_id = int(candidates[oracle_s_pos]["option_id"])

        sample_random_mean = float(cand_correct.float().mean().item())

        ae_hits.append(ae_hit)
        lens_hits.append(lens_hit)
        oracle_s_hits.append(oracle_s_hit)
        random_mean_hits.append(sample_random_mean)
        for pos, candidate in enumerate(candidates):
            option_id = int(candidate["option_id"])
            option_id_to_hits.setdefault(option_id, []).append(int(cand_correct[pos].item()))

        per_sample.append(
            {
                "sample_id": sample_id,
                "label": raw_label,
                "gt_subset_index": gt_idx,
                "ae_path": ae_path,
                "ae_pred": ae_pred_raw,
                "ae_correct": bool(ae_hit),
                "random_mean_correct": sample_random_mean,
                "lens_best_option_id": lens_option_id,
                "lens_best_position": lens_pos,
                "lens_pred": lens_pred_raw,
                "lens_correct": bool(lens_hit),
                "oracle_s_recoverable": bool(oracle_s_hit),
                "oracle_s_best_option_id": oracle_s_option_id,
                "oracle_s_best_position": oracle_s_pos,
                "num_correct_candidates": len(correct_positions),
                "correct_option_ids": [
                    int(candidates[pos]["option_id"]) for pos in correct_positions
                ],
            }
        )

    fixed_results = []
    for option_id, hits in sorted(option_id_to_hits.items()):
        summary = summarize_binary_hits(hits)
        fixed_results.append(
            {
                "option_id": option_id,
                "option_name": option_id_to_name.get(option_id, ""),
                **summary,
            }
        )
    best_fixed = max(fixed_results, key=lambda row: row["acc"])
    result = {
        "config": {
            "manifest": args.manifest,
            "data_json": args.data_json,
            "class_index_json": args.class_index_json,
            "openclip_model": args.openclip_model,
            "openclip_pretrained": args.openclip_pretrained,
            "prompt_template": args.prompt_template,
            "num_subset_classes": len(label_ids),
            "subset_label_ids": label_ids,
            "device": str(device),
            "evaluated_samples": len(items),
        },
        "summary": {
            "AE": summarize_binary_hits(ae_hits),
            "Random": summarize_float_hits(random_mean_hits),
            "Oracle-S": summarize_binary_hits(oracle_s_hits),
            "Oracle-F": best_fixed,
            "Lens": summarize_binary_hits(lens_hits),
        },
        "all_fixed_option_results": fixed_results,
        "records": per_sample,
    }
    save_json(args.output_json, result)
    print(result["summary"])
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
