from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
LENS_DIR = ROOT / "lens"

for extra_path in (ROOT, LENS_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AE / Random / Lens / Oracle-S / Oracle-F on a manifest split."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json",
        help=(
            "Split file used to filter sample_ids. Pass oracle_policy_test_labels.json "
            "if you want to match the oracle-policy split."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/acquisition_baselines_test.json",
    )
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


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


def build_option_name_map(items: List[Dict[str, Any]]) -> Dict[int, str]:
    option_id_to_name: Dict[int, str] = {}
    for item in items:
        for candidate in item["candidates"]:
            option_id = int(candidate["option_id"])
            option_name = str(candidate.get("meta", {}).get("option_name", ""))
            previous = option_id_to_name.get(option_id)
            if previous is not None and previous != option_name:
                raise ValueError(
                    f"option_id={option_id} maps to both {previous} and {option_name}"
                )
            option_id_to_name[option_id] = option_name
    return option_id_to_name


def resolve_ae_path(sample_id: str, candidate_path: str) -> str:
    parts = sample_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample_id={sample_id}")

    env, class_id, stem = parts[0], parts[1], "__".join(parts[2:])
    suffix = os.path.splitext(candidate_path)[1]
    path = (
        ROOT
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure"
        / env
        / "param_1"
        / class_id
        / f"{stem}{suffix}"
    )
    if not path.exists():
        raise FileNotFoundError(f"AE image not found for sample_id={sample_id}: {path}")
    return str(path)


def build_candidate_tensor(candidates: List[Dict[str, Any]], transform) -> torch.Tensor:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    return torch.stack(images, dim=0)


def summarize_binary_hits(hits: List[int]) -> Dict[str, Any]:
    correct = int(sum(hits))
    total = len(hits)
    return {
        "correct": correct,
        "total": total,
        "acc": 100.0 * correct / max(1, total),
    }


def summarize_float_scores(values: List[float]) -> Dict[str, Any]:
    total = len(values)
    mean_value = 100.0 * sum(values) / max(1, total)
    return {
        "mean": mean_value,
        "num_samples": total,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ManifestLensDataset(args.manifest)
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)
    allowed_sample_ids = load_allowed_sample_ids(args.data_json)
    items = filter_manifest_items(dataset.items, allowed_sample_ids)
    option_id_to_name = build_option_name_map(items)

    ae_hits: List[int] = []
    lens_hits: List[int] = []
    oracle_s_hits: List[int] = []
    random_mean_hits: List[float] = []
    option_id_to_hits: Dict[int, List[int]] = {}
    per_sample: List[Dict[str, Any]] = []

    for item in tqdm(items, desc="Eval acquisition baselines"):
        sample_id = str(item["id"])
        label = int(item["label"])
        candidates = item["candidates"]
        candidate_tensor = build_candidate_tensor(candidates, transform)
        ae_path = resolve_ae_path(sample_id, candidates[0]["path"])
        ae_tensor = transform(load_image_rgb(ae_path)).unsqueeze(0)

        with torch.no_grad():
            candidate_logits = classifier(candidate_tensor.to(device, non_blocking=True))
            ae_logits = classifier(ae_tensor.to(device, non_blocking=True))

        candidate_preds = torch.argmax(candidate_logits, dim=-1)
        candidate_correct = (candidate_preds == label).to(torch.int64)
        candidate_probs = F.softmax(candidate_logits, dim=-1)
        candidate_conf = candidate_probs.max(dim=-1).values

        ae_pred = int(torch.argmax(ae_logits[0]).item())
        ae_hit = int(ae_pred == label)

        lens_pos = int(torch.argmax(candidate_conf).item())
        lens_option_id = int(candidates[lens_pos]["option_id"])
        lens_pred = int(candidate_preds[lens_pos].item())
        lens_hit = int(lens_pred == label)

        correct_positions = (candidate_correct == 1).nonzero(as_tuple=False).flatten().tolist()
        oracle_s_hit = int(len(correct_positions) > 0)
        if oracle_s_hit:
            correct_conf = candidate_probs[correct_positions, label]
            best_correct_local = int(torch.argmax(correct_conf).item())
            oracle_s_pos = int(correct_positions[best_correct_local])
        else:
            target = torch.full(
                (candidate_logits.size(0),),
                fill_value=label,
                dtype=torch.long,
                device=candidate_logits.device,
            )
            losses = F.cross_entropy(candidate_logits, target, reduction="none")
            oracle_s_pos = int(torch.argmin(losses).item())
        oracle_s_option_id = int(candidates[oracle_s_pos]["option_id"])

        sample_random_mean = float(candidate_correct.float().mean().item())

        ae_hits.append(ae_hit)
        lens_hits.append(lens_hit)
        oracle_s_hits.append(oracle_s_hit)
        random_mean_hits.append(sample_random_mean)

        for pos, candidate in enumerate(candidates):
            option_id = int(candidate["option_id"])
            option_id_to_hits.setdefault(option_id, []).append(int(candidate_correct[pos].item()))

        per_sample.append(
            {
                "sample_id": sample_id,
                "label": label,
                "ae_path": ae_path,
                "ae_pred": ae_pred,
                "ae_correct": bool(ae_hit),
                "random_mean_correct": sample_random_mean,
                "lens_best_option_id": lens_option_id,
                "lens_best_position": lens_pos,
                "lens_pred": lens_pred,
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

    option_acc = []
    for option_id, hits in sorted(option_id_to_hits.items()):
        summary = summarize_binary_hits(hits)
        option_acc.append(
            {
                "option_id": option_id,
                "correct": summary["correct"],
                "total": summary["total"],
                "acc": summary["acc"],
            }
        )

    best_fixed_option = max(option_acc, key=lambda item: item["acc"])

    result = {
        "config": {
            "manifest": args.manifest,
            "data_json": args.data_json,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
            "evaluated_samples": len(items),
        },
        "summary": {
            "ae": summarize_binary_hits(ae_hits),
            "random": summarize_float_scores(random_mean_hits),
            "lens": summarize_binary_hits(lens_hits),
            "oracle_specific": summarize_binary_hits(oracle_s_hits),
            "oracle_fixed": {
                **best_fixed_option,
                "best_option_name": option_id_to_name.get(
                    int(best_fixed_option["option_id"]), ""
                ),
            },
        },
        "all_fixed_option_results": option_acc,
        "records": per_sample,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved acquisition baseline evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
