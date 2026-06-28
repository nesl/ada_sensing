from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]
LENS_DIR = ROOT / "lens"

for extra_path in (ROOT, LENS_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model


DEFAULT_DOWNSTREAM_JSON = (
    ROOT
    / "policy_network"
    / "results_random_noise"
    / "G_oracle_full_soft"
    / "downstream_test_best.json"
)
DEFAULT_MANIFEST = ROOT / "data" / "ImageNet-ES-Diverse" / "manifest_all.json"
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "policy_network"
    / "results_random_noise"
    / "G_oracle_full_soft"
    / "counterfactual_non24_force24.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For downstream records whose policy top-1 selected option is not the "
            "forced option_id, force that option_id and evaluate downstream correctness."
        )
    )
    parser.add_argument("--downstream_json", type=Path, default=DEFAULT_DOWNSTREAM_JSON)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force_option_id", type=int, default=24)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def build_manifest_index(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    dataset = ManifestLensDataset(str(manifest_path))
    index: Dict[str, Dict[str, Any]] = {}
    for item in dataset.items:
        sample_id = item.get("id")
        if sample_id is None:
            raise KeyError("Manifest item is missing required key 'id'.")
        index[str(sample_id)] = item
    return index


def get_candidate_by_option_id(
    manifest_item: Dict[str, Any],
    option_id: int,
) -> Dict[str, Any]:
    for candidate in manifest_item["candidates"]:
        if int(candidate["option_id"]) == option_id:
            return candidate
    sample_id = manifest_item.get("id", "<unknown>")
    raise ValueError(f"option_id={option_id} not found for sample_id={sample_id}")


def filter_records(
    downstream_payload: Dict[str, Any],
    force_option_id: int,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for record in downstream_payload.get("records", []):
        topk = record.get("policy_topk") or []
        if not topk:
            continue
        top1 = topk[0]
        if int(top1["option_id"]) != force_option_id:
            filtered.append(record)
    return filtered


def summarize_original_force_option_records(
    downstream_payload: Dict[str, Any],
    force_option_id: int,
) -> Dict[str, int]:
    total_records = 0
    original_correct = 0
    already_forced_option_total = 0
    already_forced_option_correct = 0

    for record in downstream_payload.get("records", []):
        topk = record.get("policy_topk") or []
        if not topk:
            continue
        top1 = topk[0]
        is_correct = bool(top1.get("downstream_correct", False))

        total_records += 1
        original_correct += int(is_correct)
        if int(top1["option_id"]) == force_option_id:
            already_forced_option_total += 1
            already_forced_option_correct += int(is_correct)

    return {
        "total_records": total_records,
        "original_correct": original_correct,
        "already_forced_option_total": already_forced_option_total,
        "already_forced_option_correct": already_forced_option_correct,
    }


class ForcedOptionDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        manifest_index: Dict[str, Dict[str, Any]],
        force_option_id: int,
        transform,
    ) -> None:
        self.records = records
        self.manifest_index = manifest_index
        self.force_option_id = force_option_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample_id = str(record["sample_id"])
        manifest_item = self.manifest_index.get(sample_id)
        if manifest_item is None:
            raise KeyError(f"sample_id={sample_id} from downstream json is missing in manifest.")

        forced_candidate = get_candidate_by_option_id(manifest_item, self.force_option_id)
        image = self.transform(load_image_rgb(forced_candidate["path"]))
        top1 = record["policy_topk"][0]

        return {
            "sample_id": sample_id,
            "image": image,
            "label": int(record["label"]),
            "original_selected_option_id": int(top1["option_id"]),
            "original_policy_confidence": float(top1["policy_confidence"]),
            "original_downstream_pred_label": int(top1["downstream_pred_label"]),
            "original_downstream_pred_confidence": float(top1["downstream_pred_confidence"]),
            "original_downstream_correct": bool(top1["downstream_correct"]),
            "forced_option_path": str(forced_candidate["path"]),
        }


def accuracy_payload(correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "acc": 100.0 * correct / max(1, total),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_json.parent, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    downstream_payload = load_json(args.downstream_json)
    manifest_index = build_manifest_index(args.manifest)
    filtered_records = filter_records(downstream_payload, args.force_option_id)
    original_summary = summarize_original_force_option_records(
        downstream_payload,
        args.force_option_id,
    )

    transform = imagenet_preprocess(args.image_size)
    dataset = ForcedOptionDataset(
        filtered_records,
        manifest_index,
        args.force_option_id,
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    classifier = load_timm_model(args.model, device=device)

    per_sample: List[Dict[str, Any]] = []
    total = 0
    original_correct = 0
    forced_correct = 0
    original_wrong_forced_correct = 0
    original_correct_forced_wrong = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Force option_id={args.force_option_id}"):
            images = batch["image"].to(device, non_blocking=True)
            logits = classifier(images)
            probs = torch.softmax(logits, dim=-1)
            pred_confs, pred_labels = probs.max(dim=-1)

            for i, sample_id in enumerate(batch["sample_id"]):
                label = int(batch["label"][i].item())
                original_is_correct = bool(batch["original_downstream_correct"][i].item())
                forced_pred_label = int(pred_labels[i].item())
                forced_pred_conf = float(pred_confs[i].item())
                forced_is_correct = forced_pred_label == label

                total += 1
                original_correct += int(original_is_correct)
                forced_correct += int(forced_is_correct)
                original_wrong_forced_correct += int(
                    (not original_is_correct) and forced_is_correct
                )
                original_correct_forced_wrong += int(
                    original_is_correct and (not forced_is_correct)
                )

                per_sample.append(
                    {
                        "sample_id": str(sample_id),
                        "label": label,
                        "original_selected_option_id": int(
                            batch["original_selected_option_id"][i].item()
                        ),
                        "original_policy_confidence": float(
                            batch["original_policy_confidence"][i].item()
                        ),
                        "original_downstream_pred_label": int(
                            batch["original_downstream_pred_label"][i].item()
                        ),
                        "original_downstream_pred_confidence": float(
                            batch["original_downstream_pred_confidence"][i].item()
                        ),
                        "original_downstream_correct": original_is_correct,
                        "forced_option_id": args.force_option_id,
                        "forced_option_path": batch["forced_option_path"][i],
                        "forced_downstream_pred_label": forced_pred_label,
                        "forced_downstream_pred_confidence": forced_pred_conf,
                        "forced_downstream_correct": forced_is_correct,
                    }
                )

    all_force_correct = original_summary["already_forced_option_correct"] + forced_correct
    all_force_total = original_summary["already_forced_option_total"] + total

    result = {
        "config": {
            "downstream_json": str(args.downstream_json),
            "manifest": str(args.manifest),
            "filter": f"original_selected_option_id != {args.force_option_id}",
            "force_option_id": args.force_option_id,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
        },
        "summary": {
            "all_original_correct": accuracy_payload(
                original_summary["original_correct"],
                original_summary["total_records"],
            ),
            "all_force_correct": accuracy_payload(all_force_correct, all_force_total),
            "filtered_samples": total,
            "filtered_original_correct": accuracy_payload(original_correct, total),
            "forced_correct": accuracy_payload(forced_correct, total),
            "original_wrong_forced_correct": accuracy_payload(
                original_wrong_forced_correct,
                total,
            ),
            "original_correct_forced_wrong": accuracy_payload(
                original_correct_forced_wrong,
                total,
            ),
        },
        "records": per_sample,
    }

    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved counterfactual result to {args.output_json}")


if __name__ == "__main__":
    main()
