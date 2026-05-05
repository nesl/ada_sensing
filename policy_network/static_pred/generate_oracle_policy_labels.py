from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
LENS_DIR = ROOT / "lens"

for extra_path in (ROOT, LENS_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Oracle GT labels for the sensor policy task. For each sample, "
            "choose the candidate that best helps the downstream classifier solve "
            "the ground-truth label."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/oracle_policy_labels",
    )
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_candidates", type=int, default=27)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--split_source_dir",
        type=str,
        default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/policy_labels",
        help=(
            "Optional directory containing existing policy_{train,val,test}_labels.json "
            "to reuse the exact same split assignment."
        ),
    )
    parser.add_argument("--train_groups_per_class", type=int, default=3)
    parser.add_argument("--val_groups_per_class", type=int, default=1)
    parser.add_argument("--test_groups_per_class", type=int, default=1)
    parser.add_argument("--expected_num_classes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--soft_label_mode",
        type=str,
        choices=["uniform_correct", "confidence_correct"],
        default="confidence_correct",
        help="How to distribute probability mass across oracle-correct candidates.",
    )
    parser.add_argument(
        "--all_wrong_soft_target_mode",
        type=str,
        choices=["fallback_onehot", "uniform"],
        default="fallback_onehot",
        help=(
            "How to build soft_target when no candidate is downstream-correct. "
            "The default preserves the original fallback one-hot behavior."
        ),
    )
    parser.add_argument(
        "--all_wrong_sample_weight",
        type=float,
        default=1.0,
        help=(
            "Training sample weight to write for all-wrong samples. "
            "Use 0.1 with --all_wrong_soft_target_mode uniform for v2 labels."
        ),
    )
    return parser.parse_args()


def parse_group_id(sample_id: str) -> str:
    parts = sample_id.split("__")
    if len(parts) >= 3:
        return "__".join(parts[1:])
    return sample_id


def parse_class_id_from_group_id(group_id: str) -> str:
    parts = group_id.split("__")
    if len(parts) >= 2:
        return parts[0]
    raise ValueError(f"Cannot parse class id from group_id={group_id}")


def resolve_policy_input_path(sample_id: str, candidate_path: str) -> str:
    parts = sample_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample_id={sample_id}")

    env, class_id, stem = parts[0], parts[1], "__".join(parts[2:])
    suffix = os.path.splitext(candidate_path)[1]
    path = ROOT / "data" / "ImageNet-ES-Diverse" / "es-diverse-test" / "auto_exposure" / env / "param_1" / class_id / f"{stem}{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"Policy input image not found for sample_id={sample_id}: {path}"
        )
    return str(path)


def save_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def split_by_class_group_counts(
    records: List[Dict[str, Any]],
    train_groups_per_class: int,
    val_groups_per_class: int,
    test_groups_per_class: int,
    expected_num_classes: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        group_to_records.setdefault(record["group_id"], []).append(record)

    class_to_groups: Dict[str, List[str]] = {}
    for group_id in group_to_records:
        class_id = parse_class_id_from_group_id(group_id)
        class_to_groups.setdefault(class_id, []).append(group_id)

    all_classes = sorted(class_to_groups.keys())
    if len(all_classes) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, found {len(all_classes)}: "
            f"{all_classes[:10]}{' ...' if len(all_classes) > 10 else ''}"
        )

    groups_needed_per_class = (
        train_groups_per_class + val_groups_per_class + test_groups_per_class
    )
    rng = random.Random(seed)
    train_groups = set()
    val_groups = set()
    test_groups = set()

    for class_id in all_classes:
        groups = sorted(class_to_groups[class_id])
        if len(groups) != groups_needed_per_class:
            raise ValueError(
                f"Class {class_id} has {len(groups)} reference-image groups, "
                f"but split requires exactly {groups_needed_per_class} "
                f"({train_groups_per_class} train + {val_groups_per_class} val + "
                f"{test_groups_per_class} test)."
            )

        rng.shuffle(groups)
        n_train = train_groups_per_class
        n_val = val_groups_per_class

        train_groups.update(groups[:n_train])
        val_groups.update(groups[n_train:n_train + n_val])
        test_groups.update(groups[n_train + n_val:])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for group_id, group_records in group_to_records.items():
        if group_id in train_groups:
            train_records.extend(group_records)
        elif group_id in val_groups:
            val_records.extend(group_records)
        elif group_id in test_groups:
            test_records.extend(group_records)
        else:
            raise RuntimeError(f"Group {group_id} was not assigned to any split.")

    return train_records, val_records, test_records


def load_split_groups(split_source_dir: str) -> Dict[str, str]:
    split_to_filename = {
        "train": "policy_train_labels.json",
        "val": "policy_val_labels.json",
        "test": "policy_test_labels.json",
    }
    group_to_split: Dict[str, str] = {}

    for split_name, filename in split_to_filename.items():
        path = os.path.join(split_source_dir, filename)
        with open(path, "r") as f:
            records: List[Dict[str, Any]] = json.load(f)

        for record in records:
            group_id = record["group_id"]
            prev = group_to_split.get(group_id)
            if prev is not None and prev != split_name:
                raise ValueError(
                    f"Group {group_id} appears in both {prev} and {split_name} splits."
                )
            group_to_split[group_id] = split_name

    return group_to_split


def split_with_existing_groups(
    records: List[Dict[str, Any]],
    group_to_split: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for record in records:
        split_name = group_to_split.get(record["group_id"])
        if split_name is None:
            raise KeyError(
                f"Group {record['group_id']} missing from split_source_dir split mapping."
            )
        if split_name == "train":
            train_records.append(record)
        elif split_name == "val":
            val_records.append(record)
        elif split_name == "test":
            test_records.append(record)
        else:
            raise ValueError(f"Unexpected split name: {split_name}")

    return train_records, val_records, test_records


def build_candidate_tensor(candidates: List[Dict[str, Any]], transform) -> torch.Tensor:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    return torch.stack(images, dim=0)


def choose_oracle_candidate(
    logits: torch.Tensor,
    label: int,
) -> Tuple[int, int, bool, List[int]]:
    preds = torch.argmax(logits, dim=-1)
    correct_positions = (preds == label).nonzero(as_tuple=False).flatten()

    if correct_positions.numel() > 0:
        correct_logits = logits[correct_positions]
        correct_conf = torch.softmax(correct_logits, dim=-1)[:, label]
        best_local_idx = int(torch.argmax(correct_conf).item())
        best_idx = int(correct_positions[best_local_idx].item())
        return best_idx, int(preds[best_idx].item()), True, correct_positions.tolist()

    target = torch.full(
        (logits.size(0),),
        fill_value=label,
        dtype=torch.long,
        device=logits.device,
    )
    losses = F.cross_entropy(logits, target, reduction="none")
    best_idx = int(torch.argmin(losses).item())
    return best_idx, int(preds[best_idx].item()), False, []


def build_soft_target(
    logits: torch.Tensor,
    candidates: List[Dict[str, Any]],
    label: int,
    num_candidates: int,
    soft_label_mode: str,
    all_wrong_soft_target_mode: str,
    best_idx: int,
    correct_candidate_positions: List[int],
) -> Tuple[List[float], List[int], List[float]]:
    soft_target = torch.zeros(num_candidates, dtype=torch.float32)

    if correct_candidate_positions:
        correct_positions_tensor = torch.tensor(correct_candidate_positions, dtype=torch.long)
        correct_logits = logits[correct_positions_tensor]
        if soft_label_mode == "uniform_correct":
            weights = torch.ones(len(correct_candidate_positions), dtype=torch.float32)
        elif soft_label_mode == "confidence_correct":
            weights = torch.softmax(correct_logits, dim=-1)[:, label].detach().cpu()
        else:
            raise ValueError(f"Unsupported soft_label_mode: {soft_label_mode}")

        weights = weights / weights.sum().clamp_min(1e-12)
        correct_option_ids = [int(candidates[pos]["option_id"]) for pos in correct_candidate_positions]
        for option_id, weight in zip(correct_option_ids, weights.tolist()):
            soft_target[option_id] = float(weight)
        return soft_target.tolist(), correct_option_ids, weights.tolist()

    if all_wrong_soft_target_mode == "uniform":
        soft_target.fill_(1.0 / float(num_candidates))
        return soft_target.tolist(), [], []
    if all_wrong_soft_target_mode != "fallback_onehot":
        raise ValueError(f"Unsupported all_wrong_soft_target_mode: {all_wrong_soft_target_mode}")

    fallback_option_id = int(candidates[best_idx]["option_id"])
    soft_target[fallback_option_id] = 1.0
    return soft_target.tolist(), [fallback_option_id], [1.0]


def build_record(
    sample: Dict[str, Any],
    policy_input_path: str,
    best_candidate: Dict[str, Any],
    best_idx: int,
    pred: int,
    had_correct_candidate: bool,
    correct_candidate_positions: List[int],
    correct_option_ids: List[int],
    correct_option_weights: List[float],
    soft_target: List[float],
    sample_weight: float,
) -> Dict[str, Any]:
    sample_id = sample["id"]
    group_id = parse_group_id(sample_id)
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "class_id": parse_class_id_from_group_id(group_id),
        "env": sample["env"],
        "label": int(sample["label"]),
        "pred": pred,
        "oracle_had_correct_candidate": had_correct_candidate,
        "oracle_num_correct_candidates": len(correct_candidate_positions),
        "oracle_correct_candidate_positions": [int(pos) for pos in correct_candidate_positions],
        "oracle_correct_option_ids": [int(option_id) for option_id in correct_option_ids],
        "oracle_correct_option_weights": [float(weight) for weight in correct_option_weights],
        "soft_target": [float(value) for value in soft_target],
        "sample_weight": float(sample_weight),
        "baseline_path": policy_input_path,

        "best_idx_in_candidates": int(best_idx),
        "best_option_id": int(best_candidate["option_id"]),
        "best_option_name": best_candidate["meta"]["option_name"],
        "best_path": best_candidate["path"],
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    oracle_recoverable = sum(int(record["oracle_had_correct_candidate"]) for record in records)
    avg_num_correct = sum(int(record["oracle_num_correct_candidates"]) for record in records) / max(1, total)
    return {
        "total": total,
        "oracle_recoverable": oracle_recoverable,
        "oracle_upper_bound_acc": 100.0 * oracle_recoverable / max(1, total),
        "avg_num_correct_candidates": avg_num_correct,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ManifestLensDataset(args.manifest)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],
    )

    model = load_timm_model(args.model, device=device)
    model.eval()
    transform = imagenet_preprocess(args.image_size)

    oracle_records: List[Dict[str, Any]] = []

    for sample in tqdm(dataloader, desc="Generating oracle policy labels"):
        sample_id = sample["id"]
        label = int(sample["label"])
        candidates = sample["candidates"]

        policy_input_path = resolve_policy_input_path(
            sample_id=sample_id,
            candidate_path=candidates[0]["path"],
        )
        candidate_tensor = build_candidate_tensor(candidates, transform)

        with torch.no_grad():
            logits = model(candidate_tensor.to(device, non_blocking=True))

        # logic of generating hrad lables: if one or more candidates are correct, chosse the one with highest confidence
        # if no candidates are correct, choose the one with lowest loss on gt class
        best_idx, pred, had_correct_candidate, correct_candidate_positions = choose_oracle_candidate(
            logits=logits,
            label=label,
        )
        best_candidate = candidates[best_idx]

        soft_target, correct_option_ids, correct_option_weights = build_soft_target(
            logits=logits.detach().cpu(),
            candidates=candidates,
            label=label,
            num_candidates=args.num_candidates,
            soft_label_mode=args.soft_label_mode,
            all_wrong_soft_target_mode=args.all_wrong_soft_target_mode,
            best_idx=best_idx,
            correct_candidate_positions=correct_candidate_positions,
        )
        sample_weight = 1.0 if had_correct_candidate else float(args.all_wrong_sample_weight)

        oracle_records.append(
            build_record(
                sample=sample,
                policy_input_path=policy_input_path,
                best_candidate=best_candidate,
                best_idx=best_idx,
                pred=pred,
                had_correct_candidate=had_correct_candidate,
                correct_candidate_positions=correct_candidate_positions,
                correct_option_ids=correct_option_ids,
                correct_option_weights=correct_option_weights,
                soft_target=soft_target,
                sample_weight=sample_weight,
            )
        )

    all_labels_path = os.path.join(args.output_dir, "oracle_policy_all_labels.json")
    save_json(all_labels_path, oracle_records)

    if args.split_source_dir is not None:
        group_to_split = load_split_groups(args.split_source_dir)
        train_records, val_records, test_records = split_with_existing_groups(
            oracle_records,
            group_to_split,
        )
    else:
        train_records, val_records, test_records = split_by_class_group_counts(
            oracle_records,
            train_groups_per_class=args.train_groups_per_class,
            val_groups_per_class=args.val_groups_per_class,
            test_groups_per_class=args.test_groups_per_class,
            expected_num_classes=args.expected_num_classes,
            seed=args.seed,
        )

    train_path = os.path.join(args.output_dir, "oracle_policy_train_labels.json")
    val_path = os.path.join(args.output_dir, "oracle_policy_val_labels.json")
    test_path = os.path.join(args.output_dir, "oracle_policy_test_labels.json")
    summary_path = os.path.join(args.output_dir, "oracle_policy_summary.json")

    save_json(train_path, train_records)
    save_json(val_path, val_records)
    save_json(test_path, test_records)
    save_json(
        summary_path,
        {
            "manifest": args.manifest,
            "model": args.model,
            "num_candidates": args.num_candidates,
            "soft_label_mode": args.soft_label_mode,
            "split_source_dir": args.split_source_dir,
            "all": summarize_records(oracle_records),
            "train": summarize_records(train_records),
            "val": summarize_records(val_records),
            "test": summarize_records(test_records),
        },
    )

    print(f"Saved full oracle labels to : {all_labels_path}")
    print(f"Saved train split to        : {train_path} ({len(train_records)} samples)")
    print(f"Saved val split to          : {val_path} ({len(val_records)} samples)")
    print(f"Saved test split to         : {test_path} ({len(test_records)} samples)")
    print(f"Saved summary to            : {summary_path}")


if __name__ == "__main__":
    main()
