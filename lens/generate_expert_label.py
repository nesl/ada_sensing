"""
Use the Lens baseline to generate expert labels for each sample, then split
the resulting records into train / val / test json files.

Each record contains:
- sample_id / group_id / env / task label
- baseline image info (input to policy network)
- best option selected by lens_select_best (supervision target)
"""

from __future__ import annotations
import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import *
from lens_core import lens_select_best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--baseline_option_id", type=int, default=13)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def parse_group_id(sample_id: str) -> str:
    """
    Example:
        l1__n01443537__ILSVRC2012_val_00000994
    ->  n01443537__ILSVRC2012_val_00000994

    We remove env so that the same reference image under different lighting
    conditions stays in the same split.
    """
    parts = sample_id.split("__")
    if len(parts) >= 3:
        return "__".join(parts[1:])
    return sample_id


def check_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {s}"
        )


def split_by_group(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    check_split_ratios(train_ratio, val_ratio, test_ratio)

    group_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        gid = r["group_id"]
        group_to_records.setdefault(gid, []).append(r)

    all_groups = list(group_to_records.keys())
    rng = random.Random(seed)
    rng.shuffle(all_groups)

    n_groups = len(all_groups)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    n_test = n_groups - n_train - n_val

    train_groups = set(all_groups[:n_train])
    val_groups = set(all_groups[n_train:n_train + n_val])
    test_groups = set(all_groups[n_train + n_val:])

    train_records, val_records, test_records = [], [], []

    for gid, group_records in group_to_records.items():
        if gid in train_groups:
            train_records.extend(group_records)
        elif gid in val_groups:
            val_records.extend(group_records)
        elif gid in test_groups:
            test_records.extend(group_records)
        else:
            raise RuntimeError(f"Group {gid} was not assigned to any split.")

    return train_records, val_records, test_records


def save_json(path: str, data: List[Dict[str, Any]]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = ManifestLensDataset(args.manifest)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],
    )

    model = load_timm_model(args.model, device=device)
    model.eval()
    tfm = imagenet_preprocess(args.image_size)

    expert_records: List[Dict[str, Any]] = []

    for sample in tqdm(dl, desc="Generating expert labels"):
        sample_id = sample["id"]
        group_id = parse_group_id(sample_id)
        env = sample["env"]
        label = int(sample["label"])
        candidates = sample["candidates"]  # expected len = 27

        baseline_candidates = [
            c for c in candidates if int(c["option_id"]) == int(args.baseline_option_id)
        ]
        if len(baseline_candidates) != 1:
            raise ValueError(
                f"Sample {sample_id} has {len(baseline_candidates)} baseline matches "
                f"for option_id={args.baseline_option_id}"
            )
        baseline_candidate = baseline_candidates[0]

        imgs = [tfm(load_image_rgb(c["path"])) for c in candidates]
        imgs = torch.stack(imgs, dim=0)  # [27, 3, H, W]

        with torch.no_grad():
            best_idx, _, best_logits = lens_select_best(model, imgs, device)

        best_candidate = candidates[best_idx]
        pred = int(torch.argmax(best_logits).item())

        record = {
            "sample_id": sample_id,
            "group_id": group_id,
            "env": env,
            "label": label,
            "pred": pred,

            "baseline_option_id": int(args.baseline_option_id),
            "baseline_option_name": baseline_candidate["meta"]["option_name"],
            "baseline_path": baseline_candidate["path"],

            "best_idx_in_candidates": int(best_idx),
            "best_option_id": int(best_candidate["option_id"]),
            "best_option_name": best_candidate["meta"]["option_name"],
            "best_path": best_candidate["path"],
        }

        expert_records.append(record)

    # save full records first
    full_path = os.path.join(args.output_dir, "policy_all_labels.json")
    save_json(full_path, expert_records)

    # split by reference-image group to avoid leakage across envs
    train_records, val_records, test_records = split_by_group(
        expert_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_path = os.path.join(args.output_dir, "policy_train_labels.json")
    val_path = os.path.join(args.output_dir, "policy_val_labels.json")
    test_path = os.path.join(args.output_dir, "policy_test_labels.json")

    save_json(train_path, train_records)
    save_json(val_path, val_records)
    save_json(test_path, test_records)

    print(f"Saved full records to  : {full_path}")
    print(f"Saved train split to   : {train_path} ({len(train_records)} samples)")
    print(f"Saved val split to     : {val_path} ({len(val_records)} samples)")
    print(f"Saved test split to    : {test_path} ({len(test_records)} samples)")

    # also print group-level stats
    full_groups = len(set(r["group_id"] for r in expert_records))
    train_groups = len(set(r["group_id"] for r in train_records))
    val_groups = len(set(r["group_id"] for r in val_records))
    test_groups = len(set(r["group_id"] for r in test_records))

    print(
        f"Group counts -> full: {full_groups}, "
        f"train: {train_groups}, val: {val_groups}, test: {test_groups}"
    )


if __name__ == "__main__":
    main()