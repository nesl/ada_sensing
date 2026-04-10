"""
Use the Lens baseline to generate expert labels for each sample, then split
the resulting records into train / val / test json files.

Each record contains:
- sample_id / group_id / env / task label
- baseline image path (input to policy network)
- best option selected by lens_select_best (supervision target)

Splitting is done at the reference-image group level:
- all lighting conditions of the same reference image stay in one split
- within each ImageNet synset, we sample a fixed number of reference images
  for train / val / test
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

    p.add_argument("--train_groups_per_class", type=int, default=3)
    p.add_argument("--val_groups_per_class", type=int, default=1)
    p.add_argument("--test_groups_per_class", type=int, default=1)
    p.add_argument("--expected_num_classes", type=int, default=200)
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
    path = os.path.join(
        "data",
        "ImageNet-ES-Diverse",
        "es-diverse-test",
        "auto_exposure",
        env,
        "param_1",
        class_id,
        f"{stem}{suffix}",
    )
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Policy input image not found for sample_id={sample_id}: {abs_path}"
        )
    return abs_path


def split_by_class_group_counts(
    records: List[Dict[str, Any]],
    train_groups_per_class: int,
    val_groups_per_class: int,
    test_groups_per_class: int,
    expected_num_classes: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        gid = r["group_id"]
        group_to_records.setdefault(gid, []).append(r)

    class_to_groups: Dict[str, List[str]] = {}
    for gid in group_to_records:
        class_id = parse_class_id_from_group_id(gid)
        class_to_groups.setdefault(class_id, []).append(gid)

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
    train_groups, val_groups, test_groups = set(), set(), set()

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

        policy_input_path = resolve_policy_input_path(
            sample_id=sample_id,
            candidate_path=candidates[0]["path"],
        )

        imgs = [tfm(load_image_rgb(c["path"])) for c in candidates]
        imgs = torch.stack(imgs, dim=0)  # [27, 3, H, W]

        with torch.no_grad():
            best_idx, _, best_logits = lens_select_best(model, imgs, device)

        best_candidate = candidates[best_idx]
        pred = int(torch.argmax(best_logits).item())

        record = {
            "sample_id": sample_id,
            "group_id": group_id,
            "class_id": parse_class_id_from_group_id(group_id),
            "env": env,
            "label": label,
            "pred": pred,
            "baseline_path": policy_input_path,

            "best_idx_in_candidates": int(best_idx),
            "best_option_id": int(best_candidate["option_id"]),
            "best_option_name": best_candidate["meta"]["option_name"],
            "best_path": best_candidate["path"],
        }

        expert_records.append(record)

    # save full records first
    full_path = os.path.join(args.output_dir, "policy_all_labels.json")
    save_json(full_path, expert_records)

    # Split by reference-image group so all lighting conditions of the same
    # image stay together. Within each class, use a fixed 3/1/1 group split.
    train_records, val_records, test_records = split_by_class_group_counts(
        expert_records,
        train_groups_per_class=args.train_groups_per_class,
        val_groups_per_class=args.val_groups_per_class,
        test_groups_per_class=args.test_groups_per_class,
        expected_num_classes=args.expected_num_classes,
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
    train_classes = len(set(r["class_id"] for r in train_records))
    val_classes = len(set(r["class_id"] for r in val_records))
    test_classes = len(set(r["class_id"] for r in test_records))

    print(
        f"Group counts -> full: {full_groups}, "
        f"train: {train_groups}, val: {val_groups}, test: {test_groups}"
    )
    print(
        f"Class counts -> train: {train_classes}, "
        f"val: {val_classes}, test: {test_classes}"
    )


if __name__ == "__main__":
    main()
