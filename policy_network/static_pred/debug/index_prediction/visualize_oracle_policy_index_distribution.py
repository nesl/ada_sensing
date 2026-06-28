from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LABEL_DIR = ROOT / "data" / "ImageNet-ES-Diverse" / "oracle_policy_labels"
DEFAULT_OUTPUT_PNG = (
    ROOT / "policy_network" / "vis_results" / "oracle_policy_index_distribution.png"
)
DEFAULT_OUTPUT_JSON = (
    ROOT / "policy_network" / "results_debug" / "oracle_policy_index_distribution.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the ground-truth best_option_id distribution for the "
            "oracle policy train/val/test label JSON files."
        )
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=str(DEFAULT_LABEL_DIR),
        help="Directory containing oracle_policy_{train,val,test}_labels.json.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=str(DEFAULT_OUTPUT_PNG),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
    )
    parser.add_argument("--num_candidates", type=int, default=27)
    return parser.parse_args()


def load_split_counts(
    label_dir: Path,
    split: str,
    num_candidates: int,
) -> Tuple[List[int], Dict[int, str], int]:
    path = label_dir / f"oracle_policy_{split}_labels.json"
    with open(path, "r") as f:
        items = json.load(f)

    counts = [0 for _ in range(num_candidates)]
    option_names: Dict[int, str] = {}
    for item in items:
        option_id = int(item["best_option_id"])
        if option_id < 0 or option_id >= num_candidates:
            raise ValueError(
                f"Found best_option_id={option_id} outside [0, {num_candidates}) in {path}"
            )
        counts[option_id] += 1
        if "best_option_name" in item:
            option_names[option_id] = str(item["best_option_name"])

    return counts, option_names, len(items)


def collect_counts(
    label_dir: Path,
    num_candidates: int,
) -> Tuple[Dict[str, List[int]], Dict[int, str], Dict[str, int]]:
    split_to_counts: Dict[str, List[int]] = {}
    split_to_total: Dict[str, int] = {}
    option_names: Dict[int, str] = {}

    for split in ("train", "val", "test"):
        counts, split_option_names, total = load_split_counts(
            label_dir=label_dir,
            split=split,
            num_candidates=num_candidates,
        )
        split_to_counts[split] = counts
        split_to_total[split] = total
        option_names.update(split_option_names)

    return split_to_counts, option_names, split_to_total


def save_counts_json(
    output_json: Path,
    split_to_counts: Dict[str, List[int]],
    option_names: Dict[int, str],
    split_to_total: Dict[str, int],
) -> None:
    payload = {
        "splits": {},
        "option_names": {
            str(option_id): option_names.get(option_id, "")
            for option_id in range(len(next(iter(split_to_counts.values()))))
        },
    }
    for split, counts in split_to_counts.items():
        total = split_to_total[split]
        payload["splits"][split] = {
            "total": total,
            "counts": counts,
            "fractions": [count / max(1, total) for count in counts],
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)


def plot_distribution(
    output_png: Path,
    split_to_counts: Dict[str, List[int]],
    option_names: Dict[int, str],
    split_to_total: Dict[str, int],
) -> None:
    option_ids = np.arange(len(next(iter(split_to_counts.values()))))
    width = 0.26

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)

    ax.bar(option_ids - width, split_to_counts["train"], width=width, label="train")
    ax.bar(option_ids, split_to_counts["val"], width=width, label="val")
    ax.bar(option_ids + width, split_to_counts["test"], width=width, label="test")
    ax.set_title("Oracle Policy Ground-Truth Index Distribution")
    ax.set_xlabel("best_option_id")
    ax.set_ylabel("count")
    ax.set_xticks(option_ids)
    labels = [
        f"{option_id}\n{option_names.get(option_id, '')}"
        for option_id in option_ids
    ]
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir)
    output_png = Path(args.output_png)
    output_json = Path(args.output_json)

    split_to_counts, option_names, split_to_total = collect_counts(
        label_dir=label_dir,
        num_candidates=args.num_candidates,
    )
    save_counts_json(
        output_json=output_json,
        split_to_counts=split_to_counts,
        option_names=option_names,
        split_to_total=split_to_total,
    )
    plot_distribution(
        output_png=output_png,
        split_to_counts=split_to_counts,
        option_names=option_names,
        split_to_total=split_to_total,
    )

    print(f"Saved plot to {output_png}")
    print(f"Saved counts to {output_json}")


if __name__ == "__main__":
    main()
