from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABEL_DIR = ROOT / "data" / "ImageNet-ES-Diverse" / "oracle_policy_labels"
DEFAULT_OUTPUT_PNG = (
    ROOT / "policy_network" / "vis_results" / "oracle_policy_top5_index_distribution.png"
)
DEFAULT_OUTPUT_JSON = (
    ROOT / "policy_network" / "results_debug" / "oracle_policy_top5_index_distribution.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize train/val/test distributions of oracle top-k option indices. "
            "Top-k is computed from each sample's soft_target positive weights."
        )
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=str(DEFAULT_LABEL_DIR),
        help="Directory containing oracle_policy_{train,val,test}_labels.json.",
    )
    parser.add_argument("--output_png", type=str, default=str(DEFAULT_OUTPUT_PNG))
    parser.add_argument("--output_json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--num_candidates", type=int, default=27)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--include_fallback_targets",
        action="store_true",
        help=(
            "Include the one-hot fallback soft target used when no candidate is "
            "downstream-correct. By default, only truly correct candidates are counted."
        ),
    )
    return parser.parse_args()


def get_topk_option_ids(
    item: dict,
    num_candidates: int,
    topk: int,
    include_fallback_targets: bool,
) -> List[int]:
    if not include_fallback_targets and int(item.get("oracle_num_correct_candidates", 0)) == 0:
        return []

    if "soft_target" in item:
        soft_target = [float(x) for x in item["soft_target"]]
        if len(soft_target) != num_candidates:
            raise ValueError(
                f"Expected soft_target length {num_candidates}, got {len(soft_target)} "
                f"for sample_id={item.get('sample_id')}"
            )
        positive = [
            (option_id, weight)
            for option_id, weight in enumerate(soft_target)
            if weight > 0.0
        ]
    else:
        option_ids = item.get("oracle_correct_option_ids", [])
        weights = item.get("oracle_correct_option_weights", [])
        positive = [(int(option_id), float(weight)) for option_id, weight in zip(option_ids, weights)]

    positive.sort(key=lambda pair: (-pair[1], pair[0]))
    return [option_id for option_id, _ in positive[:topk]]


def load_split_counts(
    label_dir: Path,
    split: str,
    num_candidates: int,
    topk: int,
    include_fallback_targets: bool,
) -> Tuple[List[int], Dict[int, str], int, int]:
    path = label_dir / f"oracle_policy_{split}_labels.json"
    with open(path, "r") as f:
        items = json.load(f)

    counts = [0 for _ in range(num_candidates)]
    option_names: Dict[int, str] = {}
    total_topk_entries = 0

    for item in items:
        topk_option_ids = get_topk_option_ids(
            item,
            num_candidates,
            topk,
            include_fallback_targets,
        )
        total_topk_entries += len(topk_option_ids)
        for option_id in topk_option_ids:
            if option_id < 0 or option_id >= num_candidates:
                raise ValueError(
                    f"Found option_id={option_id} outside [0, {num_candidates}) in {path}"
                )
            counts[option_id] += 1

        if "best_option_id" in item and "best_option_name" in item:
            option_names[int(item["best_option_id"])] = str(item["best_option_name"])

    return counts, option_names, len(items), total_topk_entries


def collect_counts(
    label_dir: Path,
    num_candidates: int,
    topk: int,
    include_fallback_targets: bool,
) -> Tuple[Dict[str, List[int]], Dict[int, str], Dict[str, int], Dict[str, int]]:
    split_to_counts: Dict[str, List[int]] = {}
    split_to_total: Dict[str, int] = {}
    split_to_topk_total: Dict[str, int] = {}
    option_names: Dict[int, str] = {}

    for split in ("train", "val", "test"):
        counts, split_option_names, total, total_topk_entries = load_split_counts(
            label_dir=label_dir,
            split=split,
            num_candidates=num_candidates,
            topk=topk,
            include_fallback_targets=include_fallback_targets,
        )
        split_to_counts[split] = counts
        split_to_total[split] = total
        split_to_topk_total[split] = total_topk_entries
        option_names.update(split_option_names)

    return split_to_counts, option_names, split_to_total, split_to_topk_total


def save_counts_json(
    output_json: Path,
    split_to_counts: Dict[str, List[int]],
    option_names: Dict[int, str],
    split_to_total: Dict[str, int],
    split_to_topk_total: Dict[str, int],
    topk: int,
    include_fallback_targets: bool,
) -> None:
    payload = {
        "topk": topk,
        "include_fallback_targets": include_fallback_targets,
        "splits": {},
        "option_names": {
            str(option_id): option_names.get(option_id, "")
            for option_id in range(len(next(iter(split_to_counts.values()))))
        },
    }
    for split, counts in split_to_counts.items():
        total_samples = split_to_total[split]
        total_topk_entries = split_to_topk_total[split]
        payload["splits"][split] = {
            "total_samples": total_samples,
            "total_topk_entries": total_topk_entries,
            "counts": counts,
            "fraction_of_samples": [count / max(1, total_samples) for count in counts],
            "fraction_of_topk_entries": [
                count / max(1, total_topk_entries) for count in counts
            ],
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)


def plot_distribution(
    output_png: Path,
    split_to_counts: Dict[str, List[int]],
    option_names: Dict[int, str],
    topk: int,
    include_fallback_targets: bool,
) -> None:
    option_ids = np.arange(len(next(iter(split_to_counts.values()))))
    width = 0.26

    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)

    ax.bar(option_ids - width, split_to_counts["train"], width=width, label="train")
    ax.bar(option_ids, split_to_counts["val"], width=width, label="val")
    ax.bar(option_ids + width, split_to_counts["test"], width=width, label="test")
    suffix = "including fallback" if include_fallback_targets else "correct candidates only"
    ax.set_title(f"Oracle Policy Top-{topk} Index Distribution ({suffix})")
    ax.set_xlabel("option_id in oracle top-k")
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

    split_to_counts, option_names, split_to_total, split_to_topk_total = collect_counts(
        label_dir=label_dir,
        num_candidates=args.num_candidates,
        topk=args.topk,
        include_fallback_targets=args.include_fallback_targets,
    )
    save_counts_json(
        output_json=output_json,
        split_to_counts=split_to_counts,
        option_names=option_names,
        split_to_total=split_to_total,
        split_to_topk_total=split_to_topk_total,
        topk=args.topk,
        include_fallback_targets=args.include_fallback_targets,
    )
    plot_distribution(
        output_png=output_png,
        split_to_counts=split_to_counts,
        option_names=option_names,
        topk=args.topk,
        include_fallback_targets=args.include_fallback_targets,
    )

    print(f"Saved plot to {output_png}")
    print(f"Saved counts to {output_json}")


if __name__ == "__main__":
    main()
