from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val/test best_option_id distribution for number-probe data."
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help=(
            "Path to dataset_summary.json. Defaults to "
            "policy_network/results_number_probe/{label_kind}/lightning_class/dataset_summary.json."
        ),
    )
    parser.add_argument(
        "--label_kind",
        type=str,
        choices=["oracle", "policy"],
        default="oracle",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Defaults to the same directory as summary_json.",
    )
    return parser.parse_args()


def load_target_hist(summary_json: Path) -> Dict[str, List[int]]:
    with open(summary_json, "r") as f:
        summary = json.load(f)

    split_to_counts: Dict[str, List[int]] = {}
    for split in ("train", "val", "test"):
        raw_hist = summary[split]["target_hist"]
        counts = [0] * 27
        for option_id, count in raw_hist.items():
            counts[int(option_id)] = int(count)
        split_to_counts[split] = counts
    return split_to_counts


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    summary_json = (
        Path(args.summary_json)
        if args.summary_json is not None
        else ROOT
        / "policy_network"
        / "results_number_probe"
        / args.label_kind
        / "lightning_class"
        / "dataset_summary.json"
    )
    output_png = (
        Path(args.output_png)
        if args.output_png is not None
        else summary_json.parent / "index_distribution_train_val_test.png"
    )

    split_to_counts = load_target_hist(summary_json)
    option_ids = list(range(27))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(
        [x - width for x in option_ids],
        split_to_counts["train"],
        width=width,
        label="train",
    )
    ax.bar(option_ids, split_to_counts["val"], width=width, label="val")
    ax.bar(
        [x + width for x in option_ids],
        split_to_counts["test"],
        width=width,
        label="test",
    )

    ax.set_title("best_option_id Distribution")
    ax.set_xlabel("best_option_id")
    ax.set_ylabel("count")
    ax.set_xticks(option_ids)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
