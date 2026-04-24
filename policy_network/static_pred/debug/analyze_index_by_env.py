"""visualize the best_option_id in training dataset separated by environment"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data" / "ImageNet-ES-Diverse"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Directory containing lightning_class/, lightning/, and class/. "
            "Defaults to policy_network/results_number_probe/{label_kind}."
        ),
    )
    parser.add_argument(
        "--label_kind",
        type=str,
        choices=["oracle", "policy"],
        default="oracle",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to results_root.",
    )
    return parser.parse_args()


def get_label_paths(label_kind: str) -> Tuple[Path, Path, Path]:
    if label_kind == "oracle":
        label_dir = DEFAULT_DATA_ROOT / "oracle_policy_labels"
        prefix = "oracle_policy"
    elif label_kind == "policy":
        label_dir = DEFAULT_DATA_ROOT / "policy_labels"
        prefix = "policy"
    else:
        raise ValueError(f"Unsupported label_kind={label_kind}")
    return (
        label_dir / f"{prefix}_train_labels.json",
        label_dir / f"{prefix}_val_labels.json",
        label_dir / f"{prefix}_test_labels.json",
    )

def load_json(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def main() -> None:
    args = parse_args()

    if args.results_root is None:
        args.results_root = str(
            ROOT / "policy_network" / "results_number_probe" / args.label_kind
        )
    if args.output_dir is None:
        args.output_dir = args.results_root
    os.makedirs(args.output_dir, exist_ok=True)

    train_json, val_json, test_json = get_label_paths(args.label_kind)

    # just use the trainig set
    train_items = load_json(train_json)

    # read the best_option_id for each item and group by environment
    env_to_option_ids: Dict[str, List[int]] = {}
    for item in train_items:
        env = item.get("env")
        best_option_id = item.get("best_option_id")
        if env is None or best_option_id is None:
            continue
        env_to_option_ids.setdefault(env, []).append(int(best_option_id))

    if not env_to_option_ids:
        raise ValueError("No env or best_option_id found in training data.")

    def env_sort_key(env_name: str) -> Tuple[int, str]:
        if env_name.startswith("l") and env_name[1:].isdigit():
            return (int(env_name[1:]), env_name)
        return (999, env_name)

    env_names = sorted(env_to_option_ids.keys(), key=env_sort_key)
    all_option_ids = [opt for opts in env_to_option_ids.values() for opt in opts]
    max_option_id = max(all_option_ids)
    option_ids = np.arange(max_option_id + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, env_name in enumerate(env_names):
        ax = axes[idx]
        counts = np.bincount(
            env_to_option_ids[env_name], minlength=max_option_id + 1
        )
        ax.bar(option_ids, counts, color="#4C72B0", alpha=0.9)
        ax.set_title(f"env {env_name}")
        ax.set_xlabel("best_option_id")
        ax.set_ylabel("count")

    for idx in range(len(env_names), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"Best option id distribution by environment (train, {args.label_kind})"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = Path(args.output_dir) / f"{args.label_kind}_best_option_id_by_env.png"
    fig.savefig(output_path, dpi=200)

    summary_path = Path(args.output_dir) / f"{args.label_kind}_best_option_id_by_env.json"
    with open(summary_path, "w") as f:
        json.dump(env_to_option_ids, f, indent=2)

    print(f"Saved plot to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
    