from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LABEL_DIR = ROOT / "data" / "ImageNet-ES-Diverse" / "oracle_policy_labels"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all-set downstream-correct option_id distribution for selected envs."
        )
    )
    parser.add_argument("--label_dir", type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--envs", type=str, default="l1,l7")
    parser.add_argument("--num_options", type=int, default=27)
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def env_sort_key(env_name: str) -> tuple[int, str]:
    if env_name.startswith("l") and env_name[1:].isdigit():
        return int(env_name[1:]), env_name
    return 999, env_name


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_envs = [env.strip() for env in args.envs.split(",") if env.strip()]
    env_to_counts = {env: Counter() for env in keep_envs}
    env_to_samples = {env: 0 for env in keep_envs}

    for split in ("train", "val", "test"):
        path = label_dir / f"oracle_policy_{split}_labels.json"
        for item in load_json(path):
            env = str(item.get("env"))
            if env not in env_to_counts:
                continue
            env_to_samples[env] += 1
            for option_id in item.get("oracle_correct_option_ids", []):
                env_to_counts[env][int(option_id)] += 1

    envs = sorted(keep_envs, key=env_sort_key)
    option_ids = np.arange(args.num_options)
    width = 0.8 / max(1, len(envs))

    fig, ax = plt.subplots(figsize=(12, 5))
    for env_idx, env in enumerate(envs):
        counts = [env_to_counts[env].get(int(option_id), 0) for option_id in option_ids]
        offset = (env_idx - (len(envs) - 1) / 2) * width
        ax.bar(option_ids + offset, counts, width=width, label=f"{env} (n={env_to_samples[env]})")

    ax.set_title("Downstream-correct option_id distribution by env (train+val+test)")
    ax.set_xlabel("option_id")
    ax.set_ylabel("correct count")
    ax.set_xticks(option_ids)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "all_env_downstream_correct_index_distribution.png"
    fig.savefig(plot_path, dpi=200)

    summary = {
        "label_dir": str(label_dir),
        "envs": envs,
        "sample_count_by_env": env_to_samples,
        "correct_index_count_by_env": {
            env: {str(option_id): env_to_counts[env].get(option_id, 0) for option_id in range(args.num_options)}
            for env in envs
        },
    }
    summary_path = output_dir / "all_env_downstream_correct_index_distribution.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
