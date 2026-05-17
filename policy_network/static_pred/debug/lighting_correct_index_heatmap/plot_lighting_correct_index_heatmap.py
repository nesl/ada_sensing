from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LABEL_PATH = (
    ROOT
    / "data"
    / "ImageNet-ES-Diverse"
    / "oracle_policy_labels"
    / "oracle_policy_all_labels.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_ENVS = ("l1", "l2", "l3", "l4", "l6", "l7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a lighting x option heatmap of downstream-correct option counts. "
            "Every option in oracle_correct_option_ids is counted."
        )
    )
    parser.add_argument("--label_path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_options", type=int, default=27)
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Draw numeric counts in each heatmap cell.",
    )
    return parser.parse_args()


def load_items(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Expected a list in {path}")
    return items


def parse_envs(envs: str) -> List[str]:
    parsed = [env.strip() for env in envs.split(",") if env.strip()]
    if not parsed:
        raise ValueError("At least one env is required.")
    return parsed


def validate_option_ids(
    option_ids: Iterable[Any],
    num_options: int,
    sample_id: str,
) -> List[int]:
    valid_option_ids: List[int] = []
    for raw_option_id in option_ids:
        option_id = int(raw_option_id)
        if option_id < 0 or option_id >= num_options:
            raise ValueError(
                f"option_id={option_id} outside [0, {num_options}) "
                f"for sample_id={sample_id}"
            )
        valid_option_ids.append(option_id)
    return valid_option_ids


def build_matrix(
    items: List[Dict[str, Any]],
    envs: List[str],
    num_options: int,
) -> tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    env_to_row = {env: row_idx for row_idx, env in enumerate(envs)}
    matrix = np.zeros((len(envs), num_options), dtype=np.int64)
    sample_count_by_env = Counter({env: 0 for env in envs})
    correct_entry_count_by_env = Counter({env: 0 for env in envs})

    for item in items:
        env = str(item.get("env", ""))
        if env not in env_to_row:
            continue

        sample_count_by_env[env] += 1
        option_ids = validate_option_ids(
            item.get("oracle_correct_option_ids", []),
            num_options=num_options,
            sample_id=str(item.get("sample_id", "<missing sample_id>")),
        )
        correct_entry_count_by_env[env] += len(option_ids)
        for option_id in option_ids:
            matrix[env_to_row[env], option_id] += 1

    return (
        matrix,
        {env: int(sample_count_by_env[env]) for env in envs},
        {env: int(correct_entry_count_by_env[env]) for env in envs},
    )


def save_csv(path: Path, matrix: np.ndarray, envs: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env", *[f"index_{idx}" for idx in range(matrix.shape[1])]])
        for env, row in zip(envs, matrix.tolist()):
            writer.writerow([env, *row])


def save_json(
    path: Path,
    label_path: Path,
    matrix: np.ndarray,
    envs: List[str],
    sample_count_by_env: Dict[str, int],
    correct_entry_count_by_env: Dict[str, int],
) -> None:
    payload = {
        "label_path": str(label_path),
        "counting_rule": "Count every option_id in oracle_correct_option_ids.",
        "envs": envs,
        "num_options": int(matrix.shape[1]),
        "sample_count_by_env": sample_count_by_env,
        "correct_entry_count_by_env": correct_entry_count_by_env,
        "matrix_counts": {
            env: {str(option_id): int(matrix[row_idx, option_id]) for option_id in range(matrix.shape[1])}
            for row_idx, env in enumerate(envs)
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    envs: List[str],
    sample_count_by_env: Dict[str, int],
    annotate: bool,
) -> None:
    fig_width = max(14, matrix.shape[1] * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="downstream-correct count")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(idx) for idx in range(matrix.shape[1])], fontsize=8)
    ax.set_yticks(np.arange(len(envs)))
    ax.set_yticklabels([f"{env} (n={sample_count_by_env[env]})" for env in envs])
    ax.set_xlabel("index / option_id")
    ax.set_ylabel("lighting / env")
    ax.set_title("Downstream-correct index count by lighting")

    if annotate:
        threshold = float(matrix.max()) * 0.55 if matrix.size else 0.0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(matrix[row_idx, col_idx])
                color = "white" if value < threshold else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=6, color=color)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    label_path = args.label_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = parse_envs(args.envs)
    items = load_items(label_path)
    matrix, sample_count_by_env, correct_entry_count_by_env = build_matrix(
        items=items,
        envs=envs,
        num_options=args.num_options,
    )

    np.save(output_dir / "lighting_correct_index_counts.npy", matrix)
    save_csv(output_dir / "lighting_correct_index_counts.csv", matrix, envs)
    save_json(
        output_dir / "lighting_correct_index_counts.json",
        label_path=label_path,
        matrix=matrix,
        envs=envs,
        sample_count_by_env=sample_count_by_env,
        correct_entry_count_by_env=correct_entry_count_by_env,
    )
    plot_heatmap(
        output_dir / "lighting_correct_index_heatmap.png",
        matrix=matrix,
        envs=envs,
        sample_count_by_env=sample_count_by_env,
        annotate=args.annotate,
    )

    print(f"Saved outputs to: {output_dir}")
    print(f"matrix shape: {matrix.shape}")
    print(f"sample_count_by_env: {sample_count_by_env}")


if __name__ == "__main__":
    main()
