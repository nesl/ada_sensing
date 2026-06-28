from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LABEL_PATH = (
    ROOT
    / "data"
    / "ImageNet-ES-Diverse"
    / "oracle_policy_labels"
    / "oracle_policy_all_labels.json"
)
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "all_downstream_raw_matrix.csv"
DEFAULT_HISTOGRAM_PATH = Path(__file__).resolve().parent / "all_downstream_correct_count_histogram.png"
DEFAULT_COUNT_DISTRIBUTION_CSV = (
    Path(__file__).resolve().parent / "all_downstream_correct_count_distribution.csv"
)
DEFAULT_ENVS = ("l1", "l2", "l3", "l4", "l6", "l7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a downstream raw correctness matrix to CSV. Each row is one "
            "sample under one lighting/env. Columns option_id_0..option_id_26 are "
            "1 only when that option is truly downstream-correct, otherwise 0."
        )
    )
    parser.add_argument("--label_path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--histogram_path", type=Path, default=DEFAULT_HISTOGRAM_PATH)
    parser.add_argument(
        "--count_distribution_csv",
        type=Path,
        default=DEFAULT_COUNT_DISTRIBUTION_CSV,
    )
    parser.add_argument("--num_options", type=int, default=27)
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Prefix rows with sample_id, group_id, class_id, env, and label columns.",
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


def build_raw_matrix_rows(
    items: List[Dict[str, Any]],
    envs: List[str],
    num_options: int,
) -> List[Dict[str, Any]]:
    env_order = {env: idx for idx, env in enumerate(envs)}
    rows: List[Dict[str, Any]] = []

    for item in items:
        env = str(item.get("env", ""))
        if env not in env_order:
            continue

        sample_id = str(item.get("sample_id", "<missing sample_id>"))
        had_correct_candidate = bool(item.get("oracle_had_correct_candidate", False))
        correct_option_ids = validate_option_ids(
            item.get("oracle_correct_option_ids", []),
            num_options=num_options,
            sample_id=sample_id,
        )

        values = [0 for _ in range(num_options)]
        if had_correct_candidate:
            for option_id in correct_option_ids:
                values[option_id] = 1

        rows.append(
            {
                "sample_id": sample_id,
                "group_id": str(item.get("group_id", "")),
                "class_id": str(item.get("class_id", "")),
                "env": env,
                "label": int(item.get("label", -1)),
                "env_order": env_order[env],
                "values": values,
            }
        )

    return sorted(rows, key=lambda row: (row["sample_id"], row["env_order"]))


def save_csv(path: Path, rows: List[Dict[str, Any]], num_options: int, include_metadata: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    option_columns = [f"option_id_{idx}" for idx in range(num_options)]
    count_column = ["correct_count"]
    metadata_columns = ["sample_id", "group_id", "class_id", "env", "label"]
    header = (
        [*metadata_columns, *option_columns, *count_column]
        if include_metadata
        else [*option_columns, *count_column]
    )

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            metadata = [row[column] for column in metadata_columns]
            values = row["values"]
            correct_count = sum(values)
            writer.writerow(
                [*metadata, *values, correct_count]
                if include_metadata
                else [*values, correct_count]
            )


def build_count_distribution(rows: List[Dict[str, Any]], num_options: int) -> List[int]:
    distribution = [0 for _ in range(num_options + 1)]
    for row in rows:
        correct_count = sum(row["values"])
        distribution[correct_count] += 1
    return distribution


def save_count_distribution_csv(path: Path, distribution: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["correct_count", "num_samples"])
        for correct_count, num_samples in enumerate(distribution):
            if num_samples > 0:
                writer.writerow([correct_count, num_samples])


def plot_count_histogram(path: Path, distribution: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = list(range(len(distribution)))

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x_values, distribution, color="#2f6f7e", edgecolor="#1f2933", linewidth=0.6)
    ax.set_xlabel("count of correct downstream options")
    ax.set_ylabel("number of samples")
    ax.set_title("Downstream Correct Count Distribution")
    ax.set_xticks(x_values)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    envs = parse_envs(args.envs)
    items = load_items(args.label_path)
    rows = build_raw_matrix_rows(
        items=items,
        envs=envs,
        num_options=args.num_options,
    )
    save_csv(
        path=args.output_csv,
        rows=rows,
        num_options=args.num_options,
        include_metadata=args.include_metadata,
    )
    distribution = build_count_distribution(rows, num_options=args.num_options)
    save_count_distribution_csv(args.count_distribution_csv, distribution)
    plot_count_histogram(args.histogram_path, distribution)

    all_wrong_rows = sum(1 for row in rows if sum(row["values"]) == 0)
    print(f"Saved downstream raw matrix CSV to: {args.output_csv}")
    print(f"Saved count distribution CSV to: {args.count_distribution_csv}")
    print(f"Saved count histogram to: {args.histogram_path}")
    print(f"rows: {len(rows)}")
    print(f"option columns: {args.num_options}")
    print("extra columns: correct_count")
    print(f"all-zero rows: {all_wrong_rows}")


if __name__ == "__main__":
    main()
