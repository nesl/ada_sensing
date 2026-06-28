from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 27x27 co-winner matrix. M[i,j] counts samples where both "
            "option/index i and option/index j are downstream-correct. The "
            "diagonal is set to 0."
        )
    )
    parser.add_argument("--label_path", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_options", type=int, default=27)
    parser.add_argument("--top_pairs", type=int, default=50)
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


def has_non_fallback_correct_candidate(item: Dict[str, Any]) -> bool:
    return (
        bool(item.get("oracle_had_correct_candidate", False))
        and int(item.get("oracle_num_correct_candidates", 0)) > 0
    )


def validate_option_ids(
    raw_option_ids: Iterable[Any],
    num_options: int,
    sample_id: str,
) -> List[int]:
    option_ids = sorted({int(option_id) for option_id in raw_option_ids})
    for option_id in option_ids:
        if option_id < 0 or option_id >= num_options:
            raise ValueError(
                f"option_id={option_id} outside [0, {num_options}) "
                f"for sample_id={sample_id}"
            )
    return option_ids


def build_co_winner_matrix(
    items: List[Dict[str, Any]],
    num_options: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    matrix = np.zeros((num_options, num_options), dtype=np.int64)
    single_counts = np.zeros(num_options, dtype=np.int64)
    samples_with_any_correct = 0
    samples_with_pair = 0
    skipped_all_wrong_fallback = 0
    total_correct_entries = 0

    for item in items:
        if not has_non_fallback_correct_candidate(item):
            skipped_all_wrong_fallback += 1
            continue

        sample_id = str(item.get("sample_id", "<missing sample_id>"))
        option_ids = validate_option_ids(
            item.get("oracle_correct_option_ids", []),
            num_options=num_options,
            sample_id=sample_id,
        )
        if option_ids:
            samples_with_any_correct += 1
        if len(option_ids) >= 2:
            samples_with_pair += 1

        total_correct_entries += len(option_ids)
        for option_id in option_ids:
            single_counts[option_id] += 1

        for option_i in option_ids:
            for option_j in option_ids:
                if option_i != option_j:
                    matrix[option_i, option_j] += 1

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Co-winner matrix is not symmetric.")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("Co-winner matrix diagonal must stay zero.")

    stats = {
        "total_samples": len(items),
        "samples_with_any_correct": samples_with_any_correct,
        "samples_with_pair": samples_with_pair,
        "skipped_all_wrong_fallback": skipped_all_wrong_fallback,
        "total_correct_entries": total_correct_entries,
    }
    return matrix, single_counts, stats


def save_matrix_csv(
    path: Path,
    matrix: np.ndarray,
) -> None:
    labels = [str(option_id) for option_id in range(matrix.shape[0])]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["option", *labels])
        for option_id, row in enumerate(matrix.tolist()):
            writer.writerow([labels[option_id], *row])


def save_single_counts_csv(
    path: Path,
    single_counts: np.ndarray,
) -> None:
    rows = [
        {
            "option_id": option_id,
            "downstream_correct_count": int(single_counts[option_id]),
        }
        for option_id in range(len(single_counts))
    ]
    rows.sort(key=lambda row: (-row["downstream_correct_count"], row["option_id"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["option_id", "downstream_correct_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def get_top_pairs(
    matrix: np.ndarray,
    limit: int,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for option_i in range(matrix.shape[0]):
        for option_j in range(option_i + 1, matrix.shape[1]):
            pairs.append(
                {
                    "option_i": option_i,
                    "option_j": option_j,
                    "count": int(matrix[option_i, option_j]),
                }
            )
    pairs.sort(key=lambda row: (-row["count"], row["option_i"], row["option_j"]))
    return pairs[:limit]


def save_top_pairs_csv(path: Path, top_pairs: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "option_i",
        "option_j",
        "count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in top_pairs:
            writer.writerow({field: pair[field] for field in fieldnames})


def save_json(
    path: Path,
    label_path: Path,
    matrix: np.ndarray,
    single_counts: np.ndarray,
    stats: Dict[str, int],
    top_pairs: List[Dict[str, Any]],
) -> None:
    payload = {
        "label_path": str(label_path),
        "counting_rule": (
            "M[i,j] counts samples where both i and j are in "
            "oracle_correct_option_ids, after skipping all-wrong fallback "
            "records; diagonal entries are set to 0."
        ),
        "num_options": int(matrix.shape[0]),
        **stats,
        "single_index_correct_counts": {
            str(option_id): int(single_counts[option_id])
            for option_id in range(len(single_counts))
        },
        "top_co_winner_pairs": top_pairs,
        "matrix_counts": matrix.tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    stats: Dict[str, int],
    annotate: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="co-winner count")

    option_ids = np.arange(matrix.shape[0])
    labels = [str(option_id) for option_id in option_ids]
    ax.set_xticks(option_ids)
    ax.set_yticks(option_ids)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("downstream-correct index j")
    ax.set_ylabel("downstream-correct index i")
    ax.set_title(
        "Downstream co-winner matrix "
        f"(N={stats['total_samples']}, pair samples={stats['samples_with_pair']})"
    )

    if annotate:
        threshold = float(matrix.max()) * 0.55 if matrix.size else 0.0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(matrix[row_idx, col_idx])
                color = "white" if value < threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=color,
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.label_path)
    matrix, single_counts, stats = build_co_winner_matrix(
        items=items,
        num_options=args.num_options,
    )
    top_pairs = get_top_pairs(
        matrix=matrix,
        limit=args.top_pairs,
    )

    np.save(args.output_dir / "co_winner_matrix.npy", matrix)
    save_matrix_csv(
        path=args.output_dir / "co_winner_matrix.csv",
        matrix=matrix,
    )
    save_single_counts_csv(
        path=args.output_dir / "single_index_correct_counts.csv",
        single_counts=single_counts,
    )
    save_top_pairs_csv(
        path=args.output_dir / "top_co_winner_pairs.csv",
        top_pairs=top_pairs,
    )
    save_json(
        path=args.output_dir / "co_winner_matrix.json",
        label_path=args.label_path,
        matrix=matrix,
        single_counts=single_counts,
        stats=stats,
        top_pairs=top_pairs,
    )
    plot_heatmap(
        path=args.output_dir / "co_winner_heatmap.png",
        matrix=matrix,
        stats=stats,
        annotate=args.annotate,
    )

    print(f"Saved outputs to: {args.output_dir}")
    print(f"matrix shape: {matrix.shape}")
    print(f"stats: {stats}")
    if top_pairs:
        print(f"top pair: {top_pairs[0]}")


if __name__ == "__main__":
    main()
