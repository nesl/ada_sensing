from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
            "Build oracle top-k option co-occurrence matrices on the all split. "
            "The include-fallback matrix uses every sample's soft_target, while "
            "the correct-only matrix skips all-wrong fallback samples."
        )
    )
    parser.add_argument("--label_path", type=str, default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_candidates", type=int, default=27)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--top_pairs", type=int, default=40)
    return parser.parse_args()


def load_items(label_path: Path) -> List[Dict[str, Any]]:
    with open(label_path, "r") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of records in {label_path}")
    return items


def check_correct_flag_consistency(items: List[Dict[str, Any]]) -> None:
    mismatches = []
    for item in items:
        had_correct = bool(item.get("oracle_had_correct_candidate", False))
        num_correct = int(item.get("oracle_num_correct_candidates", 0))
        if had_correct != (num_correct > 0):
            mismatches.append(item.get("sample_id", "<missing sample_id>"))

    if mismatches:
        preview = ", ".join(mismatches[:5])
        raise ValueError(
            "oracle_had_correct_candidate is inconsistent with "
            f"oracle_num_correct_candidates > 0 for {len(mismatches)} samples: {preview}"
        )


def get_topk_option_ids(
    item: Dict[str, Any],
    num_candidates: int,
    topk: int,
) -> List[int]:
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
    positive.sort(key=lambda pair: (-pair[1], pair[0]))
    topk_option_ids = [option_id for option_id, _ in positive[:topk]]

    for option_id in topk_option_ids:
        if option_id < 0 or option_id >= num_candidates:
            raise ValueError(
                f"Found option_id={option_id} outside [0, {num_candidates}) "
                f"for sample_id={item.get('sample_id')}"
            )
    return topk_option_ids


def collect_option_names(items: List[Dict[str, Any]], num_candidates: int) -> Dict[int, str]:
    option_names: Dict[int, str] = {}
    for item in items:
        if "best_option_id" in item and "best_option_name" in item:
            option_names[int(item["best_option_id"])] = str(item["best_option_name"])

    return {
        option_id: option_names.get(option_id, "")
        for option_id in range(num_candidates)
    }


def build_cooccurrence_matrix(
    items: List[Dict[str, Any]],
    num_candidates: int,
    topk: int,
    correct_only: bool,
) -> Tuple[np.ndarray, Dict[str, int], List[int]]:
    matrix = np.zeros((num_candidates, num_candidates), dtype=np.int64)
    diagonal_counts = np.zeros(num_candidates, dtype=np.int64)

    used_samples = 0
    skipped_samples = 0
    empty_topk_samples = 0
    total_topk_entries = 0

    for item in items:
        had_correct = bool(item.get("oracle_had_correct_candidate", False))
        if correct_only and not had_correct:
            skipped_samples += 1
            continue

        topk_option_ids = get_topk_option_ids(
            item=item,
            num_candidates=num_candidates,
            topk=topk,
        )
        if not topk_option_ids:
            empty_topk_samples += 1

        used_samples += 1
        total_topk_entries += len(topk_option_ids)

        for option_id in topk_option_ids:
            diagonal_counts[option_id] += 1

        for row_option_id in topk_option_ids:
            for col_option_id in topk_option_ids:
                matrix[row_option_id, col_option_id] += 1

    if not np.array_equal(matrix, matrix.T):
        raise ValueError("Co-occurrence matrix is not symmetric.")
    if not np.array_equal(np.diag(matrix), diagonal_counts):
        raise ValueError("Matrix diagonal does not match option appearance counts.")

    stats = {
        "total_samples": len(items),
        "used_samples": used_samples,
        "skipped_samples": skipped_samples,
        "empty_topk_samples": empty_topk_samples,
        "total_topk_entries": total_topk_entries,
    }
    return matrix, stats, diagonal_counts.tolist()


def get_top_offdiagonal_pairs(
    matrix: np.ndarray,
    option_names: Dict[int, str],
    limit: int,
) -> List[Dict[str, Any]]:
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            pairs.append(
                {
                    "option_i": i,
                    "option_j": j,
                    "option_i_name": option_names.get(i, ""),
                    "option_j_name": option_names.get(j, ""),
                    "count": int(matrix[i, j]),
                }
            )
    pairs.sort(key=lambda item: (-item["count"], item["option_i"], item["option_j"]))
    return pairs[:limit]


def save_json(
    output_path: Path,
    mode: str,
    label_path: Path,
    matrix: np.ndarray,
    stats: Dict[str, int],
    diagonal_counts: List[int],
    option_names: Dict[int, str],
    top_pairs: List[Dict[str, Any]],
    topk: int,
) -> None:
    payload = {
        "mode": mode,
        "label_path": str(label_path),
        "num_candidates": int(matrix.shape[0]),
        "topk": topk,
        **stats,
        "option_names": {str(k): v for k, v in option_names.items()},
        "diagonal_counts": diagonal_counts,
        "top_offdiagonal_pairs": top_pairs,
        "matrix_counts": matrix.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def format_option_label(option_id: int, option_names: Dict[int, str]) -> str:
    option_name = option_names.get(option_id, "")
    if option_name:
        return f"{option_id}:{option_name}"
    return str(option_id)


def save_matrix_csv(
    output_path: Path,
    matrix: np.ndarray,
    option_names: Dict[int, str],
) -> None:
    labels = [
        format_option_label(option_id, option_names)
        for option_id in range(matrix.shape[0])
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["option"] + labels)
        for option_id, row in enumerate(matrix.tolist()):
            writer.writerow([labels[option_id]] + row)


def save_diagonal_csv(
    output_path: Path,
    diagonal_counts: List[int],
    option_names: Dict[int, str],
) -> None:
    rows = [
        {
            "option_id": option_id,
            "option_name": option_names.get(option_id, ""),
            "count": int(count),
        }
        for option_id, count in enumerate(diagonal_counts)
    ]
    rows.sort(key=lambda row: (-row["count"], row["option_id"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["option_id", "option_name", "count"])
        writer.writeheader()
        writer.writerows(rows)


def save_top_pairs_csv(
    output_path: Path,
    top_pairs: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "option_i",
        "option_i_name",
        "option_j",
        "option_j_name",
        "count",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in top_pairs:
            writer.writerow({field: pair[field] for field in fieldnames})


def plot_heatmap(
    output_path: Path,
    matrix: np.ndarray,
    option_names: Dict[int, str],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="count")

    option_ids = np.arange(matrix.shape[0])
    labels = [
        f"{option_id}\n{option_names.get(option_id, '')}"
        for option_id in option_ids
    ]
    ax.set_xticks(option_ids)
    ax.set_yticks(option_ids)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("option_id in top-k set")
    ax.set_ylabel("option_id in top-k set")
    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_mode(
    mode: str,
    items: List[Dict[str, Any]],
    label_path: Path,
    output_dir: Path,
    num_candidates: int,
    topk: int,
    top_pairs_limit: int,
    correct_only: bool,
    option_names: Dict[int, str],
) -> Dict[str, int]:
    matrix, stats, diagonal_counts = build_cooccurrence_matrix(
        items=items,
        num_candidates=num_candidates,
        topk=topk,
        correct_only=correct_only,
    )
    top_pairs = get_top_offdiagonal_pairs(
        matrix=matrix,
        option_names=option_names,
        limit=top_pairs_limit,
    )

    np.save(output_dir / f"{mode}_counts.npy", matrix)
    save_matrix_csv(
        output_path=output_dir / f"{mode}_matrix_counts.csv",
        matrix=matrix,
        option_names=option_names,
    )
    save_diagonal_csv(
        output_path=output_dir / f"{mode}_diagonal_counts.csv",
        diagonal_counts=diagonal_counts,
        option_names=option_names,
    )
    save_top_pairs_csv(
        output_path=output_dir / f"{mode}_top_pairs.csv",
        top_pairs=top_pairs,
    )
    save_json(
        output_path=output_dir / f"{mode}_counts.json",
        mode=mode,
        label_path=label_path,
        matrix=matrix,
        stats=stats,
        diagonal_counts=diagonal_counts,
        option_names=option_names,
        top_pairs=top_pairs,
        topk=topk,
    )
    plot_heatmap(
        output_path=output_dir / f"{mode}_heatmap.png",
        matrix=matrix,
        option_names=option_names,
        title=(
            f"Oracle Top-{topk} Co-occurrence ({mode.replace('_', ' ')}, "
            f"N={stats['used_samples']})"
        ),
    )
    return stats


def main() -> None:
    args = parse_args()
    label_path = Path(args.label_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(label_path)
    check_correct_flag_consistency(items)
    option_names = collect_option_names(items, args.num_candidates)

    include_stats = run_mode(
        mode="include_fallback",
        items=items,
        label_path=label_path,
        output_dir=output_dir,
        num_candidates=args.num_candidates,
        topk=args.topk,
        top_pairs_limit=args.top_pairs,
        correct_only=False,
        option_names=option_names,
    )
    correct_stats = run_mode(
        mode="correct_only",
        items=items,
        label_path=label_path,
        output_dir=output_dir,
        num_candidates=args.num_candidates,
        topk=args.topk,
        top_pairs_limit=args.top_pairs,
        correct_only=True,
        option_names=option_names,
    )

    print(f"Saved outputs to {output_dir}")
    print(f"include_fallback: {include_stats}")
    print(f"correct_only: {correct_stats}")


if __name__ == "__main__":
    main()
