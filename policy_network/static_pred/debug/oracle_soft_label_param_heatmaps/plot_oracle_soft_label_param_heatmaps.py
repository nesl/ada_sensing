from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPPING = ROOT / "data" / "option_param_mapping.json"
DEFAULT_WITH_FALLBACK_LABELS = (
    ROOT
    / "data"
    / "ImageNet-ES-Diverse"
    / "oracle_policy_labels"
    / "oracle_policy_all_labels.json"
)
DEFAULT_NO_FALLBACK_LABELS = (
    ROOT
    / "data"
    / "ImageNet-ES-Diverse"
    / "oracle_policy_labels_v2_allwrong_uniform_w01"
    / "oracle_policy_all_labels.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

APERTURES = ["f5.0", "f9.0", "f16"]
SS_VALUES = ["1/4", "1/60", "1/1000"]
ISO_VALUES = [250, 2000, 16000]
NUM_OPTIONS = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot oracle soft-label top-1 count heatmaps by A/ISO/SS."
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--with_fallback_labels", type=Path, default=DEFAULT_WITH_FALLBACK_LABELS
    )
    parser.add_argument(
        "--no_fallback_labels", type=Path, default=DEFAULT_NO_FALLBACK_LABELS
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tie_eps", type=float, default=1e-12)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def validate_mapping(rows: Sequence[Dict[str, Any]]) -> None:
    if len(rows) != NUM_OPTIONS:
        raise ValueError(f"Expected {NUM_OPTIONS} option mapping rows, got {len(rows)}")

    option_ids = sorted(int(row["option_id"]) for row in rows)
    expected = list(range(NUM_OPTIONS))
    if option_ids != expected:
        raise ValueError(f"Expected option ids {expected}, got {option_ids}")

    seen: Dict[Tuple[str, int, str], int] = {}
    for row in rows:
        key = (str(row["A"]), int(row["ISO"]), str(row["SS"]))
        option_id = int(row["option_id"])
        if key in seen:
            raise ValueError(
                f"Duplicate parameter key {key}: option {seen[key]} and {option_id}"
            )
        seen[key] = option_id

    expected_keys = {
        (aperture, iso, ss)
        for aperture in APERTURES
        for iso in ISO_VALUES
        for ss in SS_VALUES
    }
    missing = sorted(expected_keys - set(seen))
    if missing:
        raise ValueError(f"Missing parameter combinations: {missing}")


def argmax_with_tie_count(values: Sequence[float], eps: float) -> Tuple[int, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (NUM_OPTIONS,):
        raise ValueError(f"Expected soft_target length {NUM_OPTIONS}, got {array.size}")
    max_value = float(array.max())
    winners = np.flatnonzero(np.abs(array - max_value) <= eps)
    return int(winners[0]), int(len(winners))


def count_unique_top1(
    label_path: Path,
    *,
    skip_allwrong: bool,
    tie_eps: float,
) -> Tuple[List[int], Dict[str, Any]]:
    records = load_json(label_path)
    counts = [0 for _ in range(NUM_OPTIONS)]
    num_tied = 0
    num_counted = 0
    num_skipped_allwrong = 0
    num_oracle_allwrong = 0
    num_tied_oracle_allwrong = 0

    for record in records:
        soft_target = record.get("soft_target")
        if soft_target is None:
            raise ValueError(f"Missing soft_target for {record.get('sample_id')}")
        is_allwrong = not bool(record.get("oracle_had_correct_candidate", True))
        num_oracle_allwrong += int(is_allwrong)
        if skip_allwrong and is_allwrong:
            num_skipped_allwrong += 1
            continue
        top1, num_winners = argmax_with_tie_count(soft_target, tie_eps)
        if num_winners > 1:
            num_tied += 1
            num_tied_oracle_allwrong += int(is_allwrong)
        counts[top1] += 1
        num_counted += 1

    summary = {
        "label_path": str(label_path),
        "total_records": len(records),
        "counted_records": num_counted,
        "skipped_allwrong_records": num_skipped_allwrong,
        "tied_records": num_tied,
        "oracle_allwrong_records": num_oracle_allwrong,
        "tied_oracle_allwrong_records": num_tied_oracle_allwrong,
        "skip_allwrong": skip_allwrong,
        "tie_eps": tie_eps,
    }
    return counts, summary


def merged_rows(
    mapping_rows: Sequence[Dict[str, Any]],
    with_fallback_counts: Sequence[int],
    no_fallback_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mapping in sorted(mapping_rows, key=lambda row: int(row["option_id"])):
        option_id = int(mapping["option_id"])
        rows.append(
            {
                "option_id": option_id,
                "param_i": int(mapping["param_i"]),
                "A": str(mapping["A"]),
                "ISO": int(mapping["ISO"]),
                "SS": str(mapping["SS"]),
                "with_fallback_top1_count": int(with_fallback_counts[option_id]),
                "no_fallback_top1_count": int(no_fallback_counts[option_id]),
            }
        )
    return rows


def write_csv(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "option_id",
        "param_i",
        "A",
        "ISO",
        "SS",
        "with_fallback_top1_count",
        "no_fallback_top1_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matrix_for_metric(
    rows: Sequence[Dict[str, Any]], aperture: str, metric_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.full((len(ISO_VALUES), len(SS_VALUES)), np.nan, dtype=np.float64)
    option_ids = np.full((len(ISO_VALUES), len(SS_VALUES)), -1, dtype=np.int64)
    row_by_key = {
        (row["A"], int(row["ISO"]), row["SS"]): row
        for row in rows
        if row["A"] == aperture
    }
    for y, iso in enumerate(ISO_VALUES):
        for x, ss in enumerate(SS_VALUES):
            row = row_by_key[(aperture, iso, ss)]
            values[y, x] = float(row[metric_name])
            option_ids[y, x] = int(row["option_id"])
    return values, option_ids


def plot_heatmap_figure(
    rows: Sequence[Dict[str, Any]],
    metric_name: str,
    title: str,
    output_path: Path,
) -> None:
    all_values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
    vmax = float(np.nanmax(all_values))
    vmin = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    image = None
    for ax, aperture in zip(axes, APERTURES):
        values, option_ids = matrix_for_metric(rows, aperture, metric_name)
        image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"A = {aperture}")
        ax.set_xticks(np.arange(len(SS_VALUES)))
        ax.set_xticklabels(SS_VALUES)
        ax.set_yticks(np.arange(len(ISO_VALUES)))
        ax.set_yticklabels([str(value) for value in ISO_VALUES])
        ax.set_xlabel("SS")
        ax.set_ylabel("ISO")

        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                text_color = "white" if value > 0.55 * vmax else "black"
                ax.text(
                    x,
                    y,
                    str(option_ids[y, x]),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(-0.5, len(SS_VALUES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ISO_VALUES), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(title, fontsize=14)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, shrink=0.9)
        cbar.set_label("Top-1 count")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mapping_rows = load_json(args.mapping)
    validate_mapping(mapping_rows)

    with_fallback_counts, with_fallback_summary = count_unique_top1(
        args.with_fallback_labels,
        skip_allwrong=False,
        tie_eps=args.tie_eps,
    )
    no_fallback_counts, no_fallback_summary = count_unique_top1(
        args.no_fallback_labels,
        skip_allwrong=True,
        tie_eps=args.tie_eps,
    )

    rows = merged_rows(mapping_rows, with_fallback_counts, no_fallback_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "oracle_soft_top1_counts.csv"
    json_path = args.output_dir / "oracle_soft_top1_counts.json"
    with_fallback_png = args.output_dir / "with_fallback_top1_count_heatmap.png"
    no_fallback_png = args.output_dir / "no_fallback_top1_count_heatmap.png"

    write_csv(rows, csv_path)
    summary = {
        "mapping": str(args.mapping),
        "metrics": {
            "with_fallback_top1_count": with_fallback_summary,
            "no_fallback_top1_count": no_fallback_summary,
        },
        "counts": rows,
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    plot_heatmap_figure(
        rows,
        "with_fallback_top1_count",
        "Oracle Soft Label Top-1 Count With Downstream-All-Wrong Fallback (All Set)",
        with_fallback_png,
    )
    plot_heatmap_figure(
        rows,
        "no_fallback_top1_count",
        "Oracle Soft Label Top-1 Count Without Fallback (All Set, All-Wrong Skipped)",
        no_fallback_png,
    )

    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {with_fallback_png}")
    print(f"Saved {no_fallback_png}")
    print(
        "No-fallback all-wrong records skipped: "
        f"{no_fallback_summary['skipped_allwrong_records']}"
    )


if __name__ == "__main__":
    main()
