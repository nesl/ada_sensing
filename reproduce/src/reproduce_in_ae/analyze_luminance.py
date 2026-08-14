from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lenz_reproduce_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .datasets import DatasetRoots, build_dataset, default_roots
from .exposure import (
    ExposureSpec,
    apply_exposure,
    linear_luminance,
    srgb_u8_to_linear,
)
from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    workspace_root,
)


PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
HISTOGRAM_BINS = 256
PER_IMAGE_FIELDS = (
    "dataset",
    "environment",
    "shot",
    "class_id",
    "image_id",
    "path",
    "width",
    "height",
    "pixel_count",
    "mean_luminance",
    "log_mean_luminance",
    "std_luminance",
    "p01_luminance",
    "p05_luminance",
    "p25_luminance",
    "p50_luminance",
    "p75_luminance",
    "p95_luminance",
    "p99_luminance",
    "dynamic_range_stops_p01_p99",
    "near_black_fraction",
    "near_white_fraction",
    "any_channel_zero_fraction",
    "all_channels_zero_fraction",
    "any_channel_saturated_fraction",
    "all_channels_saturated_fraction",
)
NUMERIC_METRICS = PER_IMAGE_FIELDS[9:]
EXPECTED_COUNTS = {
    DATASET_AE_ES: 10_000,
    DATASET_AE_DIVERSE: 30_000,
}


def parse_args() -> argparse.Namespace:
    defaults = default_roots(workspace_root())
    parser = argparse.ArgumentParser(
        description="Analyze full-resolution original AE JPEG luminance."
    )
    parser.add_argument("--ae-es-root", type=Path, default=defaults.ae_es_root)
    parser.add_argument(
        "--ae-diverse-root", type=Path, default=defaults.ae_diverse_root
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ae_luminance",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_images(roots: DatasetRoots) -> list[tuple[str, str, str, str, str, str]]:
    items: list[tuple[str, str, str, str, str, str]] = []
    for dataset_name, root in (
        (DATASET_AE_ES, roots.ae_es_root),
        (DATASET_AE_DIVERSE, roots.ae_diverse_root),
    ):
        dataset = build_dataset(dataset_name, roots, transform=lambda image: image)
        for setting, child in dataset.datasets:
            environment, shot = setting.split("/", maxsplit=1)
            for path_string, _target in child.samples:
                path = Path(path_string).resolve()
                items.append(
                    (
                        dataset_name,
                        environment,
                        shot,
                        path.parent.name,
                        path.stem,
                        str(path),
                    )
                )
    items.sort()
    paths = [item[-1] for item in items]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate AE image paths were discovered")
    for dataset_name, expected in EXPECTED_COUNTS.items():
        found = sum(item[0] == dataset_name for item in items)
        if found != expected:
            raise ValueError(
                f"{dataset_name}: expected {expected} images, discovered {found}"
            )
    return items


def analyze_one(
    item: tuple[str, str, str, str, str, str]
) -> tuple[Dict[str, Any], np.ndarray]:
    dataset, environment, shot, class_id, image_id, path_string = item
    path = Path(path_string)
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    linear = srgb_u8_to_linear(rgb)
    luminance = linear_luminance(linear)
    percentile_values = np.percentile(luminance, PERCENTILES)
    mean_luminance = float(luminance.mean(dtype=np.float64))
    p01 = float(percentile_values[0])
    p99 = float(percentile_values[-1])
    histogram, _edges = np.histogram(
        luminance, bins=HISTOGRAM_BINS, range=(0.0, 1.0)
    )
    row: Dict[str, Any] = {
        "dataset": dataset,
        "environment": environment,
        "shot": shot,
        "class_id": class_id,
        "image_id": image_id,
        "path": str(path.resolve()),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "pixel_count": int(rgb.shape[0] * rgb.shape[1]),
        "mean_luminance": mean_luminance,
        "log_mean_luminance": float(
            np.exp(np.log(luminance + 1e-6).mean(dtype=np.float64)) - 1e-6
        ),
        "std_luminance": float(luminance.std(dtype=np.float64)),
        **{
            f"p{percentile:02d}_luminance": float(value)
            for percentile, value in zip(PERCENTILES, percentile_values)
        },
        "dynamic_range_stops_p01_p99": float(
            math.log2((p99 + 1e-6) / (p01 + 1e-6))
        ),
        "near_black_fraction": float(np.mean(luminance <= 0.001)),
        "near_white_fraction": float(np.mean(luminance >= 0.99)),
        "any_channel_zero_fraction": float(np.mean(np.any(rgb == 0, axis=-1))),
        "all_channels_zero_fraction": float(np.mean(np.all(rgb == 0, axis=-1))),
        "any_channel_saturated_fraction": float(
            np.mean(np.any(rgb == 255, axis=-1))
        ),
        "all_channels_saturated_fraction": float(
            np.mean(np.all(rgb == 255, axis=-1))
        ),
    }
    return row, histogram.astype(np.int64)


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_per_image(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for field in ("width", "height", "pixel_count"):
                row[field] = int(raw[field])
            for field in NUMERIC_METRICS:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def run_analysis(
    items: Sequence[tuple[str, str, str, str, str, str]],
    per_image_path: Path,
    workers: int,
) -> tuple[list[Dict[str, Any]], Dict[str, np.ndarray]]:
    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = per_image_path.with_suffix(per_image_path.suffix + ".tmp")
    rows: list[Dict[str, Any]] = []
    histograms: Dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    )
    pool: Pool | None = None
    iterator: Iterable[tuple[Dict[str, Any], np.ndarray]]
    if workers > 0:
        pool = Pool(workers)
        iterator = pool.imap(analyze_one, items, chunksize=8)
    else:
        iterator = map(analyze_one, items)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS)
            writer.writeheader()
            for index, (row, histogram) in enumerate(iterator, start=1):
                writer.writerow(row)
                rows.append(row)
                histograms[f"dataset:{row['dataset']}"] += histogram
                histograms[
                    f"environment:{row['dataset']}/{row['environment']}"
                ] += histogram
                histograms[
                    f"shot:{row['dataset']}/{row['environment']}/{row['shot']}"
                ] += histogram
                if index % 1000 == 0 or index == len(items):
                    print(f"Luminance: {index}/{len(items)} images", flush=True)
        os.replace(temporary, per_image_path)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return rows, dict(histograms)


def summarize_values(values: np.ndarray) -> Dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def group_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    definitions = {
        "dataset": lambda row: (row["dataset"],),
        "environment": lambda row: (row["dataset"], row["environment"]),
        "shot": lambda row: (
            row["dataset"],
            row["environment"],
            row["shot"],
        ),
        "class": lambda row: (row["dataset"], row["class_id"]),
    }
    output: list[Dict[str, Any]] = []
    for group_type, key_function in definitions.items():
        groups: Dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key_function(row)].append(row)
        for key, members in sorted(groups.items()):
            result: Dict[str, Any] = {
                "group_type": group_type,
                "group_key": "/".join(key),
                "image_count": len(members),
            }
            for metric in NUMERIC_METRICS:
                values = np.asarray([float(row[metric]) for row in members])
                summary = summarize_values(values)
                result[f"{metric}_mean"] = summary["mean"]
                result[f"{metric}_median"] = summary["median"]
                result[f"{metric}_p05"] = summary["p05"]
                result[f"{metric}_p95"] = summary["p95"]
            output.append(result)
    return output


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def shot_stability(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["dataset"]),
                str(row["environment"]),
                str(row["class_id"]),
                str(row["image_id"]),
            )
        ].append(float(row["mean_luminance"]))
    output: list[Dict[str, Any]] = []
    for key, values_list in sorted(groups.items()):
        values = np.asarray(values_list, dtype=np.float64)
        if values.size != 5:
            raise ValueError(f"Expected five AE shots for {key}, found {values.size}")
        mean = float(np.mean(values))
        output.append(
            {
                "dataset": key[0],
                "environment": key[1],
                "class_id": key[2],
                "image_id": key[3],
                "shot_count": int(values.size),
                "mean_luminance": mean,
                "std_luminance": float(np.std(values)),
                "coefficient_of_variation": float(
                    np.std(values) / max(mean, 1e-6)
                ),
                "range_luminance": float(np.max(values) - np.min(values)),
                "range_ev": float(
                    math.log2((float(np.max(values)) + 1e-6) / (float(np.min(values)) + 1e-6))
                ),
            }
        )
    return output


def plot_mean_distributions(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, EXPECTED_COUNTS):
        values = np.asarray(
            [
                float(row["mean_luminance"])
                for row in rows
                if row["dataset"] == dataset
            ]
        )
        axis.hist(values, bins=80, range=(0.0, 1.0), color="#4472C4", alpha=0.85)
        axis.axvline(np.median(values), color="#C00000", linewidth=1.5, label="median")
        axis.set_title(dataset)
        axis.set_xlabel("Full-image linear Rec.709 mean luminance")
        axis.set_ylabel("Image count")
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_environment_shot_heatmap(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    datasets = list(EXPECTED_COUNTS)
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(12, 4.8), constrained_layout=True
    )
    for axis, dataset in zip(np.atleast_1d(axes), datasets):
        environments = sorted(
            {str(row["environment"]) for row in rows if row["dataset"] == dataset},
            key=lambda item: int(item[1:]),
        )
        shots = sorted(
            {str(row["shot"]) for row in rows if row["dataset"] == dataset},
            key=lambda item: int(item.split("_")[-1]),
        )
        matrix = np.zeros((len(environments), len(shots)), dtype=np.float64)
        for env_index, environment in enumerate(environments):
            for shot_index, shot in enumerate(shots):
                values = [
                    float(row["mean_luminance"])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["environment"] == environment
                    and row["shot"] == shot
                ]
                matrix[env_index, shot_index] = float(np.mean(values))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(shots)), shots, rotation=45, ha="right")
        axis.set_yticks(range(len(environments)), environments)
        axis.set_title(dataset)
        axis.set_xlabel("AE shot")
        axis.set_ylabel("Environment")
        fig.colorbar(image, ax=axis, label="Mean linear luminance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stability(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, EXPECTED_COUNTS):
        values = np.asarray(
            [
                float(row["range_ev"])
                for row in rows
                if row["dataset"] == dataset
            ]
        )
        upper = max(0.5, float(np.percentile(values, 99)))
        axis.hist(
            np.clip(values, 0.0, upper),
            bins=60,
            range=(0.0, upper),
            color="#70AD47",
            alpha=0.85,
        )
        axis.set_title(dataset)
        axis.set_xlabel("Five-shot mean-luminance range (EV; clipped at P99)")
        axis.set_ylabel("Reference/environment groups")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_exposure_examples(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    selected: list[Mapping[str, Any]] = []
    for dataset in EXPECTED_COUNTS:
        members = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: float(row["mean_luminance"]),
        )
        for quantile in (0.1, 0.5, 0.9):
            selected.append(members[round(quantile * (len(members) - 1))])
    variants: list[tuple[str, ExposureSpec | None]] = [
        ("Original", None),
        ("-2 EV", ExposureSpec("fixed_ev", -2.0)),
        ("+2 EV", ExposureSpec("fixed_ev", 2.0)),
        ("Target Y=0.10", ExposureSpec("target_mean_luminance", 0.10)),
    ]
    fig, axes = plt.subplots(
        len(selected),
        len(variants),
        figsize=(12, 3.0 * len(selected)),
        constrained_layout=True,
    )
    for row_index, row in enumerate(selected):
        with Image.open(str(row["path"])) as source:
            original = source.convert("RGB")
        for column_index, (label, spec) in enumerate(variants):
            if spec is None:
                shown = original
            else:
                shown, _metadata = apply_exposure(
                    original,
                    spec,
                    current_mean_luminance=float(row["mean_luminance"]),
                    original_metrics=row,
                )
            axis = axes[row_index, column_index]
            axis.imshow(shown)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(label)
            if column_index == 0:
                axis.set_ylabel(
                    f"{row['dataset']}\nY={float(row['mean_luminance']):.3f}",
                    rotation=90,
                )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    histograms: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    for dataset, expected in EXPECTED_COUNTS.items():
        members = [row for row in rows if row["dataset"] == dataset]
        datasets[dataset] = {
            "image_count": len(members),
            "expected_image_count": expected,
            "environment_count": len({row["environment"] for row in members}),
            "shot_count_per_environment": len({row["shot"] for row in members}),
            "mean_luminance": summarize_values(
                np.asarray([float(row["mean_luminance"]) for row in members])
            ),
            "log_mean_luminance": summarize_values(
                np.asarray([float(row["log_mean_luminance"]) for row in members])
            ),
            "near_black_fraction": summarize_values(
                np.asarray([float(row["near_black_fraction"]) for row in members])
            ),
            "near_white_fraction": summarize_values(
                np.asarray([float(row["near_white_fraction"]) for row in members])
            ),
            "any_channel_saturated_fraction": summarize_values(
                np.asarray(
                    [float(row["any_channel_saturated_fraction"]) for row in members]
                )
            ),
        }
    return {
        "definition": {
            "color_space": "inverse-sRGB linear RGB",
            "luminance": "Rec.709 Y = 0.2126 R + 0.7152 G + 0.0722 B",
            "scope": "full-resolution original AE JPEG pixels only; no resize or crop",
            "near_black": "Y <= 0.001",
            "near_white": "Y >= 0.99",
            "histogram_bins": HISTOGRAM_BINS,
            "histogram_range": [0.0, 1.0],
        },
        "total_images": len(rows),
        "datasets": datasets,
        "histograms": {
            key: value.tolist() for key, value in sorted(histograms.items())
        },
    }


def write_markdown_report(
    summary: Mapping[str, Any],
    grouped: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Full-image AE luminance analysis",
        "",
        (
            "All values are computed from original full-resolution AE JPEG pixels. "
            "No resize or crop is used for this analysis."
        ),
        "",
        "## Dataset summary",
        "",
        (
            "| Dataset | Images | Mean Y (mean) | Mean Y (median) | "
            "Mean Y P05–P95 | Near-white pixels (mean) | "
            "Any-channel saturation (mean) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in summary["datasets"].items():
        mean_y = row["mean_luminance"]
        lines.append(
            f"| {dataset} | {row['image_count']} | {mean_y['mean']:.6f} | "
            f"{mean_y['median']:.6f} | {mean_y['p05']:.6f}–{mean_y['p95']:.6f} | "
            f"{100.0 * row['near_white_fraction']['mean']:.3f}% | "
            f"{100.0 * row['any_channel_saturated_fraction']['mean']:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Environment summary",
            "",
            "| Dataset/environment | Images | Mean full-image Y | Median full-image Y |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in grouped:
        if row["group_type"] != "environment":
            continue
        lines.append(
            f"| {row['group_key']} | {row['image_count']} | "
            f"{float(row['mean_luminance_mean']):.6f} | "
            f"{float(row['mean_luminance_median']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Five-shot AE stability",
            "",
            (
                "Each stability group contains the five AE shots for one source "
                "image in one environment."
            ),
            "",
            "| Dataset | Groups | Median luminance range (EV) | P95 range (EV) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in EXPECTED_COUNTS:
        values = np.asarray(
            [
                float(row["range_ev"])
                for row in stability
                if row["dataset"] == dataset
            ]
        )
        lines.append(
            f"| {dataset} | {values.size} | {np.median(values):.6f} | "
            f"{np.percentile(values, 95):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `per_image.csv`: one record per original AE JPEG.",
            "- `group_summary.csv`: dataset/environment/shot/class aggregates.",
            "- `five_shot_stability.csv`: repeated-AE stability records.",
            "- `summary.json`: definitions, dataset summaries, and pixel histograms.",
            "- `mean_luminance_distribution.png`: per-image mean-Y distributions.",
            "- `environment_shot_heatmap.png`: environment × AE-shot means.",
            "- `five_shot_stability.png`: repeated-shot range distributions.",
            "- `exposure_examples.png`: deterministic full-image exposure examples.",
            "",
        ]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    roots = DatasetRoots(
        default_roots(workspace_root()).in_root,
        args.ae_es_root,
        args.ae_diverse_root,
    )
    items = discover_images(roots)
    per_image_path = args.output_dir / "per_image.csv"
    if per_image_path.exists() and not args.overwrite:
        print(f"Reuse existing per-image statistics: {per_image_path}")
        rows = read_per_image(per_image_path)
        if len(rows) != len(items):
            raise ValueError(
                f"Existing {per_image_path} has {len(rows)} rows; expected {len(items)}"
            )
        expected_paths = {item[-1] for item in items}
        actual_paths = {str(row["path"]) for row in rows}
        if actual_paths != expected_paths:
            raise ValueError("Existing per-image statistics do not match AE image paths")
        summary_path = args.output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError(
                f"{per_image_path} exists without {summary_path}; use --overwrite "
                "to rebuild the complete analysis"
            )
        with summary_path.open("r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
        histograms = {
            key: np.asarray(value, dtype=np.int64)
            for key, value in previous_summary.get("histograms", {}).items()
        }
        if not histograms:
            raise ValueError(
                f"{summary_path} has no histograms; use --overwrite to rebuild"
            )
    else:
        rows, histograms = run_analysis(items, per_image_path, args.workers)

    groups = group_rows(rows)
    stability = shot_stability(rows)
    write_csv(groups, args.output_dir / "group_summary.csv")
    write_csv(stability, args.output_dir / "five_shot_stability.csv")
    summary = build_summary(rows, histograms)
    summary["outputs"] = {
        "per_image_csv": str(per_image_path.resolve()),
        "group_summary_csv": str((args.output_dir / "group_summary.csv").resolve()),
        "five_shot_stability_csv": str(
            (args.output_dir / "five_shot_stability.csv").resolve()
        ),
    }
    atomic_json_dump(summary, args.output_dir / "summary.json")
    write_markdown_report(
        summary,
        groups,
        stability,
        args.output_dir / "report.md",
    )
    plot_mean_distributions(rows, args.output_dir / "mean_luminance_distribution.png")
    plot_environment_shot_heatmap(
        rows, args.output_dir / "environment_shot_heatmap.png"
    )
    plot_stability(stability, args.output_dir / "five_shot_stability.png")
    plot_exposure_examples(rows, args.output_dir / "exposure_examples.png")
    print(
        f"Wrote full-image luminance analysis for {len(rows)} images to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
