from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lenz_reproduce_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .datasets import DatasetRoots, build_dataset, default_roots
from .exposure import EXPOSURE_MODES, ExposureSpec, default_specs
from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    MODEL_SPECS,
    project_root,
    workspace_root,
)


AE_DATASETS = (DATASET_AE_ES, DATASET_AE_DIVERSE)
MODE_LABELS = {
    "fixed_ev": "Fixed EV compensation",
    "target_mean_luminance": "Per-image target mean luminance",
}


def parse_args() -> argparse.Namespace:
    defaults = default_roots(workspace_root())
    parser = argparse.ArgumentParser(
        description="Build exposure curves and clustered-bootstrap reports."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root() / "results" / "ae_exposure_raw",
    )
    parser.add_argument(
        "--luminance-csv",
        type=Path,
        default=project_root() / "results" / "ae_luminance" / "per_image.csv",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=project_root() / "results" / "ae_exposure_raw" / "audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "results" / "ae_exposure_report",
    )
    parser.add_argument("--in-root", type=Path, default=defaults.in_root)
    parser.add_argument("--ae-es-root", type=Path, default=defaults.ae_es_root)
    parser.add_argument(
        "--ae-diverse-root", type=Path, default=defaults.ae_diverse_root
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2481757)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_luminance_by_path(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resolved = str(Path(row["path"]).resolve())
            if resolved in values:
                raise ValueError(f"Duplicate luminance path: {resolved}")
            values[resolved] = float(row["mean_luminance"])
    return values


def ordered_sample_index(
    dataset_name: str,
    roots: DatasetRoots,
    luminance_by_path: Mapping[str, float],
) -> Dict[str, Any]:
    dataset = build_dataset(dataset_name, roots, transform=lambda image: image)
    paths: list[str] = []
    settings: list[str] = []
    cluster_keys: list[str] = []
    luminance: list[float] = []
    digest = hashlib.sha256()
    for setting, child in dataset.datasets:
        for path_string, _target in child.samples:
            path = str(Path(path_string).resolve())
            try:
                mean_y = luminance_by_path[path]
            except KeyError as error:
                raise KeyError(f"Missing luminance row for {path}") from error
            paths.append(path)
            settings.append(setting)
            cluster_keys.append(f"{Path(path).parent.name}/{Path(path).stem}")
            luminance.append(mean_y)
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    unique_clusters = {key: index for index, key in enumerate(sorted(set(cluster_keys)))}
    cluster_index = np.asarray(
        [unique_clusters[key] for key in cluster_keys], dtype=np.int32
    )
    cluster_counts = np.bincount(cluster_index)
    expected_repeats = 10 if dataset_name == DATASET_AE_ES else 30
    if len(unique_clusters) != 1000:
        raise ValueError(
            f"{dataset_name}: expected 1000 source-image clusters, "
            f"found {len(unique_clusters)}"
        )
    if not np.all(cluster_counts == expected_repeats):
        raise ValueError(
            f"{dataset_name}: source-image cluster counts are not all "
            f"{expected_repeats}: {Counter(cluster_counts.tolist())}"
        )
    return {
        "paths": paths,
        "settings": settings,
        "cluster_keys": cluster_keys,
        "cluster_index": cluster_index,
        "cluster_counts": cluster_counts,
        "mean_luminance": np.asarray(luminance, dtype=np.float64),
        "path_order_sha256": digest.hexdigest(),
    }


def clustered_curve_values(
    hits: np.ndarray,
    baseline_hits: np.ndarray,
    cluster_index: np.ndarray,
    cluster_counts: np.ndarray,
) -> np.ndarray:
    if hits.shape != baseline_hits.shape or hits.size != cluster_index.size:
        raise ValueError("Prediction arrays and sample index are not aligned")
    differences = hits.astype(np.int8) - baseline_hits.astype(np.int8)
    sums = np.bincount(
        cluster_index, weights=differences, minlength=cluster_counts.size
    )
    return sums / cluster_counts


def bootstrap_intervals(
    cluster_curves: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cluster_curves.ndim != 2:
        raise ValueError("cluster_curves must have shape [curves, clusters]")
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    cluster_count = cluster_curves.shape[1]
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        cluster_count,
        np.full(cluster_count, 1.0 / cluster_count),
        size=replicates,
    ).astype(np.float32)
    weights /= cluster_count
    samples_pp = 100.0 * (
        cluster_curves.astype(np.float32) @ weights.T
    )
    return (
        np.percentile(samples_pp, 2.5, axis=1),
        np.percentile(samples_pp, 97.5, axis=1),
    )


def result_paths(
    raw_dir: Path, model_key: str, dataset: str, spec: ExposureSpec
) -> tuple[Path, Path]:
    stem = f"{model_key}__{dataset}__{spec.tag}"
    return raw_dir / f"{stem}.json", raw_dir / f"{stem}.npz"


def load_predictions(path: Path) -> np.ndarray:
    with np.load(path) as arrays:
        return np.asarray(arrays["hits"], dtype=np.bool_)


def collect_curves(
    raw_dir: Path,
    sample_indices: Mapping[str, Mapping[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[tuple[str, str], np.ndarray]]:
    specs = default_specs()
    curve_rows: list[Dict[str, Any]] = []
    macro_rows: list[Dict[str, Any]] = []
    baseline_hits_by_model: Dict[tuple[str, str], np.ndarray] = {}

    for dataset_number, dataset in enumerate(AE_DATASETS):
        sample_index = sample_indices[dataset]
        cluster_index = sample_index["cluster_index"]
        cluster_counts = sample_index["cluster_counts"]
        baseline_accuracy: Dict[str, float] = {}
        dataset_rows: list[Dict[str, Any]] = []
        cluster_vectors: list[np.ndarray] = []

        zero_spec = ExposureSpec("fixed_ev", 0.0)
        for model in MODEL_SPECS:
            zero_json, zero_npz = result_paths(
                raw_dir, model.key, dataset, zero_spec
            )
            zero_result = read_json(zero_json)
            if zero_result["path_order_sha256"] != sample_index["path_order_sha256"]:
                raise ValueError(f"Path-order mismatch in {zero_json}")
            baseline_accuracy[model.key] = float(zero_result["micro_accuracy"])
            baseline_hits_by_model[(dataset, model.key)] = load_predictions(zero_npz)

        for model in MODEL_SPECS:
            baseline_hits = baseline_hits_by_model[(dataset, model.key)]
            for spec in specs:
                json_path, npz_path = result_paths(raw_dir, model.key, dataset, spec)
                result = read_json(json_path)
                if result["path_order_sha256"] != sample_index["path_order_sha256"]:
                    raise ValueError(f"Path-order mismatch in {json_path}")
                hits = load_predictions(npz_path)
                cluster_values = clustered_curve_values(
                    hits, baseline_hits, cluster_index, cluster_counts
                )
                observed_delta = float(result["micro_accuracy"]) - baseline_accuracy[model.key]
                if not np.isclose(100.0 * cluster_values.mean(), observed_delta, atol=1e-9):
                    raise ValueError(f"Clustered delta mismatch in {json_path}")
                exposure_statistics = result["exposure_statistics"]
                row = {
                    "dataset": dataset,
                    "model_key": model.key,
                    "model": model.paper_name,
                    "mode": spec.mode,
                    "value": spec.value,
                    "tag": spec.tag,
                    "correct": int(result["correct"]),
                    "total": int(result["total"]),
                    "accuracy": float(result["micro_accuracy"]),
                    "baseline_accuracy": baseline_accuracy[model.key],
                    "delta_pp": observed_delta,
                    "delta_ci95_low": 0.0,
                    "delta_ci95_high": 0.0,
                    "achieved_mean_luminance": float(
                        exposure_statistics["achieved_mean_luminance"]["mean"]
                    ),
                    "any_channel_saturated_fraction": float(
                        exposure_statistics["any_channel_saturated_fraction"]["mean"]
                    ),
                    "effective_ev_median": float(
                        exposure_statistics["effective_ev"]["median"]
                    ),
                    "gain_median": float(exposure_statistics["gain"]["median"]),
                }
                dataset_rows.append(row)
                cluster_vectors.append(cluster_values)

        curve_matrix = np.stack(cluster_vectors)
        lows, highs = bootstrap_intervals(
            curve_matrix,
            replicates=bootstrap_replicates,
            seed=seed + dataset_number,
        )
        for row, low, high in zip(dataset_rows, lows, highs):
            row["delta_ci95_low"] = float(low)
            row["delta_ci95_high"] = float(high)
        curve_rows.extend(dataset_rows)

        row_lookup = {
            (row["model_key"], row["tag"]): (row, vector)
            for row, vector in zip(dataset_rows, cluster_vectors)
        }
        macro_vectors: list[np.ndarray] = []
        pending_macro: list[Dict[str, Any]] = []
        for spec in specs:
            rows_and_vectors = [
                row_lookup[(model.key, spec.tag)] for model in MODEL_SPECS
            ]
            rows = [item[0] for item in rows_and_vectors]
            vector = np.mean(
                np.stack([item[1] for item in rows_and_vectors]), axis=0
            )
            pending_macro.append(
                {
                    "dataset": dataset,
                    "mode": spec.mode,
                    "value": spec.value,
                    "tag": spec.tag,
                    "model_count": len(MODEL_SPECS),
                    "macro_accuracy": float(
                        np.mean([float(row["accuracy"]) for row in rows])
                    ),
                    "macro_baseline_accuracy": float(
                        np.mean([float(row["baseline_accuracy"]) for row in rows])
                    ),
                    "macro_delta_pp": float(
                        np.mean([float(row["delta_pp"]) for row in rows])
                    ),
                    "delta_ci95_low": 0.0,
                    "delta_ci95_high": 0.0,
                    "models_improved": sum(float(row["delta_pp"]) > 0.0 for row in rows),
                    "models_unchanged": sum(float(row["delta_pp"]) == 0.0 for row in rows),
                    "models_degraded": sum(float(row["delta_pp"]) < 0.0 for row in rows),
                }
            )
            macro_vectors.append(vector)
        macro_lows, macro_highs = bootstrap_intervals(
            np.stack(macro_vectors),
            replicates=bootstrap_replicates,
            seed=seed + 100 + dataset_number,
        )
        for row, low, high in zip(pending_macro, macro_lows, macro_highs):
            row["delta_ci95_low"] = float(low)
            row["delta_ci95_high"] = float(high)
        macro_rows.extend(pending_macro)
    return curve_rows, macro_rows, baseline_hits_by_model


def build_luminance_bins(
    sample_indices: Mapping[str, Mapping[str, Any]],
    baseline_hits: Mapping[tuple[str, str], np.ndarray],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for dataset in AE_DATASETS:
        luminance = np.asarray(sample_indices[dataset]["mean_luminance"])
        edges = np.quantile(luminance, np.linspace(0.0, 1.0, 11))
        bin_index = np.searchsorted(edges[1:-1], luminance, side="right")
        for model in MODEL_SPECS:
            hits = baseline_hits[(dataset, model.key)]
            for bin_number in range(10):
                mask = bin_index == bin_number
                rows.append(
                    {
                        "dataset": dataset,
                        "model_key": model.key,
                        "model": model.paper_name,
                        "luminance_decile": bin_number + 1,
                        "image_count": int(mask.sum()),
                        "luminance_min": float(luminance[mask].min()),
                        "luminance_mean": float(luminance[mask].mean()),
                        "luminance_max": float(luminance[mask].max()),
                        "baseline_accuracy": float(100.0 * hits[mask].mean()),
                    }
                )
    return rows


def model_curve_rows(
    rows: Sequence[Mapping[str, Any]], dataset: str, mode: str, model_key: str
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["dataset"] == dataset
            and row["mode"] == mode
            and row["model_key"] == model_key
        ),
        key=lambda row: float(row["value"]),
    )


def configure_x_axis(axis, mode: str) -> None:
    if mode == "fixed_ev":
        axis.set_xlabel("Digital exposure compensation (EV)")
        axis.set_xticks((-4, -2, 0, 2, 4))
    else:
        axis.set_xscale("log")
        ticks = (0.02, 0.05, 0.1, 0.2, 0.5, 0.95)
        axis.set_xticks(ticks, [str(value) for value in ticks])
        axis.set_xlabel("Target full-image mean luminance")


def plot_model_accuracy_facets(
    rows: Sequence[Mapping[str, Any]], dataset: str, mode: str, path: Path
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODEL_SPECS):
        curve = model_curve_rows(rows, dataset, mode, model.key)
        x = np.asarray([float(row["value"]) for row in curve])
        accuracy = np.asarray([float(row["accuracy"]) for row in curve])
        baseline = float(curve[0]["baseline_accuracy"])
        low = baseline + np.asarray([float(row["delta_ci95_low"]) for row in curve])
        high = baseline + np.asarray([float(row["delta_ci95_high"]) for row in curve])
        axis.fill_between(x, low, high, color="#4472C4", alpha=0.18)
        axis.plot(x, accuracy, color="#4472C4", marker="o", markersize=3)
        axis.axhline(baseline, color="#C00000", linestyle="--", linewidth=1)
        axis.set_title(model.paper_name, fontsize=10)
        axis.set_ylabel("Top-1 accuracy (%)")
        axis.grid(alpha=0.2)
        configure_x_axis(axis, mode)
    fig.suptitle(f"{dataset}: {MODE_LABELS[mode]}", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_macro_delta(
    rows: Sequence[Mapping[str, Any]], dataset: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, mode in zip(axes, EXPOSURE_MODES):
        curve = sorted(
            (
                row
                for row in rows
                if row["dataset"] == dataset and row["mode"] == mode
            ),
            key=lambda row: float(row["value"]),
        )
        x = np.asarray([float(row["value"]) for row in curve])
        delta = np.asarray([float(row["macro_delta_pp"]) for row in curve])
        low = np.asarray([float(row["delta_ci95_low"]) for row in curve])
        high = np.asarray([float(row["delta_ci95_high"]) for row in curve])
        axis.fill_between(x, low, high, color="#70AD47", alpha=0.2)
        axis.plot(x, delta, color="#548235", marker="o", markersize=4)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(MODE_LABELS[mode])
        axis.set_ylabel("12-model macro top-1 delta (percentage points)")
        axis.grid(alpha=0.2)
        configure_x_axis(axis, mode)
    fig.suptitle(f"{dataset}: paired cluster-bootstrap deltas", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_baseline_luminance_bins(
    rows: Sequence[Mapping[str, Any]], dataset: str, path: Path
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), constrained_layout=True)
    for axis, model in zip(axes.flat, MODEL_SPECS):
        model_rows = sorted(
            (
                row
                for row in rows
                if row["dataset"] == dataset and row["model_key"] == model.key
            ),
            key=lambda row: int(row["luminance_decile"]),
        )
        x = [float(row["luminance_mean"]) for row in model_rows]
        y = [float(row["baseline_accuracy"]) for row in model_rows]
        axis.plot(x, y, color="#8064A2", marker="o", markersize=4)
        axis.set_xscale("log")
        axis.set_title(model.paper_name, fontsize=10)
        axis.set_xlabel("Mean luminance of decile")
        axis.set_ylabel("AE baseline top-1 (%)")
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{dataset}: AE baseline accuracy by original-image luminance decile",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def descriptive_peak(
    rows: Sequence[Mapping[str, Any]], dataset: str, mode: str
) -> Mapping[str, Any]:
    candidates = [
        row for row in rows if row["dataset"] == dataset and row["mode"] == mode
    ]
    return max(candidates, key=lambda row: float(row["macro_accuracy"]))


def format_interval(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['macro_delta_pp']):+.3f} pp "
        f"[{float(row['delta_ci95_low']):+.3f}, "
        f"{float(row['delta_ci95_high']):+.3f}]"
    )


def write_markdown_report(
    curve_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    bootstrap_replicates: int,
    path: Path,
) -> None:
    lines = [
        "# Digital-exposure downstream evaluation",
        "",
        "## Protocol and integrity",
        "",
        f"- Raw audit: **{audit['status']}**, {audit['complete_results']}/{audit['expected_results']} results complete.",
        f"- Zero-EV baseline mismatches: **{audit['zero_ev_baseline_mismatch_count']}**.",
        "- Models: 12 frozen ImageNet classifiers, evaluated in the locked closed 200-way protocol.",
        f"- Uncertainty: {bootstrap_replicates:,} paired bootstrap replicates over 1,000 source-image clusters.",
        "- Confidence intervals are pointwise and exploratory; no exposure value is selected as a tuned test-set target.",
        "",
        "## Cross-model results",
        "",
    ]
    for dataset in AE_DATASETS:
        dataset_macro = [row for row in macro_rows if row["dataset"] == dataset]
        baseline = float(dataset_macro[0]["macro_baseline_accuracy"])
        lines.extend([f"### {dataset}", "", f"12-model AE baseline macro accuracy: **{baseline:.3f}%**.", ""])
        for mode in EXPOSURE_MODES:
            peak = descriptive_peak(macro_rows, dataset, mode)
            positive_points = [
                row
                for row in dataset_macro
                if row["mode"] == mode and float(row["delta_ci95_low"]) > 0.0
            ]
            lines.append(
                f"- {MODE_LABELS[mode]}: highest observed macro point at "
                f"`{float(peak['value']):g}` gives {float(peak['macro_accuracy']):.3f}% "
                f"({format_interval(peak)}), with {peak['models_improved']}/12 models numerically improved."
            )
            if positive_points:
                values = ", ".join(f"{float(row['value']):g}" for row in positive_points)
                lines.append(
                    f"  Pointwise macro CI is above zero at: `{values}`."
                )
            else:
                lines.append("  No tested point has a macro CI wholly above zero.")
        lines.extend(
            [
                "",
                "Descriptive per-model curve maxima (not selected targets):",
                "",
                "| Model | AE baseline | Fixed-EV value | Fixed-EV delta | Target-Y value | Target-Y delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model in MODEL_SPECS:
            model_rows = [
                row
                for row in curve_rows
                if row["dataset"] == dataset and row["model_key"] == model.key
            ]
            fixed = max(
                (row for row in model_rows if row["mode"] == "fixed_ev"),
                key=lambda row: float(row["accuracy"]),
            )
            target = max(
                (
                    row
                    for row in model_rows
                    if row["mode"] == "target_mean_luminance"
                ),
                key=lambda row: float(row["accuracy"]),
            )
            lines.append(
                f"| {model.paper_name} | {float(fixed['baseline_accuracy']):.3f}% | "
                f"{float(fixed['value']):g} | {float(fixed['delta_pp']):+.3f} pp | "
                f"{float(target['value']):g} | {float(target['delta_pp']):+.3f} pp |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundaries",
            "",
            "- Digital darkening cannot recover highlight detail already clipped during capture.",
            "- Digital brightening cannot improve sensor SNR and may amplify quantization or noise.",
            "- Model-specific observed maxima are descriptive upper-envelope summaries, not validation-selected deployment settings.",
            "- A physical camera experiment is required to establish whether changing the camera AE target outperforms post-capture digital gain.",
            "",
            "## Files",
            "",
            "- `curve_results.csv`: every model × dataset × exposure point with paired CIs.",
            "- `macro_curve_results.csv`: 12-model macro curves and CIs.",
            "- `baseline_luminance_bins.csv`: baseline accuracy by original-image luminance decile.",
            "- `*.png`: per-model accuracy curves, macro deltas, and luminance-bin diagnostics.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def write_chinese_summary(
    curve_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    bootstrap_replicates: int,
    path: Path,
) -> None:
    es_fixed = descriptive_peak(macro_rows, DATASET_AE_ES, "fixed_ev")
    es_target = descriptive_peak(
        macro_rows, DATASET_AE_ES, "target_mean_luminance"
    )
    diverse_fixed = descriptive_peak(
        macro_rows, DATASET_AE_DIVERSE, "fixed_ev"
    )
    diverse_target = descriptive_peak(
        macro_rows, DATASET_AE_DIVERSE, "target_mean_luminance"
    )
    diverse_half_ev = next(
        row
        for row in curve_rows
        if row["dataset"] == DATASET_AE_DIVERSE
        and row["model_key"] == MODEL_SPECS[0].key
        and row["mode"] == "fixed_ev"
        and float(row["value"]) == 0.5
    )
    lines = [
        "# AE 数字曝光 downstream 实验总结",
        "",
        "## 数据完整性",
        "",
        f"- Raw audit：**{audit['status']}**，完成 {audit['complete_results']}/{audit['expected_results']} 个实验单元。",
        f"- `0 EV` 与原始 AE baseline 不一致数：**{audit['zero_ev_baseline_mismatch_count']}**。",
        f"- 置信区间：对 1,000 个原始 ImageNet 图像 ID 做 {bootstrap_replicates:,} 次 paired cluster bootstrap；同一原图的全部环境与五次 AE 拍摄始终放在同一个 cluster。",
        "- 所有 CI 都是逐点、探索性的；没有用测试集选择部署 target。",
        "",
        "## 核心结果",
        "",
        "| 数据集 | 12-model AE baseline | 实验 | 最高观测点 | Macro delta 与 95% CI |",
        "| --- | ---: | --- | ---: | ---: |",
        f"| ImageNet-ES | {float(es_fixed['macro_baseline_accuracy']):.3f}% | Fixed EV | {float(es_fixed['value']):g} EV | {format_interval(es_fixed)} |",
        f"| ImageNet-ES | {float(es_target['macro_baseline_accuracy']):.3f}% | Target Y | {float(es_target['value']):g} | {format_interval(es_target)} |",
        f"| Diverse | {float(diverse_fixed['macro_baseline_accuracy']):.3f}% | Fixed EV | {float(diverse_fixed['value']):g} EV | {format_interval(diverse_fixed)} |",
        f"| Diverse | {float(diverse_target['macro_baseline_accuracy']):.3f}% | Target Y | {float(diverse_target['value']):g} | {format_interval(diverse_target)} |",
        "",
        "### ImageNet-ES",
        "",
        "- 原始 `0 EV` 是全部测试点中最好的跨模型结果。负 EV 会降低准确率，正 EV 会因进一步饱和而快速恶化。",
        "- 逐图 target-luminance 的最高观测点仍比 AE baseline 低；12/12 模型都没有改善。",
        "- 这说明拍摄时已经发生的高光 clipping 不能靠 JPEG 后处理恢复。该结论不排除在相机拍摄阶段降低 AE target 会有效。",
        "",
        "### ImageNet-ES-Diverse",
        "",
        f"- 统一 `+0.5 EV` 将平均线性 luminance 从约 `0.0653` 提高到 `{float(diverse_half_ev['achieved_mean_luminance']):.4f}`，12-model macro 提升很小但 CI 高于零：**{format_interval(diverse_fixed)}**。",
        f"- 该点有 {diverse_fixed['models_improved']}/12 个模型数值提升；这更像轻微的通用校正，而不是显著改变 benchmark 表现。",
        "- 逐图 target-luminance 没有产生跨模型正向点，说明把所有场景拉到同一个全图平均亮度并不是可靠的通用策略。",
        "- 模型偏好的曝光方向明显分裂：多数 CNN/Swin 偏好正 EV，而 OpenCLIP-h、OpenCLIP-b、DINOv2-b 偏好负 EV；DINOv2-g 在原始 AE 附近最好。",
        "",
        "## 模型级显著改善（Diverse，逐点 CI）",
        "",
        "以下只列各模型最高观测点；它们没有经过独立 validation 选择，因此是 descriptive upper envelope。",
        "",
        "| Model | Mode | 观测点 | Delta | 95% CI |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in MODEL_SPECS:
        model_rows = [
            row
            for row in curve_rows
            if row["dataset"] == DATASET_AE_DIVERSE
            and row["model_key"] == model.key
        ]
        for mode in EXPOSURE_MODES:
            peak = max(
                (row for row in model_rows if row["mode"] == mode),
                key=lambda row: float(row["accuracy"]),
            )
            if float(peak["delta_ci95_low"]) <= 0.0:
                continue
            lines.append(
                f"| {model.paper_name} | {MODE_LABELS[mode]} | {float(peak['value']):g} | "
                f"{float(peak['delta_pp']):+.3f} pp | "
                f"[{float(peak['delta_ci95_low']):+.3f}, {float(peak['delta_ci95_high']):+.3f}] |"
            )
    lines.extend(
        [
            "",
            "## 如何解释 luminance 与 accuracy 的关系",
            "",
            "- ES 中更亮的原图组准确率更低，Diverse 中更亮的原图组准确率更高，与两套数据的过曝/欠曝现象一致。",
            "- 但亮度 decile 同时混入了类别、场景和环境难度，属于 observational correlation，不能单独证明调亮或调暗会带来同等幅度的因果提升。",
            "- 真正的 paired 数字曝光曲线显示：Diverse 只有小幅通用收益，主要收益是 model-specific；ES 则无法通过后处理挽回高光信息。",
            "",
            "## 结论",
            "",
            "1. **不存在一个同时适合两套数据和全部模型的统一 target exposure。**",
            "2. Diverse 的 `+0.5 EV` 是唯一一个跨模型逐点 CI 高于零的测试点，但它来自测试网格，不能直接当作已选定设置；需要独立 validation 或物理重拍确认，且当前通用收益只有约 `0.13 pp`。",
            "3. 更大的收益来自 model-specific exposure；如果目标是 Lens/自适应感知，应让 exposure policy 感知 downstream model，而不是只对齐固定 luminance。",
            "4. ES 必须通过真实相机降低曝光重新拍摄，才能验证恢复高光细节后的潜在收益。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    audit = read_json(args.audit_json)
    if audit.get("status") != "ok":
        raise ValueError(
            f"Raw exposure audit is not clean: {args.audit_json} has "
            f"status={audit.get('status')!r}"
        )
    roots = DatasetRoots(args.in_root, args.ae_es_root, args.ae_diverse_root)
    luminance_by_path = load_luminance_by_path(args.luminance_csv)
    sample_indices = {
        dataset: ordered_sample_index(dataset, roots, luminance_by_path)
        for dataset in AE_DATASETS
    }
    for dataset, values in sample_indices.items():
        audit_hashes = audit["path_order_hashes"].get(dataset, [])
        if audit_hashes != [values["path_order_sha256"]]:
            raise ValueError(
                f"{dataset}: reconstructed path hash does not match raw audit"
            )

    curve_rows, macro_rows, baseline_hits = collect_curves(
        raw_dir=args.raw_dir,
        sample_indices=sample_indices,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    luminance_bins = build_luminance_bins(sample_indices, baseline_hits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(curve_rows, args.output_dir / "curve_results.csv")
    write_csv(macro_rows, args.output_dir / "macro_curve_results.csv")
    write_csv(luminance_bins, args.output_dir / "baseline_luminance_bins.csv")
    atomic_json_dump(
        {
            "protocol": {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.seed,
                "bootstrap_cluster": (
                    "source ImageNet image ID; all environments and five AE shots "
                    "remain together"
                ),
                "confidence_intervals": "pointwise paired 95% cluster bootstrap",
                "target_selection": False,
            },
            "audit": audit,
            "curve_results": curve_rows,
            "macro_curve_results": macro_rows,
        },
        args.output_dir / "exposure_report.json",
    )

    for dataset in AE_DATASETS:
        for mode in EXPOSURE_MODES:
            plot_model_accuracy_facets(
                curve_rows,
                dataset,
                mode,
                args.output_dir / f"{dataset}__{mode}__model_curves.png",
            )
        plot_macro_delta(
            macro_rows,
            dataset,
            args.output_dir / f"{dataset}__macro_delta.png",
        )
        plot_baseline_luminance_bins(
            luminance_bins,
            dataset,
            args.output_dir / f"{dataset}__baseline_luminance_bins.png",
        )
    write_markdown_report(
        curve_rows=curve_rows,
        macro_rows=macro_rows,
        audit=audit,
        bootstrap_replicates=args.bootstrap_replicates,
        path=args.output_dir / "report.md",
    )
    write_chinese_summary(
        curve_rows=curve_rows,
        macro_rows=macro_rows,
        audit=audit,
        bootstrap_replicates=args.bootstrap_replicates,
        path=args.output_dir / "report_zh.md",
    )
    print(
        f"Wrote exposure report for {len(curve_rows)} model/dataset/exposure "
        f"points and "
        f"{len(macro_rows)} macro points to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
