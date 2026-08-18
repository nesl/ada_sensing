from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lenz_reproduce_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .exposure import linear_luminance, srgb_u8_to_linear
from .protocol import workspace_root


GROUP_CLEAN = "clean_source"
GROUP_ORIGINAL = "original_diverse"
GROUP_Z001 = "dpi600_z001"
GROUP_Z002 = "dpi600_z002"
GROUP_ORDER = (GROUP_CLEAN, GROUP_ORIGINAL, GROUP_Z001, GROUP_Z002)
GROUP_LABELS = {
    GROUP_CLEAN: "Clean source images",
    GROUP_ORIGINAL: "Original Diverse AE",
    GROUP_Z001: "600-PPI z001 ROI",
    GROUP_Z002: "600-PPI z002 ROI",
}
GROUP_COLORS = {
    GROUP_CLEAN: "#262626",
    GROUP_ORIGINAL: "#C00000",
    GROUP_Z001: "#4472C4",
    GROUP_Z002: "#70AD47",
}

ORIGINAL_LIGHTING_IDS = ("l1", "l2", "l3", "l4", "l6", "l7")
ORIGINAL_SHOT_IDS = tuple(f"param_{index}" for index in range(1, 6))
DPI600_LIGHTING_IDS = ("b010", "b200", "b500", "b700", "b1000")
DPI600_SHOT_IDS = tuple(f"ae_{index:02d}" for index in range(1, 4))
DEFAULT_BIN_COUNT = 400
DEFAULT_EXPECTED_COMMON_SAMPLES = 5

PER_CAPTURE_FIELDS = (
    "group",
    "group_label",
    "sample_id",
    "lighting_id",
    "shot_id",
    "path",
    "width",
    "height",
    "pixel_count",
    "mean_luminance",
)


@dataclass(frozen=True)
class Capture:
    group: str
    sample_id: str
    lighting_id: str
    shot_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(
        description=(
            "Compare mean luminance for clean source images, original Diverse AE, "
            "and 600-PPI z001/z002 AE ROI captures."
        )
    )
    parser.add_argument(
        "--original-ae-root",
        type=Path,
        default=(
            workspace
            / "data"
            / "ImageNet-ES-Diverse"
            / "es-diverse-test"
            / "auto_exposure"
        ),
    )
    parser.add_argument(
        "--clean-source-root",
        type=Path,
        default=(
            workspace
            / "data"
            / "ImageNet-ES-Diverse"
            / "es-diverse-test"
            / "sampled_tin_no_resize2"
        ),
    )
    parser.add_argument(
        "--dpi600-roi-root",
        type=Path,
        default=workspace / "data" / "replication" / "dpi600_roi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace
        / "replicate_result"
        / "comparison"
        / "ae_luminance",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BIN_COUNT,
        help="Number of shared intervals used to evaluate the smoothed KDE curves.",
    )
    parser.add_argument(
        "--expected-common-samples",
        type=int,
        default=DEFAULT_EXPECTED_COMMON_SAMPLES,
    )
    return parser.parse_args()


def _sort_natural_identifier(value: str) -> tuple[str, int]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _capture_key(capture: Capture) -> tuple[str, str, str, str]:
    return (
        capture.group,
        capture.sample_id,
        capture.lighting_id,
        capture.shot_id,
    )


def _check_duplicate_captures(captures: Sequence[Capture]) -> None:
    seen: set[tuple[str, str, str, str]] = set()
    duplicates: list[tuple[str, str, str, str]] = []
    for capture in captures:
        key = _capture_key(capture)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate captures: {sorted(set(duplicates))[:5]}")


def _load_dpi600_ae_captures(roi_root: Path) -> list[Capture]:
    captures: list[Capture] = []
    for row in _read_jsonl(roi_root / "crop_manifest.jsonl"):
        if str(row.get("exposure_mode")) != "auto":
            continue
        zoom_id = str(row.get("zoom_id"))
        if zoom_id not in {"z001", "z002"}:
            raise ValueError(f"Unexpected zoom ID in DPI600 manifest: {zoom_id}")
        group = GROUP_Z001 if zoom_id == "z001" else GROUP_Z002
        shot_id = str(row.get("parameter_key", ""))
        if not shot_id:
            ae_shot = row.get("ae_shot")
            if not isinstance(ae_shot, int):
                raise ValueError(f"Missing AE shot for {row.get('capture_key')}")
            shot_id = f"ae_{ae_shot:02d}"
        relative_path = Path(str(row.get("cropped_image_path", "")))
        if not relative_path.as_posix() or relative_path.is_absolute():
            raise ValueError(
                f"Invalid cropped_image_path for {row.get('capture_key')}: {relative_path}"
            )
        image_path = roi_root / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing DPI600 ROI image: {image_path}")
        captures.append(
            Capture(
                group=group,
                sample_id=str(row.get("sample_id")),
                lighting_id=str(row.get("light_id")),
                shot_id=shot_id,
                path=image_path.resolve(),
            )
        )
    if not captures:
        raise ValueError(f"No auto-exposure rows found in {roi_root}")
    _check_duplicate_captures(captures)
    return captures


def _original_sample_ids(original_root: Path) -> set[str]:
    if not original_root.is_dir():
        raise FileNotFoundError(f"Original AE root does not exist: {original_root}")
    return {path.stem for path in original_root.glob("l*/param_*/*/*.JPEG")}


def _clean_sample_ids(clean_root: Path) -> set[str]:
    if not clean_root.is_dir():
        raise FileNotFoundError(f"Clean source root does not exist: {clean_root}")
    return {path.stem for path in clean_root.glob("*/*.JPEG")}


def _load_clean_captures(
    clean_root: Path, common_samples: set[str]
) -> list[Capture]:
    captures: list[Capture] = []
    for path in clean_root.glob("*/*.JPEG"):
        if path.stem not in common_samples:
            continue
        captures.append(
            Capture(
                group=GROUP_CLEAN,
                sample_id=path.stem,
                lighting_id="clean",
                shot_id="source",
                path=path.resolve(),
            )
        )
    if not captures:
        raise ValueError("No clean source images match the AE sample IDs")
    _check_duplicate_captures(captures)
    return captures


def _load_original_captures(
    original_root: Path, common_samples: set[str]
) -> list[Capture]:
    captures: list[Capture] = []
    for path in original_root.glob("l*/param_*/*/*.JPEG"):
        if path.stem not in common_samples:
            continue
        relative = path.relative_to(original_root)
        if len(relative.parts) != 4:
            raise ValueError(f"Unexpected original AE path layout: {path}")
        captures.append(
            Capture(
                group=GROUP_ORIGINAL,
                sample_id=path.stem,
                lighting_id=relative.parts[0],
                shot_id=relative.parts[1],
                path=path.resolve(),
            )
        )
    if not captures:
        raise ValueError("No original Diverse captures match the DPI600 sample IDs")
    _check_duplicate_captures(captures)
    return captures


def validate_complete_grid(
    captures: Sequence[Capture],
    group: str,
    sample_ids: Sequence[str],
    lighting_ids: Sequence[str],
    shot_ids: Sequence[str],
) -> None:
    members = [capture for capture in captures if capture.group == group]
    expected = {
        (sample_id, lighting_id, shot_id)
        for sample_id in sample_ids
        for lighting_id in lighting_ids
        for shot_id in shot_ids
    }
    actual = {
        (capture.sample_id, capture.lighting_id, capture.shot_id)
        for capture in members
    }
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"Incomplete {group} capture grid; missing={missing}, extra={extra}"
        )
    if len(members) != len(expected):
        raise ValueError(
            f"{group}: expected {len(expected)} unique captures, found {len(members)}"
        )


def discover_captures(
    original_root: Path,
    clean_source_root: Path,
    dpi600_roi_root: Path,
    expected_common_samples: int = DEFAULT_EXPECTED_COMMON_SAMPLES,
    original_lighting_ids: Sequence[str] = ORIGINAL_LIGHTING_IDS,
    original_shot_ids: Sequence[str] = ORIGINAL_SHOT_IDS,
    dpi600_lighting_ids: Sequence[str] = DPI600_LIGHTING_IDS,
    dpi600_shot_ids: Sequence[str] = DPI600_SHOT_IDS,
) -> tuple[list[Capture], list[str]]:
    dpi_captures = _load_dpi600_ae_captures(dpi600_roi_root)
    dpi_samples_by_group = {
        group: {
            capture.sample_id for capture in dpi_captures if capture.group == group
        }
        for group in (GROUP_Z001, GROUP_Z002)
    }
    common = (
        _original_sample_ids(original_root)
        & _clean_sample_ids(clean_source_root)
        & dpi_samples_by_group[GROUP_Z001]
        & dpi_samples_by_group[GROUP_Z002]
    )
    common_samples = sorted(common)
    if len(common_samples) != expected_common_samples:
        raise ValueError(
            f"Expected {expected_common_samples} common samples, found "
            f"{len(common_samples)}: {common_samples}"
        )
    selected_dpi = [
        capture for capture in dpi_captures if capture.sample_id in common
    ]
    clean_captures = _load_clean_captures(clean_source_root, common)
    original_captures = _load_original_captures(original_root, common)
    captures = clean_captures + original_captures + selected_dpi
    _check_duplicate_captures(captures)
    validate_complete_grid(
        captures,
        GROUP_CLEAN,
        common_samples,
        ("clean",),
        ("source",),
    )
    validate_complete_grid(
        captures,
        GROUP_ORIGINAL,
        common_samples,
        original_lighting_ids,
        original_shot_ids,
    )
    for group in (GROUP_Z001, GROUP_Z002):
        validate_complete_grid(
            captures,
            group,
            common_samples,
            dpi600_lighting_ids,
            dpi600_shot_ids,
        )
    captures.sort(
        key=lambda capture: (
            GROUP_ORDER.index(capture.group),
            capture.sample_id,
            _sort_natural_identifier(capture.lighting_id),
            _sort_natural_identifier(capture.shot_id),
        )
    )
    return captures, common_samples


def analyze_capture(capture: Capture) -> dict[str, Any]:
    with Image.open(capture.path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luminance = linear_luminance(srgb_u8_to_linear(rgb))
    return {
        "group": capture.group,
        "group_label": GROUP_LABELS[capture.group],
        "sample_id": capture.sample_id,
        "lighting_id": capture.lighting_id,
        "shot_id": capture.shot_id,
        "path": str(capture.path),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "pixel_count": int(rgb.shape[0] * rgb.shape[1]),
        "mean_luminance": float(luminance.mean(dtype=np.float64)),
    }


def _summarize_values(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        raise ValueError("Cannot summarize an empty luminance group")
    return {
        "capture_count": int(values.size),
        "mean_luminance": float(values.mean()),
        "median_luminance": float(np.median(values)),
        "p05_luminance": float(np.percentile(values, 5)),
        "p95_luminance": float(np.percentile(values, 95)),
        "min_luminance": float(values.min()),
        "max_luminance": float(values.max()),
    }


def build_group_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groupings = (
        ("overall", lambda row: str(row["group"])),
        (
            "lighting",
            lambda row: f"{row['group']}/{row['lighting_id']}",
        ),
        ("sample", lambda row: f"{row['group']}/{row['sample_id']}"),
    )
    for group_type, key_function in groupings:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(key_function(row), []).append(row)
        for group_key, members in grouped.items():
            group = str(members[0]["group"])
            values = np.asarray(
                [float(member["mean_luminance"]) for member in members],
                dtype=np.float64,
            )
            output.append(
                {
                    "group_type": group_type,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "group_key": group_key,
                    "sample_count": len({str(member["sample_id"]) for member in members}),
                    "lighting_count": len(
                        {str(member["lighting_id"]) for member in members}
                    ),
                    "shot_count": len({str(member["shot_id"]) for member in members}),
                    **_summarize_values(values),
                }
            )
    type_order = {"overall": 0, "lighting": 1, "sample": 2}
    output.sort(
        key=lambda row: (
            type_order[str(row["group_type"])],
            GROUP_ORDER.index(str(row["group"])),
            str(row["group_key"]),
        )
    )
    return output


def build_distribution(
    rows: Sequence[Mapping[str, Any]], bin_count: int
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    if bin_count <= 0:
        raise ValueError("Bin count must be positive")
    all_values = np.asarray(
        [float(row["mean_luminance"]) for row in rows], dtype=np.float64
    )
    if all_values.size == 0:
        raise ValueError("Cannot build a distribution without luminance rows")
    upper = min(1.0, max(float(all_values.max()) * 1.05, 1e-6))
    edges = np.linspace(0.0, upper, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    densities: dict[str, np.ndarray] = {}
    bandwidths: dict[str, float] = {}
    for group in GROUP_ORDER:
        values = np.asarray(
            [
                float(row["mean_luminance"])
                for row in rows
                if row["group"] == group
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError(f"No luminance values found for {group}")
        standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        q25, q75 = np.percentile(values, (25, 75))
        robust_scale = float((q75 - q25) / 1.349)
        positive_scales = [
            scale for scale in (standard_deviation, robust_scale) if scale > 0.0
        ]
        scale = min(positive_scales) if positive_scales else float(widths.mean())
        bandwidth = max(
            0.9 * scale * values.size ** (-0.2),
            2.0 * float(widths.mean()),
            1e-4,
        )
        offsets = (centers[:, None] - values[None, :]) / bandwidth
        reflected_offsets = (centers[:, None] + values[None, :]) / bandwidth
        normalizer = values.size * bandwidth * math.sqrt(2.0 * math.pi)
        density = (
            np.exp(-0.5 * offsets**2).sum(axis=1)
            + np.exp(-0.5 * reflected_offsets**2).sum(axis=1)
        ) / normalizer
        discrete_area = float(np.sum(density * widths))
        if not math.isfinite(discrete_area) or discrete_area <= 0.0:
            raise ValueError(f"Invalid KDE density for {group}: area={discrete_area}")
        density /= discrete_area
        densities[group] = density
        bandwidths[group] = bandwidth
    return edges, densities, bandwidths


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(
    rows: Sequence[Mapping[str, Any]], path: Path, fields: Sequence[str] | None = None
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(fields or rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def plot_distribution(
    rows: Sequence[Mapping[str, Any]],
    edges: np.ndarray,
    densities: Mapping[str, np.ndarray],
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for group in GROUP_ORDER:
        members = [row for row in rows if row["group"] == group]
        values = np.asarray(
            [float(row["mean_luminance"]) for row in members], dtype=np.float64
        )
        color = GROUP_COLORS[group]
        median = float(np.median(values))
        axis.plot(
            centers,
            densities[group],
            color=color,
            linewidth=2.0,
            label=f"{GROUP_LABELS[group]} (n={values.size}, median={median:.4f})",
        )
        axis.fill_between(
            centers,
            densities[group],
            color=color,
            alpha=0.06,
            linewidth=0.0,
        )
        axis.axvline(median, color=color, linewidth=1.2, linestyle="--", alpha=0.8)
    axis.set_xlim(float(edges[0]), float(edges[-1]))
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Per-image / ROI mean linear Rec.709 luminance")
    axis.set_ylabel("Normalized density")
    axis.set_title("Source and AE ROI luminance distributions")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)


def _overall_by_group(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["group"]): row
        for row in summaries
        if row["group_type"] == "overall"
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    common_samples: Sequence[str],
    edges: np.ndarray,
    densities: Mapping[str, np.ndarray],
    bandwidths: Mapping[str, float],
) -> dict[str, Any]:
    overall = _overall_by_group(summaries)
    original_median = float(overall[GROUP_ORIGINAL]["median_luminance"])
    gaps = {
        group: {
            "median_ratio_to_original": float(
                float(overall[group]["median_luminance"]) / original_median
            ),
            "median_ev_above_original": float(
                math.log2(float(overall[group]["median_luminance"]) / original_median)
            ),
        }
        for group in (GROUP_CLEAN, GROUP_Z001, GROUP_Z002)
    }
    return {
        "definition": {
            "color_space": "inverse-sRGB linear RGB",
            "luminance": "Rec.709 Y = 0.2126 R + 0.7152 G + 0.0722 B",
            "metric": "per-image or per-capture full-ROI mean luminance",
            "clean_source": "the complete clean source JPEG",
            "original_roi": "the complete original Diverse AE JPEG",
            "dpi600_roi": "the existing cropped DPI600 ROI JPEG",
            "resize_or_model_preprocessing": "none",
            "capture_weighting": (
                "one equal-weight observation per clean image or AE capture"
            ),
        },
        "common_sample_count": len(common_samples),
        "common_samples": list(common_samples),
        "groups": {group: dict(overall[group]) for group in GROUP_ORDER},
        "median_gaps": gaps,
        "distribution": {
            "method": "Gaussian kernel density estimate with reflection at Y=0",
            "bin_count": int(edges.size - 1),
            "bin_edges": edges.tolist(),
            "bandwidths": {group: float(bandwidths[group]) for group in GROUP_ORDER},
            "normalized_density": {
                group: densities[group].tolist() for group in GROUP_ORDER
            },
        },
    }


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    groups = summary["groups"]
    gaps = summary["median_gaps"]
    original_median = float(groups[GROUP_ORIGINAL]["median_luminance"])
    lines = [
        "# Four-group source and AE ROI luminance comparison",
        "",
        (
            "Each source image or auto-exposure capture contributes one equal-weight "
            "observation: the mean linear Rec.709 luminance over the complete image or "
            "target ROI. No resize or model preprocessing is applied."
        ),
        "",
        "## Overall distribution summary",
        "",
        "| Group | Captures | Samples | Lighting conditions | Mean Y | Median Y | P05-P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUP_ORDER:
        row = groups[group]
        lines.append(
            f"| {GROUP_LABELS[group]} | {row['capture_count']} | "
            f"{row['sample_count']} | {row['lighting_count']} | "
            f"{float(row['mean_luminance']):.6f} | "
            f"{float(row['median_luminance']):.6f} | "
            f"{float(row['p05_luminance']):.6f}-{float(row['p95_luminance']):.6f} |"
        )
    z001_ev = float(gaps[GROUP_Z001]["median_ev_above_original"])
    z002_ev = float(gaps[GROUP_Z002]["median_ev_above_original"])
    clean_ev = float(gaps[GROUP_CLEAN]["median_ev_above_original"])
    lines.extend(
        [
            "",
            "## Finding",
            "",
            (
                f"The original Diverse AE median ROI luminance is `{original_median:.6f}`. "
                f"The clean source median is "
                f"`{float(groups[GROUP_CLEAN]['median_luminance']):.6f}` "
                f"(`+{clean_ev:.3f} EV` above original Diverse AE). The 600-PPI z001 "
                f"and z002 medians are respectively "
                f"`{float(groups[GROUP_Z001]['median_luminance']):.6f}` and "
                f"`{float(groups[GROUP_Z002]['median_luminance']):.6f}`, or "
                f"`+{z001_ev:.3f} EV` and `+{z002_ev:.3f} EV` above the original median."
            ),
            "",
            (
                "This verifies that the original AE target regions are systematically "
                "darker for these five shared samples. It does not identify a specific "
                "aperture, shutter, or ISO cause because the original JPEGs do not retain "
                "the required exposure EXIF, and the two datasets' lighting labels are not "
                "directly calibrated to one another."
            ),
            "",
            "## Inputs and comparability",
            "",
            "- Clean source: one original JPEG for each of the five shared samples.",
            (
                "- Original Diverse: all six available lighting environments and five AE "
                "shots for each shared sample."
            ),
            (
                "- 600-PPI z001/z002: all five lighting intensities and three AE shots for "
                "each shared sample; manual-exposure captures are excluded."
            ),
            (
                "- The four Gaussian KDE curves use a shared evaluation grid and are "
                "normalized independently, so their areas are comparable despite "
                "different capture counts."
            ),
            "",
            "## Files",
            "",
            (
                "- `ae_roi_luminance_distribution.png`: four normalized, smoothed "
                "distribution curves."
            ),
            (
                "- `per_capture_luminance.csv`: one luminance record per clean image "
                "or AE capture."
            ),
            "- `group_summary.csv`: overall, lighting, and sample summaries.",
            "- `summary.json`: definitions, exact summary values, and plotted densities.",
            "",
        ]
    )
    _atomic_write_text("\n".join(lines), path)


def run_comparison(
    original_root: Path,
    clean_source_root: Path,
    dpi600_roi_root: Path,
    output_dir: Path,
    bin_count: int = DEFAULT_BIN_COUNT,
    expected_common_samples: int = DEFAULT_EXPECTED_COMMON_SAMPLES,
    original_lighting_ids: Sequence[str] = ORIGINAL_LIGHTING_IDS,
    original_shot_ids: Sequence[str] = ORIGINAL_SHOT_IDS,
    dpi600_lighting_ids: Sequence[str] = DPI600_LIGHTING_IDS,
    dpi600_shot_ids: Sequence[str] = DPI600_SHOT_IDS,
) -> dict[str, Any]:
    captures, common_samples = discover_captures(
        original_root=original_root,
        clean_source_root=clean_source_root,
        dpi600_roi_root=dpi600_roi_root,
        expected_common_samples=expected_common_samples,
        original_lighting_ids=original_lighting_ids,
        original_shot_ids=original_shot_ids,
        dpi600_lighting_ids=dpi600_lighting_ids,
        dpi600_shot_ids=dpi600_shot_ids,
    )
    rows = [analyze_capture(capture) for capture in captures]
    summaries = build_group_summaries(rows)
    edges, densities, bandwidths = build_distribution(rows, bin_count)
    summary = build_summary(
        rows,
        summaries,
        common_samples,
        edges,
        densities,
        bandwidths,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "per_capture_luminance.csv", PER_CAPTURE_FIELDS)
    _write_csv(summaries, output_dir / "group_summary.csv")
    _write_json(summary, output_dir / "summary.json")
    write_report(summary, output_dir / "report.md")
    plot_distribution(
        rows,
        edges,
        densities,
        output_dir / "ae_roi_luminance_distribution.png",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run_comparison(
        original_root=args.original_ae_root,
        clean_source_root=args.clean_source_root,
        dpi600_roi_root=args.dpi600_roi_root,
        output_dir=args.output_dir,
        bin_count=args.bins,
        expected_common_samples=args.expected_common_samples,
    )
    counts = {
        group: summary["groups"][group]["capture_count"] for group in GROUP_ORDER
    }
    print(
        f"Wrote AE ROI luminance comparison for {summary['common_sample_count']} "
        f"common samples to {args.output_dir.resolve()}: {counts}"
    )


if __name__ == "__main__":
    main()
