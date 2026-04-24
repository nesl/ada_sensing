from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
FEATURE_MODES = ("lightning_class", "lightning", "class")
NUM_CANDIDATES = 27


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test pred_best_index distribution from number-probe downstream results."
    )
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
        "--output_png",
        type=str,
        default=None,
        help="Defaults to {results_root}/number_probe_downstream_pred_distribution.png.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Defaults to {results_root}/number_probe_downstream_pred_distribution.json.",
    )
    return parser.parse_args()


def load_downstream_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing downstream result: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing or empty records in {path}")
    return records


def build_pred_hist(records: List[Dict[str, Any]]) -> List[int]:
    counts = Counter(int(record["pred_best_index"]) for record in records)
    return [int(counts.get(option_id, 0)) for option_id in range(NUM_CANDIDATES)]


def draw_histograms(output_png: Path, histograms: Dict[str, List[int]], label_kind: str) -> None:
    panel_width = 440
    panel_height = 280
    margin = 24
    header_h = 36
    footer_h = 30
    gap = 18
    width = (panel_width * len(FEATURE_MODES)) + (gap * (len(FEATURE_MODES) - 1)) + (margin * 2)
    height = panel_height + header_h + footer_h + (margin * 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    max_count = max(max(counts) for counts in histograms.values())
    if max_count <= 0:
        max_count = 1

    draw.text(
        (margin, margin),
        f"Number Probe Downstream Test Predicted Index Distribution ({label_kind})",
        fill="black",
    )

    for panel_idx, feature_mode in enumerate(FEATURE_MODES):
        left = margin + panel_idx * (panel_width + gap)
        top = margin + header_h
        right = left + panel_width
        bottom = top + panel_height
        counts = histograms[feature_mode]

        draw.rectangle([left, top, right, bottom], outline=(180, 180, 180), width=1)
        draw.text((left, top - 18), feature_mode, fill="black")

        inner_left = left + 16
        inner_top = top + 10
        inner_right = right - 8
        inner_bottom = bottom - 24
        inner_width = inner_right - inner_left
        inner_height = inner_bottom - inner_top
        draw.line([inner_left, inner_bottom, inner_right, inner_bottom], fill=(0, 0, 0), width=1)
        draw.line([inner_left, inner_top, inner_left, inner_bottom], fill=(0, 0, 0), width=1)

        bar_gap = 2
        bar_width = max(1, (inner_width - bar_gap * (NUM_CANDIDATES - 1)) // NUM_CANDIDATES)
        for option_id, count in enumerate(counts):
            x0 = inner_left + option_id * (bar_width + bar_gap)
            x1 = max(x0, x0 + bar_width - 1)
            bar_height = 0 if count == 0 else max(1, int((count / max_count) * (inner_height - 8)))
            y0 = max(inner_top, inner_bottom - bar_height)
            y1 = max(y0, inner_bottom - 1)
            draw.rectangle([x0, y0, x1, y1], fill=(76, 114, 176))
            if option_id < 10:
                label_x = x0 + max(0, (bar_width // 2) - 3)
                draw.text((label_x, inner_bottom + 2), str(option_id), fill=(80, 80, 80))

        draw.text((inner_left, inner_top), str(max_count), fill=(80, 80, 80))
        draw.text((inner_right - 18, inner_bottom + 2), "26", fill=(80, 80, 80))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def main() -> None:
    args = parse_args()
    results_root = (
        Path(args.results_root)
        if args.results_root is not None
        else ROOT / "policy_network" / "results_number_probe" / args.label_kind
    )
    output_png = (
        Path(args.output_png)
        if args.output_png is not None
        else results_root / "number_probe_downstream_pred_distribution.png"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else results_root / "number_probe_downstream_pred_distribution.json"
    )

    histograms: Dict[str, List[int]] = {}
    for feature_mode in FEATURE_MODES:
        downstream_json = results_root / feature_mode / "downstream_test_acc.json"
        records = load_downstream_records(downstream_json)
        histograms[feature_mode] = build_pred_hist(records)

    draw_histograms(output_png=output_png, histograms=histograms, label_kind=args.label_kind)

    summary = {
        feature_mode: {
            "pred_best_index_hist": histograms[feature_mode],
            "top10_pred_best_indices": sorted(
                (
                    (option_id, count)
                    for option_id, count in enumerate(histograms[feature_mode])
                    if count > 0
                ),
                key=lambda item: (-item[1], item[0]),
            )[:10],
        }
        for feature_mode in FEATURE_MODES
    }
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {output_png}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
