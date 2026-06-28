from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select three fixed policy-input probe options by maximizing "
            "brightness-histogram diversity. The selection is unsupervised: it "
            "uses image brightness only, not policy labels or downstream correctness."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--data_json",
        type=str,
        nargs="+",
        required=True,
        help="One or more policy label json files whose sample_ids define the selection set.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument(
        "--resize",
        type=int,
        default=128,
        help="Resize shorter analysis dimension for faster histograms. Use 0 to disable.",
    )
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="Seeds used to produce random3 control probe sets.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_selection_sample_ids(paths: Iterable[str]) -> List[str]:
    sample_ids: List[str] = []
    seen = set()
    for path in paths:
        for item in load_json(path):
            sample_id = str(item["sample_id"])
            if sample_id not in seen:
                seen.add(sample_id)
                sample_ids.append(sample_id)
    return sample_ids


def build_manifest_index(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    manifest_items = load_json(manifest_path)
    return {str(item["id"]): item for item in manifest_items}


def get_option_name_map(manifest_items: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    option_name_map: Dict[int, str] = {}
    for item in manifest_items:
        for candidate in item["candidates"]:
            option_id = int(candidate["option_id"])
            option_name = str(candidate.get("meta", {}).get("option_name", ""))
            previous = option_name_map.get(option_id)
            if previous is not None and previous != option_name:
                raise ValueError(
                    f"option_id={option_id} maps to both {previous} and {option_name}"
                )
            option_name_map[option_id] = option_name
    return option_name_map


def brightness_histogram(path: str, bins: int, resize: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize > 0:
            img.thumbnail((resize, resize))
        arr = np.asarray(img, dtype=np.float32)
    brightness = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    hist, _ = np.histogram(brightness, bins=bins, range=(0.0, 255.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError(f"Empty brightness histogram for {path}")
    return hist / total


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def compute_option_histograms(
    manifest_index: Dict[str, Dict[str, Any]],
    sample_ids: List[str],
    bins: int,
    resize: int,
) -> Dict[int, np.ndarray]:
    hist_sums: Dict[int, np.ndarray] = {}
    hist_counts: Dict[int, int] = {}

    for sample_id in tqdm(sample_ids, desc="Brightness histograms"):
        if sample_id not in manifest_index:
            raise KeyError(f"sample_id={sample_id} not found in manifest.")
        for candidate in manifest_index[sample_id]["candidates"]:
            option_id = int(candidate["option_id"])
            hist = brightness_histogram(candidate["path"], bins=bins, resize=resize)
            if option_id not in hist_sums:
                hist_sums[option_id] = np.zeros_like(hist)
                hist_counts[option_id] = 0
            hist_sums[option_id] += hist
            hist_counts[option_id] += 1

    return {
        option_id: hist_sums[option_id] / hist_counts[option_id]
        for option_id in sorted(hist_sums)
    }


def build_pairwise_distances(option_hists: Dict[int, np.ndarray]) -> Dict[int, Dict[int, float]]:
    option_ids = sorted(option_hists)
    distances: Dict[int, Dict[int, float]] = {option_id: {} for option_id in option_ids}
    for i, option_i in enumerate(option_ids):
        for option_j in option_ids[i:]:
            dist = js_distance(option_hists[option_i], option_hists[option_j])
            distances[option_i][option_j] = dist
            distances[option_j][option_i] = dist
    return distances


def select_farthest_three(option_hists: Dict[int, np.ndarray]) -> List[int]:
    option_ids = sorted(option_hists)
    global_mean = np.mean(np.stack([option_hists[option_id] for option_id in option_ids]), axis=0)
    first = max(option_ids, key=lambda option_id: js_distance(option_hists[option_id], global_mean))
    selected = [first]

    while len(selected) < 3:
        remaining = [option_id for option_id in option_ids if option_id not in selected]
        next_option = max(
            remaining,
            key=lambda option_id: min(
                js_distance(option_hists[option_id], option_hists[chosen])
                for chosen in selected
            ),
        )
        selected.append(next_option)

    return selected


def summarize_selection(
    option_ids: List[int],
    option_name_map: Dict[int, str],
    pairwise_distances: Dict[int, Dict[int, float]],
) -> Dict[str, Any]:
    return {
        "selected_option_ids": option_ids,
        "selected_option_names": [option_name_map.get(option_id, "") for option_id in option_ids],
        "selected_pairwise_js_distances": {
            f"{option_i}-{option_j}": pairwise_distances[option_i][option_j]
            for idx, option_i in enumerate(option_ids)
            for option_j in option_ids[idx + 1 :]
        },
    }


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be >= 2.")

    manifest_index = build_manifest_index(args.manifest)
    option_name_map = get_option_name_map(manifest_index.values())
    sample_ids = load_selection_sample_ids(args.data_json)

    option_hists = compute_option_histograms(
        manifest_index=manifest_index,
        sample_ids=sample_ids,
        bins=args.bins,
        resize=args.resize,
    )
    pairwise_distances = build_pairwise_distances(option_hists)
    hist3_ids = select_farthest_three(option_hists)

    random3 = {}
    option_ids = sorted(option_hists)
    for seed in args.random_seeds:
        rng = random.Random(seed)
        selected = rng.sample(option_ids, 3)
        random3[str(seed)] = summarize_selection(
            selected,
            option_name_map,
            pairwise_distances,
        )

    output = {
        "method": "brightness_histogram_farthest_point",
        "selection_data_json": args.data_json,
        "manifest": args.manifest,
        "num_samples": len(sample_ids),
        "bins": args.bins,
        "resize": args.resize,
        "distance": "jensen_shannon",
        "hist3": summarize_selection(hist3_ids, option_name_map, pairwise_distances),
        "random3": random3,
        "option_name_map": {
            str(option_id): option_name_map.get(option_id, "")
            for option_id in option_ids
        },
        "mean_histogram_by_option_id": {
            str(option_id): option_hists[option_id].tolist()
            for option_id in option_ids
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("hist3:", output["hist3"])
    for seed, selection in output["random3"].items():
        print(f"random3 seed={seed}:", selection)
    print(f"Saved probe selection to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
