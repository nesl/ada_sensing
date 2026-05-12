from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = (
    ROOT / "policy_network" / "results" / "results_random_input_ablation_inference"
)

METRIC_NAMES = [
    "index_acc",
    "downstream_top1_acc",
]

SPECS = [
    (
        "single input",
        "single_fixedk",
        [
            ("real_fixed_input", "true"),
            ("random_fixed_input", "random"),
        ],
    ),
    (
        "dual input",
        "dual_fixedk",
        [
            ("real_ae_real_fixed_input", "trueAE_trueK"),
            ("random_ae_real_fixed_input", "randomAE_trueK"),
            ("real_ae_random_fixed_input", "trueAE_randomK"),
            ("random_ae_random_fixed_input", "randomAE_randomK"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize inference-stage random-input ablation results."
    )
    parser.add_argument("--results_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoint_kind", type=str, default="best")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary_best.csv",
        help="Wide CSV with avg/std across seed directories.",
    )
    parser.add_argument(
        "--raw_output_csv",
        type=Path,
        default=None,
        help="Optional long-form per-seed CSV for debugging.",
    )
    parser.add_argument(
        "--aggregate_seed_dirs",
        action="store_true",
        help="Collect seed-* result directories before summarizing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def percent_payload(payload: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, dict) and "acc" in current:
        return float(current["acc"])
    if isinstance(current, (int, float)):
        return float(current)
    return None


def collect_rows(
    results_root: Path, checkpoint_kind: str, seed: Optional[str] = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tag, family_dir, settings in SPECS:
        for setting_label, setting_dir_name in settings:
            setting_dir = results_root / family_dir / setting_dir_name
            for run_dir in sorted(setting_dir.glob("fixed_k_*/F_oracle_full_hard")):
                input_index = int(run_dir.parent.name.rsplit("_", 1)[1])
                index_payload = load_json(run_dir / f"index_test_{checkpoint_kind}.json")
                downstream_payload = load_json(
                    run_dir / f"downstream_test_{checkpoint_kind}.json"
                )
                rows.append(
                    {
                        "seed": seed,
                        "tag": tag,
                        "input_index": input_index,
                        "setting": setting_label,
                        "index_acc": percent_payload(index_payload, "summary", "acc"),
                        "downstream_top1_acc": percent_payload(
                            downstream_payload,
                            "summary",
                            "rankwise_downstream_acc",
                            "rank_1",
                        ),
                    }
                )
    return rows


def seed_sort_key(path: Path) -> Tuple[int, str]:
    suffix = path.name.removeprefix("seed-").removeprefix("seed")
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (10**9, path.name)


def collect_seed_rows(results_root: Path, checkpoint_kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seed_dirs = [
        path
        for path in sorted(results_root.glob("seed*"), key=seed_sort_key)
        if path.is_dir()
    ]
    for seed_dir in seed_dirs:
        seed = seed_dir.name.removeprefix("seed-").removeprefix("seed")
        rows.extend(collect_rows(seed_dir, checkpoint_kind, seed=seed))
    return rows


def summarize_values(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def build_wide_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["tag"], row["input_index"]), []).append(row)

    wide_rows: List[Dict[str, Any]] = []
    for (tag, input_index), group_rows in sorted(grouped.items()):
        out: Dict[str, Any] = {
            "tag": tag,
            "input_index": input_index,
            "n_seeds": len(
                {row["seed"] for row in group_rows if row.get("seed") is not None}
            ),
        }
        settings = [
            setting_label
            for spec_tag, _, spec_settings in SPECS
            if spec_tag == tag
            for setting_label, _ in spec_settings
        ]
        for setting in settings:
            setting_rows = [row for row in group_rows if row["setting"] == setting]
            for metric_name in METRIC_NAMES:
                values = [
                    row[metric_name]
                    for row in setting_rows
                    if row.get(metric_name) is not None
                ]
                avg, std = summarize_values(values)
                out[f"{setting}__{metric_name}__avg"] = avg
                out[f"{setting}__{metric_name}__std"] = std
        wide_rows.append(out)
    return wide_rows


def write_csv(rows: Iterable[Dict[str, Any]], output_csv: Path) -> None:
    rows = list(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = ["tag", "input_index", "n_seeds"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_raw_csv(rows: Iterable[Dict[str, Any]], output_csv: Path) -> None:
    rows = list(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["seed", "tag", "input_index", "setting", *METRIC_NAMES]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.aggregate_seed_dirs:
        raw_rows = collect_seed_rows(args.results_root, args.checkpoint_kind)
    else:
        raw_rows = collect_rows(args.results_root, args.checkpoint_kind)

    if args.raw_output_csv is not None:
        write_raw_csv(raw_rows, args.raw_output_csv)
        print(f"Wrote {len(raw_rows)} per-seed rows to {args.raw_output_csv}")

    wide_rows = build_wide_rows(raw_rows)
    write_csv(wide_rows, args.output_csv)
    print(f"Wrote {len(wide_rows)} wide summary rows to {args.output_csv}")


if __name__ == "__main__":
    main()
