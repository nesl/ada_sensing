from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = ROOT / "policy_network" / "results_random_input_ablation_inference"


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
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "summary_best.md",
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


def collect_rows(results_root: Path, checkpoint_kind: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = [
        ("single", "single_fixedk", ["true", "random"]),
        (
            "dual",
            "dual_fixedk",
            ["trueAE_trueK", "randomAE_trueK", "trueAE_randomK", "randomAE_randomK"],
        ),
    ]
    for family, family_dir, settings in specs:
        for setting in settings:
            setting_dir = results_root / family_dir / setting
            for run_dir in sorted(setting_dir.glob("fixed_k_*/F_oracle_full_hard")):
                fixed_k = int(run_dir.parent.name.rsplit("_", 1)[1])
                index_payload = load_json(run_dir / f"index_test_{checkpoint_kind}.json")
                downstream_payload = load_json(
                    run_dir / f"downstream_test_{checkpoint_kind}.json"
                )
                rows.append(
                    {
                        "family": family,
                        "setting": setting,
                        "fixed_k": fixed_k,
                        "index_acc": percent_payload(index_payload, "summary", "acc"),
                        "index_top5_acc": percent_payload(
                            index_payload, "summary", "top5_acc"
                        ),
                        "downstream_top1_acc": percent_payload(
                            downstream_payload,
                            "summary",
                            "rankwise_downstream_acc",
                            "rank_1",
                        ),
                        "downstream_top5_cumulative_acc": percent_payload(
                            downstream_payload,
                            "summary",
                            "cumulative_topk_contains_correct",
                            "top_5_contains_correct",
                        ),
                    }
                )
    return rows


def write_csv(rows: Iterable[Dict[str, Any]], output_csv: Path) -> None:
    rows = list(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "setting",
        "fixed_k",
        "index_acc",
        "index_top5_acc",
        "downstream_top1_acc",
        "downstream_top5_cumulative_acc",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def write_markdown(rows: List[Dict[str, Any]], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Random-input ablation, inference stage",
        "",
        "| Family | Setting | k | Index Acc | Top-5 Index Acc | Downstream Top-1 | Downstream Top-5 Cumulative |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda x: (x["family"], x["setting"], x["fixed_k"])):
        lines.append(
            "| {family} | {setting} | {fixed_k} | {index_acc} | {index_top5_acc} | "
            "{downstream_top1_acc} | {downstream_top5_cumulative_acc} |".format(
                family=row["family"],
                setting=row["setting"],
                fixed_k=row["fixed_k"],
                index_acc=fmt(row["index_acc"]),
                index_top5_acc=fmt(row["index_top5_acc"]),
                downstream_top1_acc=fmt(row["downstream_top1_acc"]),
                downstream_top5_cumulative_acc=fmt(
                    row["downstream_top5_cumulative_acc"]
                ),
            )
        )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.results_root, args.checkpoint_kind)
    write_csv(rows, args.output_csv)
    write_markdown(rows, args.output_md)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote markdown summary to {args.output_md}")


if __name__ == "__main__":
    main()
