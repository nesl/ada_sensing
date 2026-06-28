from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize downstream top-1 accuracy for a fixed-k sweep."
    )
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument(
        "--run_name",
        type=str,
        default="G2_oracle_full_soft_allwrong_uniform_w01",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default="downstream_top1_best.json",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--num_k", type=int, default=27)
    return parser.parse_args()


def read_top1(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    rank_1 = payload["summary"]["rankwise_downstream_acc"]["rank_1"]
    return {
        "correct": int(rank_1["correct"]),
        "total": int(rank_1["total"]),
        "acc": float(rank_1["acc"]),
    }


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    rows: List[Dict[str, Any]] = []
    for fixed_k in range(args.num_k):
        result_path = (
            results_dir
            / f"fixed_k_{fixed_k:02d}"
            / args.run_name
            / args.json_name
        )
        row: Dict[str, Any] = {
            "fixed_k": fixed_k,
            "result_path": str(result_path),
            "exists": result_path.exists(),
        }
        if result_path.exists():
            top1 = read_top1(result_path)
            row.update(
                {
                    "downstream_top1_correct": top1["correct"],
                    "downstream_top1_total": top1["total"],
                    "downstream_top1_acc": top1["acc"],
                }
            )
        rows.append(row)

    valid_rows = [row for row in rows if row["exists"]]
    best_row = (
        max(valid_rows, key=lambda row: row["downstream_top1_acc"])
        if valid_rows
        else None
    )
    summary = {
        "results_dir": str(results_dir),
        "run_name": args.run_name,
        "json_name": args.json_name,
        "num_expected": args.num_k,
        "num_found": len(valid_rows),
        "best": best_row,
        "rows": rows,
    }

    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "summary_top1.csv"
    output_json = Path(args.output_json) if args.output_json else results_dir / "summary_top1.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "fixed_k",
        "exists",
        "downstream_top1_correct",
        "downstream_top1_total",
        "downstream_top1_acc",
        "result_path",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    if best_row is None:
        print("No downstream top-1 result files found.")
    else:
        print(
            "Best fixed_k="
            f"{best_row['fixed_k']:02d}, "
            f"top1={best_row['downstream_top1_acc']:.2f}% "
            f"({best_row['downstream_top1_correct']}/"
            f"{best_row['downstream_top1_total']})"
        )
    print(f"Saved {output_csv}")
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
