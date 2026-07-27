from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    DATASET_IN,
    MODEL_SPECS,
    project_root,
)


def read_result(raw_dir: Path, model: str, dataset: str) -> Dict[str, Any] | None:
    path = raw_dir / f"{model}__{dataset}.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("is_smoke_test"):
        return None
    return result


def reproduced(result: Dict[str, Any] | None) -> str:
    return "" if result is None else f"{result['micro_accuracy']:.6f}"


def make_rows(raw_dir: Path) -> list[Dict[str, Any]]:
    rows = []
    for spec in MODEL_SPECS:
        in_result = read_result(raw_dir, spec.key, DATASET_IN)
        ae_es_result = read_result(raw_dir, spec.key, DATASET_AE_ES)
        ae_diverse_result = read_result(raw_dir, spec.key, DATASET_AE_DIVERSE)
        rows.append(
            {
                "model": spec.paper_name,
                "checkpoint": spec.checkpoint,
                "IN_paper": spec.paper_in,
                "IN_reproduced": reproduced(in_result),
                "AE_ImageNet_ES_paper": spec.paper_ae_es,
                "AE_ImageNet_ES_reproduced": reproduced(ae_es_result),
                "AE_Diverse_paper": spec.paper_ae_diverse,
                "AE_Diverse_reproduced": reproduced(ae_diverse_result),
                "evaluation_label_space": (
                    "closed 200-way Tiny-ImageNet subset; 1000 logits sliced "
                    "to sorted 200 WNIDs before argmax"
                ),
                "AE_aggregation": (
                    "micro over all 5 AE shots and all environments "
                    "(ES: 2×5×1000; Diverse: 6×5×1000); macro setting mean retained"
                ),
            }
        )
    return rows


def write_csv(rows: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def write_markdown(rows: list[Dict[str, Any]], path: Path) -> None:
    columns = [
        "model",
        "IN_paper",
        "IN_reproduced",
        "AE_ImageNet_ES_paper",
        "AE_ImageNet_ES_reproduced",
        "AE_Diverse_paper",
        "AE_Diverse_reproduced",
    ]
    labels = [
        "Model",
        "IN paper",
        "IN reproduced",
        "AE ES paper",
        "AE ES reproduced",
        "AE Diverse paper",
        "AE Diverse reproduced",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final reproduction deliverables.")
    parser.add_argument("--raw-dir", type=Path, default=project_root() / "results" / "raw")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=project_root() / "results" / "in_ae_reproduction",
    )
    args = parser.parse_args()
    rows = make_rows(args.raw_dir)
    write_csv(rows, args.output_prefix.with_suffix(".csv"))
    write_json(rows, args.output_prefix.with_suffix(".json"))
    write_markdown(rows, args.output_prefix.with_suffix(".md"))
    print(f"Wrote {args.output_prefix}.csv/.json/.md")


if __name__ == "__main__":
    main()

