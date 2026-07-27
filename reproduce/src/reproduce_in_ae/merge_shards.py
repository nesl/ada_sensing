from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .evaluate import atomic_json_dump
from .protocol import MODEL_BY_KEY, paper_value, project_root


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("At least two shard files are required")
    shards = [load(path) for path in paths]
    model = shards[0]["model"]
    dataset = shards[0]["dataset"]
    shard_count = shards[0]["shard"]["count"]
    indices = sorted(shard["shard"]["index"] for shard in shards)
    if any(
        shard["model"] != model
        or shard["dataset"] != dataset
        or shard["shard"]["count"] != shard_count
        for shard in shards
    ):
        raise ValueError("Shard model, dataset, or shard-count mismatch")
    if len(shards) != shard_count or indices != list(range(shard_count)):
        raise ValueError(f"Expected shard indices 0..{shard_count - 1}, found {indices}")

    result = copy.deepcopy(shards[0])
    result["correct"] = sum(shard["correct"] for shard in shards)
    result["total"] = sum(shard["total"] for shard in shards)
    if result["total"] != result["full_dataset_total"]:
        raise ValueError(
            f"Merged total {result['total']} != full total "
            f"{result['full_dataset_total']}"
        )
    settings = sorted(
        {setting for shard in shards for setting in shard["per_setting"]}
    )
    result["per_setting"] = {}
    for setting in settings:
        correct = sum(
            shard["per_setting"].get(setting, {}).get("correct", 0)
            for shard in shards
        )
        total = sum(
            shard["per_setting"].get(setting, {}).get("total", 0)
            for shard in shards
        )
        result["per_setting"][setting] = {
            "correct": correct,
            "total": total,
            "accuracy": 100.0 * correct / total,
        }
    result["micro_accuracy"] = 100.0 * result["correct"] / result["total"]
    result["macro_setting_accuracy"] = sum(
        row["accuracy"] for row in result["per_setting"].values()
    ) / len(result["per_setting"])
    result["paper_value"] = paper_value(MODEL_BY_KEY[model], dataset)
    result["paper_rounding_match"] = (
        round(result["micro_accuracy"], 1) == result["paper_value"]
    )
    result["elapsed_seconds"] = max(shard["elapsed_seconds"] for shard in shards)
    result["sum_gpu_seconds"] = sum(shard["elapsed_seconds"] for shard in shards)
    result["shard"] = {
        "index": None,
        "count": shard_count,
        "merged": True,
        "source_files": [path.name for path in paths],
    }
    result["model_provenance"] = {
        **result["model_provenance"],
        "devices": [shard["model_provenance"]["device"] for shard in shards],
        "device": "merged deterministic shards",
    }
    result["protocol"]["expected_setting_counts"] = {
        key: row["total"] for key, row in result["per_setting"].items()
    }
    result["protocol"]["batch_size_per_shard"] = [
        shard["protocol"]["batch_size"] for shard in shards
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic result shards.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_KEY))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=project_root() / "results" / "raw"
    )
    args = parser.parse_args()
    paths = [
        args.raw_dir
        / f"{args.model}__{args.dataset}__shard_{index}_of_{args.shard_count}.json"
        for index in range(args.shard_count)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")
    result = merge(paths)
    output = args.raw_dir / f"{args.model}__{args.dataset}.json"
    atomic_json_dump(result, output)
    print(
        f"Merged {len(paths)} shards: {result['micro_accuracy']:.6f}% -> {output}"
    )


if __name__ == "__main__":
    main()
