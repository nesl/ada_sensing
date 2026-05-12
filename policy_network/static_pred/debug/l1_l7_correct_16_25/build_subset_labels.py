from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = ROOT / "data" / "ImageNet-ES-Diverse"
DEFAULT_SOURCE_DIR = DEFAULT_DATA_ROOT / "oracle_policy_labels"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "manifest_all.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build l1/l7 labels where option 16 or 25 is downstream-correct. "
            "The new hard target is restricted to option_id 16 or 25."
        )
    )
    parser.add_argument("--source_dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--envs", type=str, default="l1,l7")
    parser.add_argument("--target_option_ids", type=str, default="16,25")
    parser.add_argument("--num_candidates", type=int, default=27)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_csv_ints(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_strings(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_manifest_candidate_index(manifest_path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    manifest_items = load_json(manifest_path)
    index: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for item in manifest_items:
        sample_id = str(item["id"])
        index[sample_id] = {
            int(candidate["option_id"]): {
                "position": pos,
                "path": candidate["path"],
                "option_name": candidate.get("meta", {}).get("option_name", ""),
            }
            for pos, candidate in enumerate(item["candidates"])
        }
    return index


def choose_target_option_id(
    item: Dict[str, Any],
    target_option_ids: List[int],
) -> Tuple[int, str]:
    if not bool(item.get("oracle_had_correct_candidate", False)):
        raise ValueError(
            f"Sample {item.get('sample_id')} has no downstream-correct candidate; "
            "oracle_correct_option_ids contains a fallback target."
        )
    correct_ids = [int(option_id) for option_id in item.get("oracle_correct_option_ids", [])]
    present_targets = [option_id for option_id in target_option_ids if option_id in correct_ids]
    if not present_targets:
        raise ValueError(f"No target option is downstream-correct for {item.get('sample_id')}.")
    if len(present_targets) == 1:
        return present_targets[0], "single_correct_target"

    weights_by_option = {
        int(option_id): float(weight)
        for option_id, weight in zip(
            item.get("oracle_correct_option_ids", []),
            item.get("oracle_correct_option_weights", []),
        )
    }
    if weights_by_option:
        return max(present_targets, key=lambda option_id: weights_by_option.get(option_id, 0.0)), "higher_oracle_weight"

    return present_targets[0], "first_target_fallback"


def make_soft_target(target_option_id: int, num_candidates: int) -> List[float]:
    soft_target = [0.0 for _ in range(num_candidates)]
    soft_target[target_option_id] = 1.0
    return soft_target


def rewrite_record(
    item: Dict[str, Any],
    target_option_id: int,
    target_reason: str,
    target_option_ids: List[int],
    candidate_index: Dict[str, Dict[int, Dict[str, Any]]],
    num_candidates: int,
) -> Dict[str, Any]:
    sample_id = str(item["sample_id"])
    candidate = candidate_index[sample_id][target_option_id]
    rewritten = dict(item)
    rewritten["source_best_idx_in_candidates"] = int(item.get("best_idx_in_candidates", -1))
    rewritten["source_best_option_id"] = int(item.get("best_option_id", -1))
    rewritten["source_best_option_name"] = str(item.get("best_option_name", ""))
    rewritten["source_best_path"] = str(item.get("best_path", ""))
    rewritten["target_label_source"] = "downstream_correct_16_or_25"
    rewritten["target_label_reason"] = target_reason
    rewritten["target_option_ids"] = target_option_ids
    rewritten["best_idx_in_candidates"] = int(candidate["position"])
    rewritten["best_option_id"] = int(target_option_id)
    rewritten["best_option_name"] = str(candidate["option_name"])
    rewritten["best_path"] = str(candidate["path"])
    rewritten["soft_target"] = make_soft_target(target_option_id, num_candidates)
    rewritten["sample_weight"] = 1.0
    return rewritten


def summarize(records_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    all_records = [record for records in records_by_split.values() for record in records]
    return {
        "total_kept": len(all_records),
        "target_distribution": {
            str(option_id): count
            for option_id, count in sorted(
                Counter(int(record["best_option_id"]) for record in all_records).items()
            )
        },
        "target_distribution_by_split": {
            split: {
                str(option_id): count
                for option_id, count in sorted(
                    Counter(int(record["best_option_id"]) for record in records).items()
                )
            }
            for split, records in records_by_split.items()
        },
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    keep_envs = set(parse_csv_strings(args.envs))
    target_option_ids = parse_csv_ints(args.target_option_ids)
    candidate_index = build_manifest_candidate_index(Path(args.manifest))

    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        source_path = source_dir / f"oracle_policy_{split}_labels.json"
        output_path = output_dir / f"oracle_policy_{split}_labels.json"
        rewritten_records: List[Dict[str, Any]] = []

        for item in load_json(source_path):
            env = str(item.get("env"))
            if env not in keep_envs:
                continue
            if not bool(item.get("oracle_had_correct_candidate", False)):
                continue
            correct_ids = {int(option_id) for option_id in item.get("oracle_correct_option_ids", [])}
            if not any(option_id in correct_ids for option_id in target_option_ids):
                continue
            target_option_id, target_reason = choose_target_option_id(item, target_option_ids)
            rewritten_records.append(
                rewrite_record(
                    item,
                    target_option_id,
                    target_reason,
                    target_option_ids,
                    candidate_index,
                    args.num_candidates,
                )
            )

        records_by_split[split] = rewritten_records
        write_json(output_path, rewritten_records)

    summary = {
        "source_dir": str(source_dir),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "keep_envs": sorted(keep_envs),
        "target_option_ids": target_option_ids,
        **summarize(records_by_split),
    }
    write_json(output_dir / "label_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
