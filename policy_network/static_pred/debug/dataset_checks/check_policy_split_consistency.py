from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = ROOT / "data" / "ImageNet-ES-Diverse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check policy/oracle split consistency, per-class group counts, "
            "and whether both datasets use the same group-level split."
        )
    )
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def class_id_of(record: Dict[str, Any]) -> str:
    return str(record.get("class_id", str(record["group_id"]).split("__")[0]))


def summarize_split(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = {str(record["group_id"]) for record in records}
    classes = {class_id_of(record) for record in records}
    per_class_groups = defaultdict(set)
    for record in records:
        per_class_groups[class_id_of(record)].add(str(record["group_id"]))

    return {
        "samples": len(records),
        "groups": len(groups),
        "classes": len(classes),
        "per_class_group_count_hist": dict(
            sorted(Counter(len(v) for v in per_class_groups.values()).items())
        ),
    }


def build_group_to_split(split_payloads: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    group_to_split: Dict[str, str] = {}
    for split_name, records in split_payloads.items():
        for record in records:
            group_id = str(record["group_id"])
            prev = group_to_split.get(group_id)
            if prev is not None and prev != split_name:
                raise ValueError(
                    f"group_id={group_id} appears in both {prev} and {split_name}"
                )
            group_to_split[group_id] = split_name
    return group_to_split


def validate_policy_311(split_payloads: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    class_to_split_groups = defaultdict(lambda: defaultdict(set))

    for split_name, records in split_payloads.items():
        for record in records:
            class_to_split_groups[class_id_of(record)][split_name].add(str(record["group_id"]))

    failures = []
    for class_id, split_groups in sorted(class_to_split_groups.items()):
        train_count = len(split_groups["train"])
        val_count = len(split_groups["val"])
        test_count = len(split_groups["test"])
        total_count = train_count + val_count + test_count
        if (train_count, val_count, test_count, total_count) != (3, 1, 1, 5):
            failures.append(
                {
                    "class_id": class_id,
                    "train_groups": train_count,
                    "val_groups": val_count,
                    "test_groups": test_count,
                    "total_groups": total_count,
                }
            )
    return failures


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)

    datasets = {
        "policy": {
            "train": data_root / "policy_labels" / "policy_train_labels.json",
            "val": data_root / "policy_labels" / "policy_val_labels.json",
            "test": data_root / "policy_labels" / "policy_test_labels.json",
        },
        "oracle": {
            "train": data_root / "oracle_policy_labels" / "oracle_policy_train_labels.json",
            "val": data_root / "oracle_policy_labels" / "oracle_policy_val_labels.json",
            "test": data_root / "oracle_policy_labels" / "oracle_policy_test_labels.json",
        },
    }

    loaded = {
        dataset_name: {
            split_name: load_json(path) for split_name, path in split_paths.items()
        }
        for dataset_name, split_paths in datasets.items()
    }

    payload: Dict[str, Any] = {"datasets": {}, "comparisons": {}}

    for dataset_name in ("policy", "oracle"):
        dataset_summary: Dict[str, Any] = {}
        for split_name in ("train", "val", "test"):
            dataset_summary[split_name] = summarize_split(loaded[dataset_name][split_name])

        group_to_split = build_group_to_split(loaded[dataset_name])
        dataset_summary["group_assignment"] = {
            "num_unique_groups": len(group_to_split),
            "has_overlap": False,
        }
        dataset_summary["split_rule_failures"] = validate_policy_311(loaded[dataset_name])
        payload["datasets"][dataset_name] = dataset_summary

    policy_group_to_split = build_group_to_split(loaded["policy"])
    oracle_group_to_split = build_group_to_split(loaded["oracle"])
    mismatches = [
        {
            "group_id": group_id,
            "policy_split": policy_group_to_split[group_id],
            "oracle_split": oracle_group_to_split.get(group_id),
        }
        for group_id in sorted(policy_group_to_split)
        if oracle_group_to_split.get(group_id) != policy_group_to_split[group_id]
    ]

    payload["comparisons"]["policy_vs_oracle"] = {
        "same_group_universe": set(policy_group_to_split) == set(oracle_group_to_split),
        "num_policy_groups": len(policy_group_to_split),
        "num_oracle_groups": len(oracle_group_to_split),
        "num_split_mismatches": len(mismatches),
        "split_mismatches_preview": mismatches[:10],
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
