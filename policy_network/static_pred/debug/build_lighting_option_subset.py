from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / "data" / "ImageNet-ES-Diverse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a filtered oracle-policy label subset by lighting env and "
            "best_option_id, preserving the original train/val/test split."
        )
    )
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--source_dir",
        type=str,
        default="oracle_policy_labels",
        help="Directory under data_root containing oracle_policy_*_labels.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="oracle_policy_labels_l1_l7_option2_25",
        help="Output directory under data_root unless an absolute path is given.",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="l1,l7",
        help="Comma-separated env names to keep.",
    )
    parser.add_argument(
        "--option_ids",
        type=str,
        default="2,25",
        help="Comma-separated best_option_id values to keep.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_csv_strings(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> List[int]:
    return [int(part) for part in parse_csv_strings(raw)]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_env = Counter(str(record["env"]) for record in records)
    by_idx = Counter(int(record["best_option_id"]) for record in records)
    cross = Counter(
        (str(record["env"]), int(record["best_option_id"])) for record in records
    )
    option_names = sorted(
        {
            (int(record["best_option_id"]), str(record.get("best_option_name", "")))
            for record in records
        }
    )
    return {
        "samples": len(records),
        "by_env": dict(sorted(by_env.items())),
        "by_idx": {str(k): v for k, v in sorted(by_idx.items())},
        "by_env_idx": {
            f"{env}|{option_id}": count
            for (env, option_id), count in sorted(cross.items())
        },
        "option_names": [
            {"option_id": option_id, "option_name": option_name}
            for option_id, option_name in option_names
        ],
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    source_dir = data_root / args.source_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / output_dir

    keep_envs = set(parse_csv_strings(args.envs))
    keep_option_ids = set(parse_csv_ints(args.option_ids))

    summary: Dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "keep_envs": sorted(keep_envs),
        "keep_option_ids": sorted(keep_option_ids),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        input_path = source_dir / f"oracle_policy_{split}_labels.json"
        records = load_json(input_path)
        filtered = [
            record
            for record in records
            if str(record["env"]) in keep_envs
            and int(record["best_option_id"]) in keep_option_ids
        ]
        output_path = output_dir / f"oracle_policy_{split}_labels.json"
        write_json(output_path, filtered)

        split_summary = summarize(filtered)
        split_summary["source_samples"] = len(records)
        split_summary["output_json"] = str(output_path)
        summary["splits"][split] = split_summary

    summary_path = output_dir / "dataset_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
