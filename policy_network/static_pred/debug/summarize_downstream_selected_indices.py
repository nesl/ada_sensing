from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "policy_network" / "results_random_noise" / "G_oracle_full_soft"
DEFAULT_MANIFEST = ROOT / "data" / "ImageNet-ES-Diverse" / "manifest_all.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize policy top-1 selected option_id distribution."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--downstream_json",
        type=Path,
        default=None,
        help="Defaults to results_dir/downstream_test_best.json.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Defaults to results_dir/downstream_selected_index_stats.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_option_name_map(manifest_path: Optional[Path]) -> Dict[int, str]:
    if manifest_path is None or not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    for item in manifest:
        candidates = item.get("candidates") or []
        if candidates:
            return {
                int(candidate["option_id"]): str(
                    candidate.get("meta", {}).get("option_name", "")
                )
                for candidate in candidates
            }
    return {}


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def selected_distribution(
    downstream_json: Path,
    option_name_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    payload = load_json(downstream_json)
    counter: Counter[int] = Counter()

    for record in payload.get("records", []):
        topk = record.get("policy_topk") or []
        if topk:
            counter[int(topk[0]["option_id"])] += 1

    total = sum(counter.values())
    return [
        {
            "option_id": option_id,
            "option_name": option_name_map.get(option_id, ""),
            "count": count,
            "percent": pct(count, total),
        }
        for option_id, count in sorted(counter.items())
    ]


def main() -> None:
    args = parse_args()
    downstream_json = args.downstream_json or args.results_dir / "downstream_test_best.json"
    output_json = args.output_json or args.results_dir / "downstream_selected_index_stats.json"
    option_name_map = load_option_name_map(args.manifest)

    output_payload = {
        "config": {
            "results_dir": str(args.results_dir),
            "downstream_json": str(downstream_json),
            "manifest": str(args.manifest) if args.manifest else None,
        },
        "option_name_map": {
            str(key): value for key, value in sorted(option_name_map.items())
        },
        "downstream_test_best": {
            "selected_distribution": selected_distribution(
                downstream_json,
                option_name_map,
            ),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Saved JSON summary to: {output_json}")


if __name__ == "__main__":
    main()
