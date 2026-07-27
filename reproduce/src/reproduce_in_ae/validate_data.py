from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .datasets import DatasetRoots, build_dataset, default_roots, setting_counts
from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    DATASET_IN,
    project_root,
    workspace_root,
)


EXPECTED = {
    DATASET_IN: {"total": 1000, "settings": 1, "per_setting": 1000},
    DATASET_AE_ES: {"total": 10000, "settings": 10, "per_setting": 1000},
    DATASET_AE_DIVERSE: {"total": 30000, "settings": 30, "per_setting": 1000},
}


def validate(name: str, roots: DatasetRoots) -> Dict[str, Any]:
    try:
        dataset = build_dataset(name, roots)
    except FileNotFoundError as error:
        return {"dataset": name, "status": "missing", "error": str(error)}
    counts = setting_counts(dataset)
    expected = EXPECTED[name]
    issues = []
    if len(dataset.classes) != 200:
        issues.append(f"expected 200 classes, found {len(dataset.classes)}")
    if len(dataset) != expected["total"]:
        issues.append(f"expected {expected['total']} images, found {len(dataset)}")
    if len(counts) != expected["settings"]:
        issues.append(f"expected {expected['settings']} settings, found {len(counts)}")
    wrong_settings = {
        setting: count
        for setting, count in counts.items()
        if count != expected["per_setting"]
    }
    if wrong_settings:
        issues.append(f"settings with unexpected counts: {wrong_settings}")
    return {
        "dataset": name,
        "status": "ok" if not issues else "invalid",
        "root_classes": len(dataset.classes),
        "total": len(dataset),
        "setting_counts": counts,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    defaults = default_roots(workspace_root())
    parser = argparse.ArgumentParser(description="Validate paper evaluation datasets.")
    parser.add_argument("--in-root", type=Path, default=defaults.in_root)
    parser.add_argument("--ae-es-root", type=Path, default=defaults.ae_es_root)
    parser.add_argument("--ae-diverse-root", type=Path, default=defaults.ae_diverse_root)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "evidence" / "data_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = DatasetRoots(args.in_root, args.ae_es_root, args.ae_diverse_root)
    results = [
        validate(DATASET_IN, roots),
        validate(DATASET_AE_ES, roots),
        validate(DATASET_AE_DIVERSE, roots),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result['dataset']}: {result['status']}"
                + (
                    f" ({result.get('total')} images)"
                    if result["status"] != "missing"
                    else f" ({result['error']})"
                )
            )
            for issue in result.get("issues", []):
                print(f"  - {issue}")
    if any(result["status"] != "ok" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
