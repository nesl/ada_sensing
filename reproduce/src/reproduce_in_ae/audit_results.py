from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .protocol import DATASET_NAMES, MODEL_SPECS, project_root


EXPECTED_TOTALS = {
    "in": 1_000,
    "ae_imagenet_es": 10_000,
    "ae_imagenet_es_diverse": 30_000,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit(raw_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[dict[str, Any]] = []
    expected_names = {
        f"{spec.key}__{dataset}.json"
        for spec in MODEL_SPECS
        for dataset in DATASET_NAMES
    }
    actual_names = {
        path.name
        for path in raw_dir.glob("*.json")
        if "__smoke_" not in path.name and "__shard_" not in path.name
    }

    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        errors.append(f"Missing production artifacts: {missing}")
    if unexpected:
        errors.append(f"Unexpected production artifacts: {unexpected}")

    for spec in MODEL_SPECS:
        for dataset in DATASET_NAMES:
            path = raw_dir / f"{spec.key}__{dataset}.json"
            if not path.is_file():
                continue
            result = load_json(path)
            prefix = path.name
            expected_total = EXPECTED_TOTALS[dataset]
            per_setting = result.get("per_setting", {})
            protocol = result.get("protocol", {})
            provenance = result.get("model_provenance", {})

            checks = {
                "model": result.get("model") == spec.key,
                "dataset": result.get("dataset") == dataset,
                "not_smoke": result.get("is_smoke_test") is False,
                "total": result.get("total") == expected_total,
                "full_dataset_total": result.get("full_dataset_total") == expected_total,
                "correct_range": 0 <= result.get("correct", -1) <= expected_total,
                "setting_total": sum(
                    row.get("total", 0) for row in per_setting.values()
                )
                == expected_total,
                "expected_setting_counts": {
                    key: row.get("total") for key, row in per_setting.items()
                }
                == protocol.get("expected_setting_counts"),
                "micro_recomputed": math.isclose(
                    result.get("micro_accuracy", math.nan),
                    100.0 * result.get("correct", 0) / expected_total,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "macro_equals_micro_balanced": math.isclose(
                    result.get("macro_setting_accuracy", math.nan),
                    result.get("micro_accuracy", math.nan),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "resize": protocol.get("resize") == 256,
                "crop": protocol.get("crop") == 224,
                "weights_frozen": protocol.get("weights_frozen") is True,
                "no_adaptation": protocol.get("target_domain_adaptation") is False,
                "lens_torch_runtime": provenance.get("torch_version") == "2.10.0+cu128",
            }
            failed = sorted(key for key, passed in checks.items() if not passed)
            if failed:
                errors.append(f"{prefix}: failed checks {failed}")
            if not provenance.get("checkpoint_sha256"):
                warnings.append(
                    f"{prefix}: framework did not expose a single checkpoint SHA-256"
                )
            checked.append(
                {
                    "artifact": prefix,
                    "accuracy": result.get("micro_accuracy"),
                    "paper_value": result.get("paper_value"),
                    "paper_rounding_match": result.get("paper_rounding_match"),
                    "checks_passed": not failed,
                }
            )

    return {
        "status": "pass" if not errors else "fail",
        "runtime_python": sys.executable,
        "expected_production_artifacts": len(expected_names),
        "actual_production_artifacts": len(actual_names),
        "checked": checked,
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all production result JSONs.")
    parser.add_argument(
        "--raw-dir", type=Path, default=project_root() / "results" / "raw"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results" / "audit.json",
    )
    args = parser.parse_args()
    summary = audit(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{summary['status'].upper()}: "
        f"{summary['actual_production_artifacts']}/"
        f"{summary['expected_production_artifacts']} production artifacts"
    )
    if summary["errors"]:
        for error in summary["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
