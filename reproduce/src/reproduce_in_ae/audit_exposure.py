from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .exposure import default_specs
from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    MODEL_SPECS,
    project_root,
)


EXPECTED_TOTALS = {
    DATASET_AE_ES: 10_000,
    DATASET_AE_DIVERSE: 30_000,
}
EXPECTED_SETTINGS = {
    DATASET_AE_ES: 10,
    DATASET_AE_DIVERSE: 30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit completeness and invariants of AE exposure raw results."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_root() / "results" / "ae_exposure_raw",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=project_root() / "results" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results" / "ae_exposure_raw" / "audit.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def inspect_result(
    json_path: Path,
    npz_path: Path,
    model_key: str,
    dataset: str,
    mode: str,
    value: float,
) -> tuple[list[str], str | None]:
    issues: list[str] = []
    if not json_path.is_file():
        return ["missing JSON"], None
    if not npz_path.is_file():
        return ["missing NPZ"], None
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable JSON: {error}"], None
    try:
        with np.load(npz_path) as arrays:
            targets = arrays["targets"]
            predictions = arrays["predictions"]
            hits = arrays["hits"]
    except (OSError, ValueError, KeyError) as error:
        return [f"unreadable NPZ: {error}"], None

    expected_total = EXPECTED_TOTALS[dataset]
    if result.get("model") != model_key:
        issues.append(f"model={result.get('model')!r}")
    if result.get("dataset") != dataset:
        issues.append(f"dataset={result.get('dataset')!r}")
    exposure = result.get("exposure", {})
    if exposure.get("mode") != mode or float(exposure.get("value", float("nan"))) != value:
        issues.append(f"exposure={exposure!r}")
    if result.get("is_smoke_test"):
        issues.append("formal result marked as smoke test")
    if int(result.get("total", -1)) != expected_total:
        issues.append(f"total={result.get('total')}; expected {expected_total}")
    if int(result.get("full_dataset_total", -1)) != expected_total:
        issues.append(
            f"full_dataset_total={result.get('full_dataset_total')}; "
            f"expected {expected_total}"
        )
    if targets.size != expected_total:
        issues.append(f"targets size={targets.size}; expected {expected_total}")
    if predictions.size != expected_total:
        issues.append(
            f"predictions size={predictions.size}; expected {expected_total}"
        )
    if hits.size != expected_total:
        issues.append(f"hits size={hits.size}; expected {expected_total}")
    if int(hits.sum()) != int(result.get("correct", -1)):
        issues.append(
            f"NPZ hits={int(hits.sum())}; JSON correct={result.get('correct')}"
        )
    per_setting = result.get("per_setting", {})
    if len(per_setting) != EXPECTED_SETTINGS[dataset]:
        issues.append(
            f"settings={len(per_setting)}; expected {EXPECTED_SETTINGS[dataset]}"
        )
    wrong_setting_totals = {
        key: row.get("total")
        for key, row in per_setting.items()
        if int(row.get("total", -1)) != 1000
    }
    if wrong_setting_totals:
        issues.append(f"per-setting totals are not 1000: {wrong_setting_totals}")
    path_hash = result.get("path_order_sha256")
    if not isinstance(path_hash, str) or len(path_hash) != 64:
        issues.append("missing or invalid path_order_sha256")
        path_hash = None
    return issues, path_hash


def main() -> None:
    args = parse_args()
    specs = default_specs()
    expected_count = len(MODEL_SPECS) * len(EXPECTED_TOTALS) * len(specs)
    complete = 0
    problems: list[Dict[str, Any]] = []
    hashes: Dict[str, set[str]] = defaultdict(set)
    zero_baseline_checks: list[Dict[str, Any]] = []

    for model in MODEL_SPECS:
        for dataset in EXPECTED_TOTALS:
            for spec in specs:
                stem = f"{model.key}__{dataset}__{spec.tag}"
                json_path = args.raw_dir / f"{stem}.json"
                npz_path = args.raw_dir / f"{stem}.npz"
                issues, path_hash = inspect_result(
                    json_path=json_path,
                    npz_path=npz_path,
                    model_key=model.key,
                    dataset=dataset,
                    mode=spec.mode,
                    value=spec.value,
                )
                if issues:
                    problems.append(
                        {
                            "model": model.key,
                            "dataset": dataset,
                            "exposure_tag": spec.tag,
                            "issues": issues,
                        }
                    )
                else:
                    complete += 1
                    assert path_hash is not None
                    hashes[dataset].add(path_hash)

                if spec.mode == "fixed_ev" and spec.value == 0.0 and not issues:
                    with json_path.open("r", encoding="utf-8") as handle:
                        adjusted = json.load(handle)
                    baseline_path = (
                        args.baseline_dir / f"{model.key}__{dataset}.json"
                    )
                    baseline_issues: list[str] = []
                    if not baseline_path.is_file():
                        baseline_issues.append(f"missing {baseline_path}")
                    else:
                        with baseline_path.open("r", encoding="utf-8") as handle:
                            baseline = json.load(handle)
                        for field in ("correct", "total"):
                            if int(adjusted[field]) != int(baseline[field]):
                                baseline_issues.append(
                                    f"{field}: exposure={adjusted[field]}, "
                                    f"baseline={baseline[field]}"
                                )
                    zero_baseline_checks.append(
                        {
                            "model": model.key,
                            "dataset": dataset,
                            "status": "ok" if not baseline_issues else "mismatch",
                            "issues": baseline_issues,
                        }
                    )

    hash_issues = {
        dataset: sorted(values)
        for dataset, values in hashes.items()
        if len(values) > 1
    }
    baseline_mismatches = [
        row for row in zero_baseline_checks if row["status"] != "ok"
    ]
    status = (
        "ok"
        if complete == expected_count
        and not problems
        and not hash_issues
        and not baseline_mismatches
        else "incomplete_or_invalid"
    )
    payload = {
        "status": status,
        "expected_results": expected_count,
        "complete_results": complete,
        "problem_count": len(problems),
        "problems": problems,
        "path_order_hashes": {
            dataset: sorted(values) for dataset, values in hashes.items()
        },
        "path_order_hash_issues": hash_issues,
        "zero_ev_baseline_checks": zero_baseline_checks,
        "zero_ev_baseline_mismatch_count": len(baseline_mismatches),
    }
    atomic_json_dump(payload, args.output)
    print(
        f"AE exposure raw audit: {status}; "
        f"{complete}/{expected_count} complete; {len(problems)} problem(s)"
    )
    print(f"Audit: {args.output.resolve()}")
    if status != "ok" and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
