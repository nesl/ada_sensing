from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "policy_network" / "results_fixed_input13"

EXPERIMENTS = [
    {
        "id": "B",
        "label": "Lens label",
        "input_sampling": "fixed option_id=13",
        "freeze": "yes",
        "unfrozen_parts": "policy_head only",
        "loss": "hard label",
        "history": BASE / "B_lens_head_hard" / "train_history.json",
        "analysis": BASE / "B_lens_head_hard" / "test_analysis_fixed13.json",
        "compare": BASE / "B_lens_head_hard" / "test_policy_selected_visit_acc_fixed13.json",
    },
    {
        "id": "C",
        "label": "Oracle label",
        "input_sampling": "fixed option_id=13",
        "freeze": "yes",
        "unfrozen_parts": "policy_head only",
        "loss": "hard label",
        "history": BASE / "C_oracle_head_hard" / "train_history.json",
        "analysis": BASE / "C_oracle_head_hard" / "test_analysis_fixed13.json",
        "compare": BASE / "C_oracle_head_hard" / "test_oracle_compare_fixed13.json",
    },
    {
        "id": "D",
        "label": "Oracle label",
        "input_sampling": "fixed option_id=13",
        "freeze": "yes",
        "unfrozen_parts": "policy_head only",
        "loss": "soft label (`soft_kl`)",
        "history": BASE / "D_oracle_head_soft" / "train_history.json",
        "analysis": BASE / "D_oracle_head_soft" / "test_analysis_fixed13.json",
        "compare": BASE / "D_oracle_head_soft" / "test_oracle_soft_compare_fixed13.json",
    },
    {
        "id": "E",
        "label": "Oracle label",
        "input_sampling": "fixed option_id=13",
        "freeze": "partial freeze",
        "unfrozen_parts": "backbone[9:12] + feature_proj + policy_head",
        "loss": "soft label (`soft_kl`)",
        "history": BASE / "E_oracle_partial_soft" / "train_history.json",
        "analysis": BASE / "E_oracle_partial_soft" / "test_analysis_fixed13.json",
        "compare": BASE / "E_oracle_partial_soft" / "test_oracle_part_soft_compare_fixed13.json",
    },
]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "pending"
    return f"`{value:.2f}%`"


def best_val_acc(history_payload: Optional[Any]) -> Optional[float]:
    if not history_payload:
        return None
    return max((float(item["val_acc"]) for item in history_payload), default=None)


def test_index_acc(analysis_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not analysis_payload:
        return None
    return float(analysis_payload["summary"]["acc"])


def downstream_acc(compare_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not compare_payload:
        return None
    return float(compare_payload["summary"]["policy_selected_acc"]["acc"])


def main() -> None:
    print("| ID | Supervision Label | Input Sampling | Backbone Freeze | Unfrozen Parts | Loss | Best Val Index Acc | Test Index Acc | Test Downstream Acc |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for exp in EXPERIMENTS:
        history_payload = load_json(exp["history"])
        analysis_payload = load_json(exp["analysis"])
        compare_payload = load_json(exp["compare"])

        print(
            f"| {exp['id']} | {exp['label']} | {exp['input_sampling']} | "
            f"{exp['freeze']} | {exp['unfrozen_parts']} | {exp['loss']} | "
            f"{format_pct(best_val_acc(history_payload))} | "
            f"{format_pct(test_index_acc(analysis_payload))} | "
            f"{format_pct(downstream_acc(compare_payload))} |"
        )


if __name__ == "__main__":
    main()
