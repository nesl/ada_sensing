from __future__ import annotations

import unittest

from reproduce_in_ae.compare_replication import (
    GROUP_ORDER,
    build_baseline_rows,
    build_delta_rows,
    build_overview_rows,
)


MODELS = ("resnet50", "resnet152")


class CompareReplicationTests(unittest.TestCase):
    def _baseline_row(
        self,
        model: str,
        ae_accuracy: float,
        oracle_f_parameter: str,
        *,
        conditions: int = 10,
    ) -> dict:
        model_offset = 2.0 if model == "resnet152" else 0.0
        ae = ae_accuracy + model_offset
        return {
            "model": model,
            "paper_name": "ResNet-152" if model == "resnet152" else "ResNet-50",
            "ae_correct": round(ae * 3 * conditions / 100),
            "ae_total": 3 * conditions,
            "ae_top1_accuracy": ae,
            "lens_correct": round((ae - 10.0) * conditions / 100),
            "lens_total": conditions,
            "lens_top1_accuracy": ae - 10.0,
            "oracle_s_correct": round((ae + 20.0) * conditions / 100),
            "oracle_s_total": conditions,
            "oracle_s_top1_accuracy": ae + 20.0,
            "oracle_f_parameter_key": oracle_f_parameter,
            "oracle_f_correct": round((ae + 10.0) * conditions / 100),
            "oracle_f_total": conditions,
            "oracle_f_top1_accuracy": ae + 10.0,
            "random_expected_correct": (ae - 20.0) * conditions / 100,
            "random_total": conditions,
            "random_top1_accuracy": ae - 20.0,
        }

    def _summaries(self) -> dict:
        settings = {
            ("replicated_capture", "z001"): (60.0, "p001"),
            ("replicated_capture", "z002"): (70.0, "p002"),
            ("dpi600", "z001"): (40.0, "p003"),
            ("dpi600", "z002"): (50.0, "p004"),
        }
        summaries = {}
        for dataset in ("replicated_capture", "dpi600"):
            tables = {}
            for zoom_id in ("z001", "z002"):
                ae, parameter = settings[(dataset, zoom_id)]
                tables[zoom_id] = [
                    self._baseline_row(model, ae, parameter) for model in MODELS
                ]
            summaries[dataset] = {"downstream_baselines": tables}
        return summaries

    def test_builds_four_groups_with_gaps_and_stable_order(self) -> None:
        rows = build_baseline_rows(
            self._summaries(), MODELS, expected_conditions=10
        )
        self.assertEqual(len(rows), 4 * len(MODELS))
        self.assertEqual(
            [row["group_id"] for row in rows[:: len(MODELS)]], list(GROUP_ORDER)
        )
        self.assertEqual([row["model"] for row in rows[:2]], list(MODELS))
        first = rows[0]
        self.assertEqual(first["oracle_f_parameter_key"], "p001")
        self.assertAlmostEqual(first["ae_minus_lens_pp"], 10.0)
        self.assertAlmostEqual(first["ae_minus_oracle_s_pp"], -20.0)
        self.assertAlmostEqual(first["ae_minus_oracle_f_pp"], -10.0)
        self.assertAlmostEqual(first["ae_minus_random_pp"], 20.0)

    def test_macro_means_and_delta_direction(self) -> None:
        rows = build_baseline_rows(
            self._summaries(), MODELS, expected_conditions=10
        )
        overview = build_overview_rows(rows, MODELS)
        self.assertAlmostEqual(overview[0]["macro_ae_top1_accuracy"], 61.0)

        deltas = build_delta_rows(rows, overview, MODELS)
        macro = {
            row["comparison_id"]: row
            for row in deltas
            if row["model"] == "macro_mean"
        }
        self.assertAlmostEqual(
            macro["scaled_zoom_z002_minus_z001"]["delta_ae_top1_accuracy_pp"],
            10.0,
        )
        self.assertAlmostEqual(
            macro["z001_bitmap_minus_scaled"]["delta_ae_top1_accuracy_pp"],
            -20.0,
        )
        self.assertAlmostEqual(
            macro["scaled_zoom_z002_minus_z001"]["delta_ae_minus_lens_pp"],
            0.0,
        )

    def test_rejects_missing_zoom(self) -> None:
        summaries = self._summaries()
        del summaries["dpi600"]["downstream_baselines"]["z002"]
        with self.assertRaisesRegex(ValueError, "expected baseline zooms"):
            build_baseline_rows(summaries, MODELS, expected_conditions=10)

    def test_rejects_missing_model(self) -> None:
        summaries = self._summaries()
        summaries["replicated_capture"]["downstream_baselines"]["z001"].pop()
        with self.assertRaisesRegex(ValueError, "baseline models do not match"):
            build_baseline_rows(summaries, MODELS, expected_conditions=10)

    def test_rejects_incomplete_condition_total(self) -> None:
        summaries = self._summaries()
        row = summaries["replicated_capture"]["downstream_baselines"]["z001"][0]
        row["lens_total"] = 9
        with self.assertRaisesRegex(ValueError, "invalid lens_total"):
            build_baseline_rows(summaries, MODELS, expected_conditions=10)


if __name__ == "__main__":
    unittest.main()
