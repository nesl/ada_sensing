from __future__ import annotations

import unittest

from reproduce_in_ae.analyze_replication import (
    AE_PARAMETER_KEYS,
    MANUAL_PARAMETER_KEYS,
    downstream_baseline_tables,
    parameter_condition_tables,
)


class ReplicationAnalysisTests(unittest.TestCase):
    def _payload(self) -> dict:
        records = []
        for condition_index in range(10):
            sample_id = f"sample_{condition_index // 5}"
            light_id = f"light_{condition_index % 5}"
            for parameter_key in AE_PARAMETER_KEYS + MANUAL_PARAMETER_KEYS:
                if parameter_key in {"ae_01", "p001"}:
                    correct = True
                elif parameter_key == "ae_03":
                    correct = condition_index < 5
                else:
                    correct = False
                confidence = 0.9 if parameter_key == "p001" else 0.5
                if parameter_key == "p002":
                    confidence = 0.99
                records.append(
                    {
                        "model": "test_model",
                        "sample_id": sample_id,
                        "zoom_id": "z001",
                        "light_id": light_id,
                        "parameter_key": parameter_key,
                        "exposure_mode": (
                            "auto" if parameter_key.startswith("ae_") else "manual"
                        ),
                        "ae_shot": (
                            int(parameter_key[-2:])
                            if parameter_key.startswith("ae_")
                            else None
                        ),
                        "aperture": "auto" if parameter_key.startswith("ae_") else 5.0,
                        "shutter_speed": (
                            "auto" if parameter_key.startswith("ae_") else "1/60"
                        ),
                        "iso": "auto" if parameter_key.startswith("ae_") else 250,
                        "top1_confidence": confidence,
                        "correct": correct,
                    }
                )
        # Duplicate the controlled construction for z002 because production functions
        # intentionally enforce both zooms.
        records.extend([{**row, "zoom_id": "z002"} for row in list(records)])
        return {"paper_name": "Test Model", "records": records}

    def test_parameter_success_distribution(self) -> None:
        _scores, distributions = parameter_condition_tables(
            {"test_model": self._payload()}
        )
        row = distributions["z001"][0]
        self.assertEqual(row["parameter_total"], 30)
        self.assertEqual(row["parameters_correct_0"], 27)
        self.assertEqual(row["parameters_correct_5"], 1)
        self.assertEqual(row["parameters_correct_10"], 2)

    def test_downstream_baselines(self) -> None:
        summaries, details = downstream_baseline_tables(
            {"test_model": self._payload()}
        )
        row = summaries["z001"][0]
        self.assertEqual(row["ae_correct"], 15)
        self.assertAlmostEqual(row["ae_top1_accuracy"], 50.0)
        self.assertEqual(row["lens_correct"], 0)
        self.assertEqual(row["oracle_s_correct"], 10)
        self.assertEqual(row["oracle_f_parameter_key"], "p001")
        self.assertEqual(row["oracle_f_correct"], 10)
        self.assertAlmostEqual(row["random_top1_accuracy"], 100.0 / 27.0)
        self.assertEqual(len(details["z001"]), 10)


if __name__ == "__main__":
    unittest.main()
