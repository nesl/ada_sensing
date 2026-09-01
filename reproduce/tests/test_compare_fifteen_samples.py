from __future__ import annotations

import unittest

from reproduce_in_ae.compare_fifteen_samples import (
    _aggregate_model,
    _macro_row,
    _sample_balanced_accuracy,
)


class CompareFifteenSamplesTests(unittest.TestCase):
    def test_sample_balanced_aggregation(self) -> None:
        rows = []
        for sample in range(15):
            for light in range(2):
                rows.append(
                    {
                        "sample_id": f"s{sample}",
                        "lighting_id": f"l{light}",
                        "score": 1.0 if sample == 0 else 0.0,
                    }
                )
        self.assertAlmostEqual(
            _sample_balanced_accuracy(rows, "score", expected_lights=2),
            100.0 / 15.0,
        )

    def test_aggregate_and_macro(self) -> None:
        ae = []
        manual = []
        for sample in range(15):
            for light in range(2):
                for shot in range(3):
                    ae.append(
                        {
                            "sample_id": f"s{sample}",
                            "light_id": f"l{light}",
                            "correct": shot == 0,
                        }
                    )
                for parameter in range(1, 4):
                    manual.append(
                        {
                            "sample_id": f"s{sample}",
                            "light_id": f"l{light}",
                            "parameter_key": f"p{parameter:03d}",
                            "correct": parameter == 1,
                            "top1_confidence": 1.0 if parameter == 1 else 0.0,
                        }
                    )
        row, conditions = _aggregate_model(
            model="test",
            paper_name="Test",
            ae_records=ae,
            manual_records=manual,
            expected_lights=2,
            ae_count=3,
            manual_count=3,
            parameter_field="parameter_key",
        )
        self.assertEqual(len(conditions), 30)
        self.assertAlmostEqual(row["ae_top1_accuracy"], 100.0 / 3.0)
        self.assertAlmostEqual(row["lens_top1_accuracy"], 100.0)
        self.assertAlmostEqual(row["oracle_f_top1_accuracy"], 100.0)
        macro = _macro_row([row, row])
        self.assertAlmostEqual(macro["ae_top1_accuracy"], 100.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
