from __future__ import annotations

import unittest

from reproduce_in_ae.evaluate_diverse_manual_subset import (
    EXPECTED_LIGHT_IDS,
    EXPECTED_MANUAL_PARAMETERS,
    build_macro_mean,
    build_model_baseline,
)


class DiverseManualSubsetTests(unittest.TestCase):
    def test_builds_acquisition_baselines(self) -> None:
        ae_records = []
        manual_records = []
        for sample_index in range(5):
            sample_id = f"sample_{sample_index}"
            for lighting_id in EXPECTED_LIGHT_IDS:
                for shot in range(1, 6):
                    ae_records.append(
                        {
                            "sample_id": sample_id,
                            "lighting_id": lighting_id,
                            "ae_parameter": f"param_{shot}",
                            "correct": shot == 1,
                        }
                    )
                for parameter_key in EXPECTED_MANUAL_PARAMETERS:
                    manual_records.append(
                        {
                            "sample_id": sample_id,
                            "lighting_id": lighting_id,
                            "parameter_key": parameter_key,
                            "correct": parameter_key == "param_1",
                            "top1_confidence": (
                                0.9 if parameter_key == "param_1" else 0.1
                            ),
                        }
                    )

        result, condition_rows = build_model_baseline(
            "model", "Model", ae_records, manual_records
        )
        self.assertEqual(len(condition_rows), 30)
        self.assertAlmostEqual(result["ae_top1_accuracy"], 20.0)
        self.assertAlmostEqual(result["lens_top1_accuracy"], 100.0)
        self.assertAlmostEqual(result["oracle_s_top1_accuracy"], 100.0)
        self.assertAlmostEqual(result["oracle_f_top1_accuracy"], 100.0)
        self.assertEqual(result["oracle_f_parameter_key"], "param_1")
        self.assertAlmostEqual(result["random_top1_accuracy"], 100.0 / 27.0)

        macro = build_macro_mean([result, result])
        self.assertAlmostEqual(macro["macro_ae_top1_accuracy"], 20.0)
        self.assertAlmostEqual(macro["macro_lens_top1_accuracy"], 100.0)


if __name__ == "__main__":
    unittest.main()
