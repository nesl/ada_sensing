from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reproduce_in_ae.analyze_luminance import analyze_one, shot_stability
from reproduce_in_ae.exposure import (
    ExposureSpec,
    apply_exposure,
    default_specs,
    exposure_gain,
    linear_luminance,
    linear_rgb_to_srgb_u8,
    mean_linear_luminance,
    srgb_u8_to_linear,
)
from reproduce_in_ae.evaluate_exposure import result_is_complete


class ExposureTests(unittest.TestCase):
    def test_default_grid_has_29_unique_points(self) -> None:
        specs = default_specs()
        self.assertEqual(len(specs), 29)
        self.assertEqual(len({spec.tag for spec in specs}), 29)
        self.assertIn(ExposureSpec("fixed_ev", 0.0), specs)

    def test_srgb_linear_endpoints_and_rec709_weights(self) -> None:
        rgb = np.asarray([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        linear = srgb_u8_to_linear(rgb)
        np.testing.assert_allclose(linear[0, 0], 0.0)
        np.testing.assert_allclose(linear[0, 1], 1.0)
        luminance = linear_luminance(
            np.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32)
        )
        np.testing.assert_allclose(luminance, [[0.2126, 0.7152]], rtol=1e-6)

    def test_linear_srgb_round_trip_is_exact_for_uint8(self) -> None:
        values = np.arange(256, dtype=np.uint8)
        rgb = np.stack([values, values, values], axis=-1)[None, ...]
        reconstructed = linear_rgb_to_srgb_u8(srgb_u8_to_linear(rgb))
        np.testing.assert_array_equal(reconstructed, rgb)

    def test_zero_ev_is_pixel_exact_noop(self) -> None:
        array = np.asarray(
            [[[0, 12, 255], [37, 128, 240]], [[1, 2, 3], [250, 251, 252]]],
            dtype=np.uint8,
        )
        image = Image.fromarray(array, mode="RGB")
        adjusted, metadata = apply_exposure(
            image, ExposureSpec("fixed_ev", 0.0)
        )
        np.testing.assert_array_equal(np.asarray(adjusted), array)
        self.assertEqual(metadata.gain, 1.0)
        self.assertEqual(metadata.effective_ev, 0.0)

    def test_one_ev_doubles_unsaturated_linear_values(self) -> None:
        original_linear = np.full((2, 2, 3), 0.1, dtype=np.float32)
        image = Image.fromarray(linear_rgb_to_srgb_u8(original_linear), mode="RGB")
        original_mean = mean_linear_luminance(np.asarray(image))
        adjusted, metadata = apply_exposure(
            image,
            ExposureSpec("fixed_ev", 1.0),
            current_mean_luminance=original_mean,
        )
        achieved = mean_linear_luminance(np.asarray(adjusted))
        self.assertAlmostEqual(metadata.gain, 2.0)
        self.assertAlmostEqual(achieved, 2.0 * original_mean, delta=0.004)

    def test_target_gain_uses_full_image_mean(self) -> None:
        self.assertAlmostEqual(
            exposure_gain(ExposureSpec("target_mean_luminance", 0.2), 0.1),
            2.0,
        )
        self.assertAlmostEqual(
            exposure_gain(ExposureSpec("target_mean_luminance", 0.2), 0.4),
            0.5,
        )

    def test_full_image_analysis_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gray.JPEG"
            Image.new("RGB", (17, 11), color=(128, 128, 128)).save(
                path, quality=100, subsampling=0
            )
            row, histogram = analyze_one(
                (
                    "dataset",
                    "l1",
                    "param_1",
                    "n00000001",
                    "gray",
                    str(path),
                )
            )
        self.assertEqual(row["width"], 17)
        self.assertEqual(row["height"], 11)
        self.assertEqual(row["pixel_count"], 187)
        self.assertAlmostEqual(row["std_luminance"], 0.0, places=7)
        self.assertEqual(int(histogram.sum()), 187)

    def test_five_shot_stability_requires_and_summarizes_five(self) -> None:
        rows = [
            {
                "dataset": "dataset",
                "environment": "l1",
                "class_id": "n00000001",
                "image_id": "image",
                "mean_luminance": value,
            }
            for value in (0.1, 0.11, 0.09, 0.1, 0.1)
        ]
        result = shot_stability(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["shot_count"], 5)
        self.assertAlmostEqual(result[0]["mean_luminance"], 0.1)

    def test_result_completion_rejects_wrong_expected_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "result.json"
            npz_path = Path(directory) / "result.npz"
            json_path.write_text('{"total": 2}', encoding="utf-8")
            np.savez_compressed(
                npz_path,
                targets=np.asarray([0, 1], dtype=np.uint16),
                predictions=np.asarray([0, 1], dtype=np.uint16),
                hits=np.asarray([True, True], dtype=np.bool_),
            )
            self.assertTrue(result_is_complete(json_path, npz_path, 2))
            self.assertFalse(result_is_complete(json_path, npz_path, 3))


if __name__ == "__main__":
    unittest.main()
