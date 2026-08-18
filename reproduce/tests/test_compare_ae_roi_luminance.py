from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from reproduce_in_ae.compare_ae_roi_luminance import (
    GROUP_CLEAN,
    GROUP_ORIGINAL,
    GROUP_Z001,
    GROUP_Z002,
    Capture,
    analyze_capture,
    discover_captures,
    run_comparison,
    validate_complete_grid,
)
from reproduce_in_ae.exposure import linear_luminance, srgb_u8_to_linear


class CompareAeRoiLuminanceTests(unittest.TestCase):
    def _save_rgb(self, path: Path, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (value, value, value)).save(path)

    def _build_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        original = root / "original"
        clean = root / "clean"
        dpi = root / "dpi600_roi"
        samples = ("sample_1", "sample_2")
        original_lights = ("l1", "l2")
        original_shots = ("param_1", "param_2")
        dpi_lights = ("b010", "b200")
        dpi_shots = ("ae_01", "ae_02")

        for sample_index, sample in enumerate(samples):
            self._save_rgb(
                clean / f"class_{sample_index}" / f"{sample}.JPEG",
                160 + 10 * sample_index,
            )
            for light in original_lights:
                for shot in original_shots:
                    self._save_rgb(
                        original / light / shot / "class" / f"{sample}.JPEG",
                        30 + 10 * sample_index,
                    )

        manifest_rows = []
        for sample_index, sample in enumerate(samples):
            for zoom in ("z001", "z002"):
                for light in dpi_lights:
                    for shot_index, shot in enumerate(dpi_shots, start=1):
                        relative = Path(sample) / zoom / light / "ae" / f"{shot}.jpg"
                        self._save_rgb(dpi / relative, 80 + 10 * sample_index)
                        manifest_rows.append(
                            {
                                "capture_key": f"{sample}|{zoom}|{light}|{shot}",
                                "sample_id": sample,
                                "zoom_id": zoom,
                                "light_id": light,
                                "exposure_mode": "auto",
                                "ae_shot": shot_index,
                                "parameter_key": shot,
                                "cropped_image_path": relative.as_posix(),
                            }
                        )
                    manual_relative = Path(sample) / zoom / light / "manual" / "p001.jpg"
                    self._save_rgb(dpi / manual_relative, 255)
                    manifest_rows.append(
                        {
                            "capture_key": f"{sample}|{zoom}|{light}|p001",
                            "sample_id": sample,
                            "zoom_id": zoom,
                            "light_id": light,
                            "exposure_mode": "manual",
                            "parameter_key": "p001",
                            "cropped_image_path": manual_relative.as_posix(),
                        }
                    )
        with (dpi / "crop_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row) + "\n")
        return original, clean, dpi

    def test_linear_luminance_for_black_gray_and_white(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {}
            for value in (0, 128, 255):
                path = root / f"{value}.png"
                self._save_rgb(path, value)
                capture = Capture(GROUP_ORIGINAL, "sample", "l1", "param_1", path)
                expected[value] = analyze_capture(capture)["mean_luminance"]
            self.assertAlmostEqual(expected[0], 0.0)
            self.assertAlmostEqual(expected[255], 1.0)
            gray = np.full((1, 1, 3), 128, dtype=np.uint8)
            target = float(linear_luminance(srgb_u8_to_linear(gray))[0, 0])
            self.assertAlmostEqual(expected[128], target)

    def test_complete_grid_rejects_missing_and_duplicate_captures(self) -> None:
        captures = [
            Capture(GROUP_ORIGINAL, "sample", "l1", "param_1", Path("one")),
            Capture(GROUP_ORIGINAL, "sample", "l1", "param_1", Path("two")),
        ]
        with self.assertRaisesRegex(ValueError, "expected 1 unique captures"):
            validate_complete_grid(
                captures,
                GROUP_ORIGINAL,
                ("sample",),
                ("l1",),
                ("param_1",),
            )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_complete_grid(
                captures[:1],
                GROUP_ORIGINAL,
                ("sample",),
                ("l1", "l2"),
                ("param_1",),
            )

    def test_end_to_end_filters_manual_and_writes_normalized_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, clean, dpi = self._build_fixture(root)
            output = root / "output"
            summary = run_comparison(
                original_root=original,
                clean_source_root=clean,
                dpi600_roi_root=dpi,
                output_dir=output,
                bin_count=8,
                expected_common_samples=2,
                original_lighting_ids=("l1", "l2"),
                original_shot_ids=("param_1", "param_2"),
                dpi600_lighting_ids=("b010", "b200"),
                dpi600_shot_ids=("ae_01", "ae_02"),
            )
            self.assertEqual(summary["common_samples"], ["sample_1", "sample_2"])
            self.assertIn("Gaussian kernel", summary["distribution"]["method"])
            self.assertEqual(summary["groups"][GROUP_CLEAN]["capture_count"], 2)
            for group in (GROUP_ORIGINAL, GROUP_Z001, GROUP_Z002):
                self.assertEqual(summary["groups"][group]["capture_count"], 8)
            edges = np.asarray(summary["distribution"]["bin_edges"])
            widths = np.diff(edges)
            for density in summary["distribution"]["normalized_density"].values():
                self.assertTrue(np.all(np.asarray(density) >= 0.0))
                self.assertAlmostEqual(float(np.sum(np.asarray(density) * widths)), 1.0)
            for filename in (
                "ae_roi_luminance_distribution.png",
                "per_capture_luminance.csv",
                "group_summary.csv",
                "summary.json",
                "report.md",
            ):
                self.assertTrue((output / filename).is_file())

            captures, _samples = discover_captures(
                original,
                clean,
                dpi,
                expected_common_samples=2,
                original_lighting_ids=("l1", "l2"),
                original_shot_ids=("param_1", "param_2"),
                dpi600_lighting_ids=("b010", "b200"),
                dpi600_shot_ids=("ae_01", "ae_02"),
            )
            self.assertEqual(len(captures), 26)

    def test_missing_original_condition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, clean, dpi = self._build_fixture(root)
            missing = original / "l2" / "param_2" / "class" / "sample_1.JPEG"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete original_diverse"):
                discover_captures(
                    original,
                    clean,
                    dpi,
                    expected_common_samples=2,
                    original_lighting_ids=("l1", "l2"),
                    original_shot_ids=("param_1", "param_2"),
                    dpi600_lighting_ids=("b010", "b200"),
                    dpi600_shot_ids=("ae_01", "ae_02"),
                )


if __name__ == "__main__":
    unittest.main()
