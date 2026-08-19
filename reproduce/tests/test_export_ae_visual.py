from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from reproduce_in_ae.export_ae_visual import export_ae_visual


class ExportAeVisualTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path | str]:
        sample_id = "sample_001"
        wnid = "n00000001"
        labels_csv = root / "manual" / "labels.csv"
        labels_csv.parent.mkdir(parents=True)
        with labels_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "original_path",
                    "wnid",
                    "source_relative_path",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "original_path": f"original/{sample_id}.JPEG",
                    "wnid": wnid,
                    "source_relative_path": f"{wnid}/{sample_id}.JPEG",
                }
            )

        original = labels_csv.parent / "original" / f"{sample_id}.JPEG"
        dpi = root / "dpi600_roi"
        replicated = root / "replicated_roi"
        auto = root / "auto_exposure"
        sources = [original]
        for capture_root in (dpi, replicated):
            for zoom in ("z001", "z002"):
                sources.append(
                    capture_root
                    / sample_id
                    / zoom
                    / "b700"
                    / "ae"
                    / "ae_03.jpg"
                )
        sources.append(auto / "l2" / "param_3" / wnid / f"{sample_id}.JPEG")
        for index, path in enumerate(sources, start=1):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"source-{index}".encode())
        return {
            "sample_id": sample_id,
            "labels_csv": labels_csv,
            "dpi": dpi,
            "replicated": replicated,
            "auto": auto,
        }

    def test_exports_six_byte_identical_files_with_configurable_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            output = root / "output"
            result = export_ae_visual(
                str(fixture["sample_id"]),
                output,
                labels_csv=Path(fixture["labels_csv"]),
                dpi600_roi_root=Path(fixture["dpi"]),
                replicated_roi_root=Path(fixture["replicated"]),
                imagenet_es_auto_root=Path(fixture["auto"]),
                light_id="b700",
                ae_shot=3,
                reference_light_id="l2",
            )

            self.assertEqual(len(result.images), 6)
            self.assertEqual(
                [item.output.name for item in result.images],
                [
                    "1_original_imagenet_no_resize.JPEG",
                    "2_dpi600_roi_zoom1_b700_ae_03.jpg",
                    "3_dpi600_roi_zoom2_b700_ae_03.jpg",
                    "4_replicated_roi_zoom1_b700_ae_03.jpg",
                    "5_replicated_roi_zoom2_b700_ae_03.jpg",
                    "6_imagenet_es_diverse_test_l2_ae_03.JPEG",
                ],
            )
            for item in result.images:
                self.assertEqual(item.output.read_bytes(), item.source.read_bytes())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            kwargs = {
                "labels_csv": Path(fixture["labels_csv"]),
                "dpi600_roi_root": Path(fixture["dpi"]),
                "replicated_roi_root": Path(fixture["replicated"]),
                "imagenet_es_auto_root": Path(fixture["auto"]),
                "light_id": "b700",
                "ae_shot": 3,
                "reference_light_id": "l2",
            }
            export_ae_visual(str(fixture["sample_id"]), root / "output", **kwargs)
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                export_ae_visual(str(fixture["sample_id"]), root / "output", **kwargs)

    def test_unknown_sample_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "not present"):
                export_ae_visual(
                    "unknown",
                    root / "output",
                    labels_csv=Path(fixture["labels_csv"]),
                )


if __name__ == "__main__":
    unittest.main()
