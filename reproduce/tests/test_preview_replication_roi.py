from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from reproduce_in_ae.preview_replication_roi import generate_previews


class ReplicationRoiPreviewTests(unittest.TestCase):
    def test_preview_only_writes_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            config = root / "roi.json"
            output = root / "review"
            rois = []
            for sample in ("sample_a", "sample_b"):
                for zoom in ("z001", "z002"):
                    path = source / sample / zoom / "b500" / "ae" / "ae_01.jpg"
                    path.parent.mkdir(parents=True)
                    Image.new("RGB", (6528, 4352), (100, 120, 140)).save(path)
                    rois.append(
                        {
                            "sample_id": sample,
                            "zoom_id": zoom,
                            "box": [100, 200, 500, 600],
                        }
                    )
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "coordinate_convention": "test",
                        "representative_relative_path": "b500/ae/ae_01.jpg",
                        "rois": rois,
                    }
                ),
                encoding="utf-8",
            )

            written = generate_previews(source, config, output)

            self.assertEqual(len(written), 11)
            self.assertTrue((output / "roi_boxes_overview.jpg").is_file())
            self.assertTrue((output / "roi_crops_overview.jpg").is_file())
            self.assertFalse((root / "replicated_capture_roi").exists())
            review = json.loads((output / "roi_config.review.json").read_text())
            self.assertEqual(review["status"], "review_only_not_approved")
            self.assertEqual(len(review["review_rows"]), 4)

    def test_existing_previews_require_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            path = source / "sample" / "z001" / "b500" / "ae" / "ae_01.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (6528, 4352)).save(path)
            config = root / "roi.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "representative_relative_path": "b500/ae/ae_01.jpg",
                        "rois": [
                            {
                                "sample_id": "sample",
                                "zoom_id": "z001",
                                "box": [0, 0, 100, 100],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "review"
            generate_previews(source, config, output)
            with self.assertRaises(FileExistsError):
                generate_previews(source, config, output)


if __name__ == "__main__":
    unittest.main()
