from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reproduce_in_ae.review_replication_roi import ReviewSession, check_approval


class ReplicationRoiManualReviewTests(unittest.TestCase):
    def test_draft_approval_and_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "roi.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "dpi600",
                        "representative_relative_path": "b500/ae/ae_01.jpg",
                        "rois": [
                            {
                                "sample_id": "sample",
                                "zoom_id": "z001",
                                "box": [4, 5, 60, 45],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "review"
            approval = root / "roi.approval.json"
            with mock.patch(
                "reproduce_in_ae.review_replication_roi.EXPECTED_IMAGE_SIZE", (64, 48)
            ):
                session = ReviewSession("dpi600", root / "source", config, output, approval)
                pending = [
                    {
                        "sample_id": "sample",
                        "zoom_id": "z001",
                        "box": [5, 5, 61, 45],
                        "approved": False,
                    }
                ]
                session.save_draft(pending)
                self.assertTrue(session.draft_path.is_file())
                with self.assertRaisesRegex(ValueError, "must be approved"):
                    session.approve(pending)
                pending[0]["approved"] = True
                result = session.approve(pending)

            self.assertEqual(result["approval"]["changed_keys"], ["sample/z001"])
            self.assertEqual(json.loads(config.read_text())["rois"][0]["box"], [5, 5, 61, 45])
            checked = check_approval("dpi600", config, approval)
            self.assertEqual(checked["status"], "approved")
            payload = json.loads(config.read_text())
            payload["rois"][0]["box"] = [6, 5, 62, 45]
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after approval"):
                check_approval("dpi600", config, approval)


if __name__ == "__main__":
    unittest.main()
