from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from reproduce_in_ae.audit_replication import audit_model_result
from reproduce_in_ae.evaluate_replication import (
    _valid_completed_result,
    prediction_rows_from_batch,
)
from reproduce_in_ae.prepare_replication import prepare_dataset
from reproduce_in_ae.replication import (
    EXPECTED_CAPTURE_COUNT,
    atomic_csv_dump,
    atomic_json_dump,
    load_cropped_manifest,
    parameter_key,
    sha256_file,
)


SAMPLES = (
    ("ILSVRC2012_val_00034232", 107, "n01910747", "jellyfish"),
    ("ILSVRC2012_val_00038504", 79, "n01784675", "centipede"),
)
ZOOMS = ("z001", "z002")
LIGHTS = (("b010", 10), ("b200", 200), ("b500", 500), ("b700", 700), ("b1000", 1000))


def make_synthetic_capture_tree(root: Path) -> Path:
    config_rows = []
    for sample_id, _index, _wnid, _name in SAMPLES:
        for zoom_id in ZOOMS:
            config_rows.append(
                {"sample_id": sample_id, "zoom_id": zoom_id, "box": [4, 5, 60, 45]}
            )
    config = root / "roi.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_convention": "test",
                "representative_relative_path": "b500/ae/ae_01.jpg",
                "rois": config_rows,
            }
        ),
        encoding="utf-8",
    )

    capture_root = root / "source"
    rows = []
    plan_index = 0
    for sample_id, class_index, wnid, class_name in SAMPLES:
        for zoom_id in ZOOMS:
            for light_id, light_intensity in LIGHTS:
                for ae_shot in (1, 2, 3):
                    plan_index += 1
                    relative = Path(sample_id) / zoom_id / light_id / "ae" / f"ae_{ae_shot:02d}.jpg"
                    path = capture_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (64, 48), (light_intensity % 255, 20, 30)).save(path)
                    rows.append(
                        {
                            "capture_status": "captured",
                            "capture_key": f"{sample_id}|{zoom_id}|{light_id}|ae_{ae_shot:02d}",
                            "image_path": relative.as_posix(),
                            "sample_id": sample_id,
                            "zoom_id": zoom_id,
                            "light_id": light_id,
                            "light_intensity": light_intensity,
                            "light_percent": light_intensity / 10.0,
                            "exposure_mode": "auto",
                            "ae_shot": ae_shot,
                            "parameter_id": None,
                            "parameter_number": None,
                            "aperture": "auto",
                            "shutter_speed": "auto",
                            "iso": "auto",
                            "class_index": class_index,
                            "wnid": wnid,
                            "class_name": class_name,
                            "plan_index": plan_index,
                        }
                    )
                for parameter_number in range(1, 28):
                    plan_index += 1
                    parameter_id = f"p{parameter_number:03d}"
                    relative = Path(sample_id) / zoom_id / light_id / "manual" / f"{parameter_id}.jpg"
                    path = capture_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (64, 48), (parameter_number, 40, 50)).save(path)
                    rows.append(
                        {
                            "capture_status": "captured",
                            "capture_key": f"{sample_id}|{zoom_id}|{light_id}|{parameter_id}",
                            "image_path": relative.as_posix(),
                            "sample_id": sample_id,
                            "zoom_id": zoom_id,
                            "light_id": light_id,
                            "light_intensity": light_intensity,
                            "light_percent": light_intensity / 10.0,
                            "exposure_mode": "manual",
                            "ae_shot": None,
                            "parameter_id": parameter_id,
                            "parameter_number": parameter_number,
                            "aperture": 5.0,
                            "shutter_speed": "1/60",
                            "iso": 250,
                            "class_index": class_index,
                            "wnid": wnid,
                            "class_name": class_name,
                            "plan_index": plan_index,
                        }
                    )
    manifest = capture_root / "captures.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return config


class ReplicationPipelineTests(unittest.TestCase):
    def test_parameter_keys(self) -> None:
        self.assertEqual(parameter_key({"exposure_mode": "auto", "ae_shot": 2}), "ae_02")
        self.assertEqual(
            parameter_key({"exposure_mode": "manual", "parameter_id": "p027"}),
            "p027",
        )

    def test_prepare_dataset_creates_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_synthetic_capture_tree(root)
            output = root / "cropped"
            with mock.patch(
                "reproduce_in_ae.prepare_replication.EXPECTED_IMAGE_SIZE", (64, 48)
            ):
                summary = prepare_dataset(root / "source", output, config)
            rows = load_cropped_manifest(output)
            self.assertEqual(len(rows), EXPECTED_CAPTURE_COUNT)
            self.assertEqual(summary["counts"]["exposure_mode_counts"], {"auto": 60, "manual": 540})
            self.assertEqual(set(summary["parameter_counts"]), {"ae_01", "ae_02", "ae_03", *(f"p{i:03d}" for i in range(1, 28))})
            self.assertTrue(all(row["cropped_image_size"] == [56, 40] for row in rows))
            self.assertEqual(len({row["capture_key"] for row in rows}), 600)

    def test_prediction_rows_use_closed_200_way_top1(self) -> None:
        closed_wnids = [f"n{index:08d}" for index in range(200)]
        closed_wnids[7] = "n01910747"
        output_indices = list(range(200))
        source = {
            "source_manifest_index": 0,
            "capture_key": "capture",
            "sample_id": "sample",
            "zoom_id": "z001",
            "light_id": "b010",
            "light_intensity": 10,
            "light_percent": 1.0,
            "exposure_mode": "auto",
            "parameter_key": "ae_01",
            "ae_shot": 1,
            "parameter_id": None,
            "aperture": "auto",
            "shutter_speed": "auto",
            "iso": "auto",
            "cropped_image_path": "x.jpg",
            "class_index": 7,
            "wnid": "n01910747",
            "class_name": "jellyfish",
        }
        rows = prediction_rows_from_batch(
            model_key="resnet50",
            source_rows=[source],
            row_indices=[0],
            predicted_closed=[7],
            confidences=[0.75],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name={index: f"class_{index}" for index in range(200)},
        )
        self.assertEqual(rows[0]["top1_wnid"], "n01910747")
        self.assertEqual(rows[0]["top1_confidence"], 0.75)
        self.assertTrue(rows[0]["correct"])

    def test_completed_result_validation_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions"
            predictions.mkdir()
            manifest_rows = [{"capture_key": f"capture_{index:03d}"} for index in range(600)]
            records = [
                {
                    "model": "resnet50",
                    "capture_key": row["capture_key"],
                    "top1_closed_index": index % 200,
                    "target_closed_index": index % 200,
                    "top1_confidence": 0.5,
                    "correct": True,
                }
                for index, row in enumerate(manifest_rows)
            ]
            payload = {
                "status": "complete",
                "model": "resnet50",
                "dataset_manifest_sha256": "manifest-hash",
                "total": 600,
                "is_smoke_test": False,
                "records": records,
            }
            atomic_csv_dump(records, predictions / "resnet50.csv")
            atomic_json_dump(payload, predictions / "resnet50.json")
            self.assertTrue(
                _valid_completed_result(
                    predictions / "resnet50.json",
                    "resnet50",
                    "manifest-hash",
                    600,
                    False,
                )
            )
            audit = audit_model_result(
                "resnet50", predictions, manifest_rows, "manifest-hash"
            )
            self.assertEqual(audit["status"], "complete")
            records[10]["correct"] = False
            payload["records"] = records
            atomic_json_dump(payload, predictions / "resnet50.json")
            audit = audit_model_result(
                "resnet50", predictions, manifest_rows, "manifest-hash"
            )
            self.assertEqual(audit["status"], "invalid")
            self.assertIn("record_10_correctness_mismatch", audit["errors"])


if __name__ == "__main__":
    unittest.main()
