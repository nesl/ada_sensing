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
    merge_records,
    prediction_rows_from_batch,
    reusable_records,
    resume_source_rows,
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

    def test_prepare_dataset_recrops_same_size_roi_change_and_archives_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_synthetic_capture_tree(root)
            source_path = (
                root
                / "source"
                / SAMPLES[0][0]
                / "z001"
                / "b500"
                / "ae"
                / "ae_01.jpg"
            )
            gradient = Image.new("RGB", (64, 48))
            gradient.putdata([(x * 3, y * 4, (x + y) * 2) for y in range(48) for x in range(64)])
            gradient.save(source_path, quality=100, subsampling=0)
            output = root / "cropped"
            with mock.patch(
                "reproduce_in_ae.prepare_replication.EXPECTED_IMAGE_SIZE", (64, 48)
            ):
                first = prepare_dataset(root / "source", output, config)
                first_crop = (output / source_path.relative_to(root / "source")).read_bytes()
                payload = json.loads(config.read_text(encoding="utf-8"))
                payload["rois"][0]["box"] = [5, 5, 61, 45]
                config.write_text(json.dumps(payload), encoding="utf-8")
                prepare_dataset(root / "source", output, config)
            second_crop = (output / source_path.relative_to(root / "source")).read_bytes()
            self.assertNotEqual(first_crop, second_crop)
            history = output / "manifest_history" / f"{first['crop_manifest_sha256']}.jsonl"
            self.assertTrue(history.is_file())
            current = load_cropped_manifest(output)
            changed = next(row for row in current if row["capture_key"].endswith("b500|ae_01") and row["sample_id"] == SAMPLES[0][0] and row["zoom_id"] == "z001")
            self.assertEqual(changed["roi_box"], [5, 5, 61, 45])

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

    def test_resume_reuses_subset_and_merges_in_current_manifest_order(self) -> None:
        closed_wnids = [f"n{index:08d}" for index in range(200)]
        closed_wnids[7] = "n01910747"
        output_indices = list(range(200))

        def source(capture_key: str, source_index: int) -> dict:
            return {
                "source_manifest_index": source_index,
                "capture_key": capture_key,
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
                "cropped_image_path": f"{capture_key}.jpg",
                "source_image_path": f"source/{capture_key}.jpg",
                "source_sha256": f"source-{capture_key}",
                "roi_box": [1, 2, 30, 40],
                "cropped_image_size": [29, 38],
                "cropped_sha256": f"crop-{capture_key}",
                "class_index": 7,
                "wnid": "n01910747",
                "class_name": "jellyfish",
            }

        source_rows = [source("old", 0), source("new", 1)]
        old_record = prediction_rows_from_batch(
            model_key="resnet50",
            source_rows=source_rows,
            row_indices=[0],
            predicted_closed=[7],
            confidences=[0.75],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name={index: f"class_{index}" for index in range(200)},
        )[0]
        new_record = prediction_rows_from_batch(
            model_key="resnet50",
            source_rows=source_rows,
            row_indices=[1],
            predicted_closed=[7],
            confidences=[0.8],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name={index: f"class_{index}" for index in range(200)},
        )[0]
        reused = reusable_records(
            {"records": [old_record]},
            model_key="resnet50",
            source_rows=source_rows,
            closed_wnids=closed_wnids,
            output_indices=output_indices,
        )
        merged = merge_records(source_rows, reused, [new_record])
        self.assertEqual([row["capture_key"] for row in merged], ["old", "new"])
        self.assertEqual([row["source_manifest_index"] for row in merged], [0, 1])
        self.assertEqual(merged[0]["top1_confidence"], 0.75)
        self.assertEqual(merged[1]["top1_confidence"], 0.8)

    def test_resume_skips_records_when_roi_crop_identity_changed(self) -> None:
        closed_wnids = [f"n{index:08d}" for index in range(200)]
        closed_wnids[7] = "n01910747"
        output_indices = list(range(200))
        historical = {
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
            "source_image_path": "source.jpg",
            "source_sha256": "source-hash",
            "roi_box": [1, 2, 30, 40],
            "cropped_image_path": "crop.jpg",
            "cropped_image_size": [29, 38],
            "cropped_sha256": "old-crop",
            "class_index": 7,
            "wnid": "n01910747",
            "class_name": "jellyfish",
        }
        record = prediction_rows_from_batch(
            model_key="resnet50",
            source_rows=[historical],
            row_indices=[0],
            predicted_closed=[7],
            confidences=[0.75],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name={index: f"class_{index}" for index in range(200)},
        )[0]
        current = dict(historical)
        current["roi_box"] = [2, 2, 31, 40]
        current["cropped_sha256"] = "new-crop"
        reused = reusable_records(
            {"records": [record]},
            model_key="resnet50",
            source_rows=[current],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            resume_source_rows=[historical],
        )
        self.assertEqual(reused, {})

    def test_resume_manifest_history_is_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "manifest_history"
            history.mkdir()
            prior = history / "bad-hash.jsonl"
            prior.write_text('{"capture_key":"old"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resume_source_rows(
                    {"dataset_manifest_sha256": "bad-hash"},
                    root,
                    [],
                    "current-hash",
                )

    def test_resume_rejects_changed_capture_metadata(self) -> None:
        closed_wnids = [f"n{index:08d}" for index in range(200)]
        closed_wnids[7] = "n01910747"
        output_indices = list(range(200))
        source_row = {
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
            "cropped_image_path": "capture.jpg",
            "class_index": 7,
            "wnid": "n01910747",
            "class_name": "jellyfish",
        }
        record = prediction_rows_from_batch(
            model_key="resnet50",
            source_rows=[source_row],
            row_indices=[0],
            predicted_closed=[7],
            confidences=[0.75],
            closed_wnids=closed_wnids,
            output_indices=output_indices,
            index_to_name={index: f"class_{index}" for index in range(200)},
        )[0]
        changed = dict(source_row)
        changed["light_intensity"] = 200
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            reusable_records(
                {"records": [record]},
                model_key="resnet50",
                source_rows=[changed],
                closed_wnids=closed_wnids,
                output_indices=output_indices,
            )

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
