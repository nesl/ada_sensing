from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from reproduce_in_ae.compare_references import compare
from reproduce_in_ae.datasets import DatasetRoots, build_dataset, paper_transform
from reproduce_in_ae.evaluate import output_indices
from reproduce_in_ae.merge_shards import merge
from reproduce_in_ae.protocol import (
    DATASET_IN,
    MODEL_BY_KEY,
    MODEL_SPECS,
    PAPER_CROP_SIZE,
    parse_dataset_names,
    parse_model_keys,
)


class ProtocolTests(unittest.TestCase):
    def test_exact_paper_model_coverage(self) -> None:
        self.assertEqual(len(MODEL_SPECS), 12)
        self.assertEqual(len(MODEL_BY_KEY), 12)
        self.assertIn("swin_v2_s", MODEL_BY_KEY)
        self.assertIn("resnet50_deepaugment_augmix", MODEL_BY_KEY)

    def test_parser_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_keys("not-a-paper-model")
        with self.assertRaises(ValueError):
            parse_dataset_names("not-a-paper-dataset")

    def test_paper_transform_shape(self) -> None:
        image = Image.new("RGB", (640, 480), color=(10, 20, 30))
        tensor = paper_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, PAPER_CROP_SIZE, PAPER_CROP_SIZE))

    def test_closed_set_class_order_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "in"
            for wnid in ("n00000002", "n00000001"):
                class_dir = root / wnid
                class_dir.mkdir(parents=True)
                Image.new("RGB", (300, 260)).save(class_dir / "x.JPEG")
            roots = DatasetRoots(root, Path(directory) / "missing1", Path(directory) / "missing2")
            dataset = build_dataset(DATASET_IN, roots)
            self.assertEqual(dataset.classes, ["n00000001", "n00000002"])

    def test_closed_set_indices_follow_dataset_class_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            class_index = Path(directory) / "imagenet_class_index.json"
            class_index.write_text(
                json.dumps(
                    {
                        "0": ["n00000002", "second"],
                        "7": ["n00000001", "first"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                output_indices(["n00000001", "n00000002"], class_index),
                [7, 0],
            )

    def test_reference_comparison_uses_content_not_only_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original"
            diverse = Path(directory) / "diverse"
            for root in (original, diverse):
                (root / "n00000001").mkdir(parents=True)
                (root / "n00000001" / "x.JPEG").write_bytes(b"same")
            self.assertTrue(compare(original, diverse)["same_bytes_at_relative_paths"])
            (diverse / "n00000001" / "x.JPEG").write_bytes(b"different")
            result = compare(original, diverse)
            self.assertFalse(result["same_bytes_at_relative_paths"])
            self.assertFalse(result["same_content_multiset_per_class"])

    def test_deterministic_shards_merge_by_counts(self) -> None:
        template = {
            "model": "dinov2_g",
            "dataset": "ae_imagenet_es_diverse",
            "correct": 1,
            "total": 2,
            "full_dataset_total": 4,
            "micro_accuracy": 50.0,
            "macro_setting_accuracy": 50.0,
            "paper_value": 62.8,
            "paper_rounding_match": False,
            "per_setting": {
                "env/shot": {"correct": 1, "total": 2, "accuracy": 50.0}
            },
            "elapsed_seconds": 1.0,
            "model_provenance": {"device": "cuda:0"},
            "protocol": {"batch_size": 2, "expected_setting_counts": {"env/shot": 2}},
            "shard": {"index": 0, "count": 2},
        }
        second = deepcopy(template)
        second["correct"] = 2
        second["per_setting"]["env/shot"] = {
            "correct": 2,
            "total": 2,
            "accuracy": 100.0,
        }
        second["model_provenance"]["device"] = "cuda:1"
        second["shard"]["index"] = 1
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "shard0.json", Path(directory) / "shard1.json"]
            paths[0].write_text(json.dumps(template), encoding="utf-8")
            paths[1].write_text(json.dumps(second), encoding="utf-8")
            result = merge(paths)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["micro_accuracy"], 75.0)
        self.assertEqual(result["macro_setting_accuracy"], 75.0)


if __name__ == "__main__":
    unittest.main()
