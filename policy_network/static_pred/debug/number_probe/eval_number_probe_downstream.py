from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]
LENS_DIR = ROOT / "lens"
NUMBER_PROBE_DIR = Path(__file__).resolve().parent

for extra_path in (ROOT, LENS_DIR, NUMBER_PROBE_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model
from train_number_probe import (
    DROPOUT,
    ENCODING_DIM,
    HIDDEN_DIM,
    MAX_PERIOD,
    NUM_CANDIDATES,
    NUM_LAYERS,
    NumberProbeDataset,
    NumberProbeMLP,
    get_label_paths,
    load_json,
)


FEATURE_MODES = ("lightning_class", "lightning", "class")
DEFAULT_MANIFEST = ROOT / "data" / "ImageNet-ES-Diverse" / "manifest_all.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate downstream test accuracy for number-probe checkpoints."
    )
    parser.add_argument("--label_kind", type=str, choices=["oracle", "policy"], default="oracle")
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Defaults to policy_network/results_number_probe/{label_kind}.",
    )
    parser.add_argument(
        "--feature_mode",
        type=str,
        choices=[*FEATURE_MODES, "all"],
        default="all",
    )
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_manifest_index(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    dataset = ManifestLensDataset(manifest_path)
    return {str(item["id"]): item for item in dataset.items}


def build_candidate_tensor(
    candidates: List[Dict[str, Any]],
    transform,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    option_id_to_pos = {
        int(candidate["option_id"]): pos for pos, candidate in enumerate(candidates)
    }
    return torch.stack(images, dim=0), option_id_to_pos


def load_probe(feature_mode: str, results_root: Path, device: torch.device) -> Tuple[NumberProbeMLP, Dict[str, int], Dict[str, int], Path]:
    checkpoint_path = results_root / feature_mode / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NumberProbeMLP(
        feature_mode=feature_mode,
        encoding_dim=ENCODING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_candidates=NUM_CANDIDATES,
        max_period=MAX_PERIOD,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["env_to_number"], checkpoint["class_to_number"], checkpoint_path


def predict_probe_records(
    model: NumberProbeMLP,
    test_items: List[Dict[str, Any]],
    env_to_number: Dict[str, int],
    class_to_number: Dict[str, int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    dataset = NumberProbeDataset(test_items, env_to_number, class_to_number)
    records = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            model_batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in batch.items()
            }
            logits = model(model_batch)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)
            item = test_items[idx]
            records.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "pred_best_index": int(pred.item()),
                    "top1_confidence": float(confidence.item()),
                    "target_best_index": int(item["best_option_id"]),
                }
            )

    return records


def summarize_accuracy(correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "acc": 100.0 * correct / max(1, total),
    }


def evaluate_downstream(
    prediction_records: List[Dict[str, Any]],
    manifest_index: Dict[str, Dict[str, Any]],
    classifier,
    transform,
    device: torch.device,
) -> Dict[str, Any]:
    total = 0
    downstream_correct = 0
    option_acc_correct = 0
    missing_samples = []
    per_sample = []

    for record in tqdm(prediction_records, desc="Number probe downstream"):
        sample_id = str(record["sample_id"])
        manifest_item = manifest_index.get(sample_id)
        if manifest_item is None:
            missing_samples.append(sample_id)
            continue

        label = int(manifest_item["label"])
        candidates = manifest_item["candidates"]
        candidate_tensor, option_id_to_pos = build_candidate_tensor(candidates, transform)
        pred_option_id = int(record["pred_best_index"])
        if pred_option_id not in option_id_to_pos:
            raise ValueError(
                f"Predicted option_id {pred_option_id} not found in manifest candidates for {sample_id}."
            )

        with torch.no_grad():
            logits = classifier(candidate_tensor.to(device, non_blocking=True))

        selected_pos = option_id_to_pos[pred_option_id]
        selected_logits = logits[selected_pos]
        downstream_pred_label = int(torch.argmax(selected_logits).item())
        downstream_hit = downstream_pred_label == label
        option_hit = pred_option_id == int(record["target_best_index"])

        downstream_correct += int(downstream_hit)
        option_acc_correct += int(option_hit)
        total += 1
        per_sample.append(
            {
                "sample_id": sample_id,
                "label": label,
                "pred_best_index": pred_option_id,
                "target_best_index": int(record["target_best_index"]),
                "probe_option_correct": option_hit,
                "downstream_prediction": downstream_pred_label,
                "downstream_correct": downstream_hit,
            }
        )

    return {
        "evaluated_samples": total,
        "missing_manifest_samples": len(missing_samples),
        "probe_option_acc": summarize_accuracy(option_acc_correct, total),
        "downstream_test_acc": summarize_accuracy(downstream_correct, total),
        "missing_samples": missing_samples,
        "records": per_sample,
    }


def main() -> None:
    args = parse_args()
    results_root = (
        Path(args.results_root)
        if args.results_root is not None
        else ROOT / "policy_network" / "results_number_probe" / args.label_kind
    )
    feature_modes = FEATURE_MODES if args.feature_mode == "all" else (args.feature_mode,)

    _, _, test_json = get_label_paths(args.label_kind)
    test_items = load_json(test_json)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)
    classifier = load_timm_model(args.model, device=device)
    manifest_index = build_manifest_index(args.manifest)

    summary: Dict[str, Any] = {}
    for feature_mode in feature_modes:
        model, env_to_number, class_to_number, checkpoint_path = load_probe(
            feature_mode=feature_mode,
            results_root=results_root,
            device=device,
        )
        prediction_records = predict_probe_records(
            model=model,
            test_items=test_items,
            env_to_number=env_to_number,
            class_to_number=class_to_number,
            device=device,
        )
        result = evaluate_downstream(
            prediction_records=prediction_records,
            manifest_index=manifest_index,
            classifier=classifier,
            transform=transform,
            device=device,
        )
        result["config"] = {
            "label_kind": args.label_kind,
            "feature_mode": feature_mode,
            "checkpoint": str(checkpoint_path),
            "test_json": str(test_json),
            "manifest": args.manifest,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
        }

        output_json = results_root / feature_mode / "downstream_test_acc.json"
        os.makedirs(output_json.parent, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)

        summary[feature_mode] = {
            "probe_option_acc": result["probe_option_acc"],
            "downstream_test_acc": result["downstream_test_acc"],
            "output_json": str(output_json),
        }
        print(f"{feature_mode}:")
        print(json.dumps(summary[feature_mode], indent=2))

    summary_json = results_root / "number_probe_downstream_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_json}")


if __name__ == "__main__":
    main()
