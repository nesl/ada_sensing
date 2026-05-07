from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LENS_DIR = ROOT / "lens"
POLICY_DIR = ROOT / "policy_network" / "static_pred"
OPENCLIP_DIR = ROOT / "openclip_ds_policy"

for extra_path in (ROOT, LENS_DIR, POLICY_DIR, OPENCLIP_DIR):
    path = str(extra_path)
    if path not in sys.path:
        sys.path.insert(0, path)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb
from openclip200 import (
    DEFAULT_PROMPT_TEMPLATE,
    OpenCLIP200Classifier,
    filter_manifest_items,
    get_subset_label_ids,
    save_json,
    summarize_binary_hits,
)
from policy_dataset import PolicyDataset
from policy_model import (
    SensorPolicyNetwork,
    infer_backbone_name_from_checkpoint,
    infer_input_mode_from_checkpoint,
    infer_num_input_views_from_checkpoint,
    normalize_policy_checkpoint_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy-selected candidate with OpenCLIP-200 top-1.")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/manifest_all.json"))
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--class_index_json", type=str, default=str(ROOT / "data/ImageNet-ES-Diverse/imagenet_class_index.json"))
    parser.add_argument("--openclip_model", type=str, default="ViT-B-32")
    parser.add_argument("--openclip_pretrained", type=str, default="openai")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_manifest_index(manifest_path: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    dataset = ManifestLensDataset(manifest_path)
    return {str(item["id"]): item for item in dataset.items}, dataset.items


def load_policy_predictions(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    input_variant = checkpoint.get("input_variant") or "real"
    dataset = PolicyDataset(
        args.data_json,
        transform=imagenet_preprocess(args.image_size),
        manifest_path=args.manifest,
        input_mode=input_mode,
        env_option_id=checkpoint.get("env_option_id"),
        env_option_ids=checkpoint.get("env_option_ids"),
        include_ae_input=bool(checkpoint.get("include_ae_input", False)),
        input_variant=input_variant,
        ae_input_variant=checkpoint.get("ae_input_variant") or input_variant,
        env_input_variant=checkpoint.get("env_input_variant") or "real",
        single_input_source=checkpoint.get("single_input_source") or "baseline",
        noise_seed=int(checkpoint.get("noise_seed", 0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    backbone_name = infer_backbone_name_from_checkpoint(checkpoint)
    state_dict = normalize_policy_checkpoint_state_dict(
        checkpoint["model_state_dict"],
        backbone_name,
    )
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
        backbone_name=backbone_name,
        input_mode=input_mode,
        num_input_views=infer_num_input_views_from_checkpoint(checkpoint),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Policy inference"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            for sample_id, pred, conf in zip(batch["sample_id"], preds, confs):
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "pred_best_index": int(pred.item()),
                        "policy_top1_confidence": float(conf.item()),
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest_index, all_items = build_manifest_index(args.manifest)
    label_ids = get_subset_label_ids(all_items)
    classifier = OpenCLIP200Classifier(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
        class_index_json=args.class_index_json,
        label_ids=label_ids,
        device=device,
        prompt_template=args.prompt_template,
    )
    allowed_items = filter_manifest_items(all_items, args.data_json)
    allowed_ids = {str(item["id"]) for item in allowed_items}
    prediction_records = [
        record for record in load_policy_predictions(args, device)
        if str(record["sample_id"]) in allowed_ids
    ]

    hits: List[int] = []
    per_sample: List[Dict[str, Any]] = []
    missing: List[str] = []

    for record in tqdm(prediction_records, desc="OpenCLIP downstream top1"):
        sample_id = str(record["sample_id"])
        item = manifest_index.get(sample_id)
        if item is None:
            missing.append(sample_id)
            continue
        pred_option_id = int(record["pred_best_index"])
        option_to_candidate = {
            int(candidate["option_id"]): candidate
            for candidate in item["candidates"]
        }
        if pred_option_id not in option_to_candidate:
            raise ValueError(f"Predicted option_id={pred_option_id} missing for sample_id={sample_id}")

        selected = option_to_candidate[pred_option_id]
        image_tensor = classifier.preprocess_images([load_image_rgb(selected["path"])])
        similarity = classifier.similarity_from_images(image_tensor)
        pred_raw = int(classifier.pred_raw_labels(similarity)[0].item())
        raw_label = int(item["label"])
        hit = int(pred_raw == raw_label)
        hits.append(hit)
        per_sample.append(
            {
                "sample_id": sample_id,
                "label": raw_label,
                "pred_best_index": pred_option_id,
                "policy_top1_confidence": record["policy_top1_confidence"],
                "selected_path": selected["path"],
                "openclip_prediction": pred_raw,
                "openclip_correct": bool(hit),
            }
        )

    result = {
        "config": {
            "manifest": args.manifest,
            "data_json": args.data_json,
            "checkpoint": args.checkpoint,
            "class_index_json": args.class_index_json,
            "openclip_model": args.openclip_model,
            "openclip_pretrained": args.openclip_pretrained,
            "prompt_template": args.prompt_template,
            "num_subset_classes": len(label_ids),
            "subset_label_ids": label_ids,
            "device": str(device),
        },
        "summary": {
            "policy_selected_openclip_top1": summarize_binary_hits(hits),
            "missing_manifest_samples": len(missing),
        },
        "missing_samples": missing,
        "records": per_sample,
    }
    save_json(args.output_json, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
