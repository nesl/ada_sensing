from __future__ import annotations

"""
这个脚本做两件事：
1. 拿 policy network 预测出来的 best index，回到 manifest 里找到对应 candidate 图。
2. 把这张图重新送进 ViSIT / Lens 用的分类模型，统计 top-1 acc。

同时它也会在同一批样本上跑一遍 Lens 自己的选择逻辑：
- 对每个 sample 的全部 candidate 图做前向
- 用 max-softmax confidence 选出 Lens 认为最好的那一张
- 统计 Lens 选图后的 top-1 acc

最后输出：
- policy 选图后的 acc
- lens 选图后的 acc
- 两者差值
- 两者选到相同 index 的比例
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# 把项目根目录、lens 目录、policy 目录都加入 import path，
# 这样这个 debug 脚本可以直接复用现有模块。
ROOT = Path(__file__).resolve().parents[3]
LENS_DIR = ROOT / "lens"
POLICY_DIR = ROOT / "policy_network" / "static_pred"

for extra_path in (ROOT, LENS_DIR, POLICY_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from lens.data_utils import ManifestLensDataset, imagenet_preprocess, load_image_rgb, load_timm_model
from policy_dataset import PolicyDataset
from policy_model import (
    SensorPolicyNetwork,
    infer_backbone_name_from_checkpoint,
    infer_input_mode_from_checkpoint,
    infer_num_input_views_from_checkpoint,
    normalize_policy_checkpoint_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate object classification accuracy when using policy-network "
            "predicted candidate indices, and compare against Lens selection."
        )
    )
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    # 两种输入方式二选一：
    # 1) 直接给已有的 predictions_json
    # 2) 给 policy checkpoint + data_json，当场重新跑一遍 policy inference
    parser.add_argument("--predictions_json", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_json", type=str, default=None)

    # 这里的 model 是回灌时用的分类模型，也就是 Lens / ViSIT 侧的 backbone。
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--eval_ae_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
    )
    parser.add_argument(
        "--eval_env_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
    )
    parser.add_argument(
        "--eval_single_input_source",
        type=str,
        choices=["baseline", "env"],
        default=None,
    )
    parser.add_argument("--eval_noise_seed", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    has_prediction_file = args.predictions_json is not None
    has_checkpoint_inputs = args.checkpoint is not None or args.data_json is not None

    if has_prediction_file and has_checkpoint_inputs:
        raise ValueError("Use either --predictions_json or (--checkpoint and --data_json), not both.")

    if not has_prediction_file and not (args.checkpoint and args.data_json):
        raise ValueError(
            "Provide --predictions_json, or provide both --checkpoint and --data_json."
        )


def load_prediction_records(args: argparse.Namespace, device: torch.device) -> List[Dict[str, Any]]:
    """
    返回 policy 的预测结果列表，每个元素至少包含：
    - sample_id
    - pred_best_index
    - top1_confidence

    如果已经有 analysis json，就直接读取。
    否则就用 checkpoint + data_json 重新跑一遍 policy network。
    """
    if args.predictions_json is not None:
        with open(args.predictions_json, "r") as f:
            payload = json.load(f)
        return payload["records"]

    checkpoint = torch.load(args.checkpoint, map_location=device)
    transform = imagenet_preprocess(args.image_size)
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    checkpoint_input_variant = checkpoint.get("input_variant") or "real"
    ae_input_variant = (
        args.eval_ae_input_variant
        or checkpoint.get("ae_input_variant")
        or checkpoint_input_variant
    )
    env_input_variant = (
        args.eval_env_input_variant
        or checkpoint.get("env_input_variant")
        or "real"
    )
    single_input_source = (
        args.eval_single_input_source
        or checkpoint.get("single_input_source")
        or "baseline"
    )
    noise_seed = (
        args.eval_noise_seed
        if args.eval_noise_seed is not None
        else int(checkpoint.get("noise_seed", 0))
    )
    dataset = PolicyDataset(
        args.data_json,
        transform=transform,
        manifest_path=args.manifest,
        input_mode=input_mode,
        env_option_id=checkpoint.get("env_option_id"),
        env_option_ids=checkpoint.get("env_option_ids"),
        include_ae_input=bool(checkpoint.get("include_ae_input", False)),
        input_variant=checkpoint_input_variant,
        ae_input_variant=ae_input_variant,
        env_input_variant=env_input_variant,
        single_input_source=single_input_source,
        noise_seed=noise_seed,
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
                        "sample_id": sample_id,
                        "pred_best_index": int(pred.item()),
                        "top1_confidence": float(conf.item()),
                    }
                )

    return records


def build_manifest_index(manifest_path: str) -> Dict[str, Dict[str, Any]]:
    """
    把 manifest 做成 sample_id -> manifest_item 的字典，
    方便后面根据 policy 的 sample_id 直接找到全部 candidate 信息。
    """
    dataset = ManifestLensDataset(manifest_path)
    manifest_index: Dict[str, Dict[str, Any]] = {}
    for item in dataset.items:
        sample_id = item.get("id")
        if sample_id is None:
            raise KeyError("Manifest item is missing required key 'id'.")
        manifest_index[sample_id] = item
    return manifest_index


def build_candidate_tensor(
    candidates: List[Dict[str, Any]],
    transform,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    对一个 sample 的全部 candidate 图做预处理，并返回：
    - 堆起来的图像 tensor，shape [K, 3, H, W]
    - option_id -> 在 tensor 中位置 的映射

    这里的映射很关键，因为 policy 预测的是 option_id，
    但真正拿 logits 时我们要知道它在 candidates 列表中的位置。
    """
    images = [transform(load_image_rgb(candidate["path"])) for candidate in candidates]
    option_id_to_pos = {
        int(candidate["option_id"]): pos for pos, candidate in enumerate(candidates)
    }
    return torch.stack(images, dim=0), option_id_to_pos


def summarize_accuracy(num_correct: int, total: int) -> Dict[str, Any]:
    return {
        "correct": num_correct,
        "total": total,
        "acc": 100.0 * num_correct / max(1, total),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = imagenet_preprocess(args.image_size)

    # 这个 classifier 就是“回灌”时用的视觉分类模型。
    # policy 只负责选 index，不负责最终类别判断。
    classifier = load_timm_model(args.model, device=device)

    # prediction_records: policy 输出的 sample_id -> pred_best_index
    # manifest_index: sample_id -> 该样本的全部 candidate 图与 GT label
    prediction_records = load_prediction_records(args, device=device)
    manifest_index = build_manifest_index(args.manifest)

    per_sample: List[Dict[str, Any]] = []
    total = 0
    policy_correct = 0
    lens_correct = 0
    same_choice = 0
    missing_samples: List[str] = []

    for record in tqdm(prediction_records, desc="Evaluate policy vs lens"):
        sample_id = record["sample_id"]
        manifest_item = manifest_index.get(sample_id)
        if manifest_item is None:
            missing_samples.append(sample_id)
            continue

        label = int(manifest_item["label"])
        candidates = manifest_item["candidates"]
        candidate_tensor, option_id_to_pos = build_candidate_tensor(candidates, transform)

        # policy 预测的是 best option_id，比如 8、12、24 这种。
        pred_best_index = int(record["pred_best_index"])
        if pred_best_index not in option_id_to_pos:
            raise ValueError(
                f"Predicted option_id {pred_best_index} not found in manifest candidates for {sample_id}."
            )

        # 先把这个 sample 的所有 candidate 一次性过 classifier，
        # 后面 policy 和 lens 都直接复用这份 logits，避免重复前向。
        with torch.no_grad():
            logits = classifier(candidate_tensor.to(device, non_blocking=True))

        # -----------------------------
        # 1. Policy 路径
        # -----------------------------
        # 用 policy 给出的 option_id 找到对应的 candidate 位置，
        # 再看那张图经过 classifier 后的类别预测是否正确。
        policy_pos = option_id_to_pos[pred_best_index]
        policy_logits = logits[policy_pos]
        policy_pred_label = int(torch.argmax(policy_logits).item())
        policy_hit = int(policy_pred_label == label)

        # -----------------------------
        # 2. Lens 路径
        # -----------------------------
        # Lens 的策略是：对每张 candidate 图看分类 logits 的 max-softmax confidence，
        # 选择 confidence 最大的那一张作为 best candidate。
        conf = torch.softmax(logits, dim=-1).max(dim=-1).values
        lens_pos = int(torch.argmax(conf).item())
        lens_conf = float(conf[lens_pos].item())
        lens_logits = logits[lens_pos]
        lens_option_id = int(candidates[lens_pos]["option_id"])
        lens_pred_label = int(torch.argmax(lens_logits).item())
        lens_hit = int(lens_pred_label == label)

        policy_correct += policy_hit
        lens_correct += lens_hit
        same_choice += int(pred_best_index == lens_option_id)
        total += 1

        # 保留逐样本结果，方便后面 debug：
        # 例如查 policy 选错了哪些 index、Lens 选的又是什么。
        per_sample.append(
            {
                "sample_id": sample_id,
                "label": label,
                "policy_pred_best_index": pred_best_index,
                "policy_pred_confidence": record.get("top1_confidence"),
                "policy_class_prediction": policy_pred_label,
                "policy_class_correct": bool(policy_hit),
                "lens_best_index": lens_option_id,
                "lens_confidence": lens_conf,
                "lens_class_prediction": lens_pred_label,
                "lens_class_correct": bool(lens_hit),
                "same_selected_index": pred_best_index == lens_option_id,
            }
        )

    policy_acc = summarize_accuracy(policy_correct, total)
    lens_acc = summarize_accuracy(lens_correct, total)

    result = {
        "config": {
            "manifest": args.manifest,
            "predictions_json": args.predictions_json,
            "checkpoint": args.checkpoint,
            "data_json": args.data_json,
            "model": args.model,
            "image_size": args.image_size,
            "device": str(device),
        },
        "summary": {
            "evaluated_samples": total,
            "missing_manifest_samples": len(missing_samples),
            "policy_selected_acc": policy_acc,
            "lens_selected_acc": lens_acc,
            "policy_minus_lens_acc": policy_acc["acc"] - lens_acc["acc"],
            "same_index_rate": 100.0 * same_choice / max(1, total),
        },
        "missing_samples": missing_samples,
        "records": per_sample,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved evaluation to {args.output_json}")


if __name__ == "__main__":
    main()
