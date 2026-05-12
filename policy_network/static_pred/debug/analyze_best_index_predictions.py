from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

# 这个脚本位于 debug/ 子目录下，直接运行时 Python 的 import 路径里
# 可能找不到同级项目模块，所以这里手动把项目根目录和 static_pred 目录加进去。
ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "policy_network" / "static_pred"

for extra_path in (ROOT, POLICY_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from policy_dataset import PolicyDataset
from policy_model import (
    SensorPolicyNetwork,
    infer_backbone_name_from_checkpoint,
    infer_input_mode_from_checkpoint,
    infer_num_input_views_from_checkpoint,
    normalize_policy_checkpoint_state_dict,
)
from utils import imagenet_preprocess


def parse_args() -> argparse.Namespace:
    # 这个脚本的输入/输出都比较直接：
    # - checkpoint: 训练好的模型权重
    # - data_json: 要评估的数据列表
    # - output_json: 统计结果保存位置
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_json", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--eval_ae_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
        help="Inference-time override for the AE/baseline view.",
    )
    p.add_argument(
        "--eval_env_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
        help="Inference-time override for fixed option views.",
    )
    p.add_argument(
        "--eval_single_input_source",
        type=str,
        choices=["baseline", "env"],
        default=None,
        help="Inference-time override for single-input source.",
    )
    p.add_argument(
        "--eval_noise_seed",
        type=int,
        default=None,
        help="Inference-time noise seed override.",
    )
    return p.parse_args()


def evaluate_predictions(
    model,
    loader,
    device: torch.device,
    topk: int,
) -> Dict[str, Any]:
    # 切换到 eval 模式，关闭 dropout / 使用 BN 的推理行为。
    model.eval()

    # records: 每个样本的详细预测结果，最后会写入 json
    # y_true / y_pred / confidences: 额外保存成扁平列表，方便后面做汇总统计
    records: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    topk_hits = 0

    # 这里只做推理和分析，不需要反向传播。
    with torch.no_grad():
        for batch in loader:
            # PolicyDataset 返回的是字典，至少包含：
            # - image: 输入图像
            # - target: GT best index
            # - sample_id: 样本标识，方便回查具体样本
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            # logits 是每个 candidate index 的原始分数。
            logits = model(images)
            # softmax 后转成概率分布；再取最大概率对应的 index 作为预测类别。
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            # top-k 用于衡量“正确标签是否出现在前几个高概率候选里”。
            # 如果类别总数少于 requested topk，就退化成 top-num_classes。
            effective_topk = min(topk, probs.shape[-1])
            topk_confs, topk_preds = torch.topk(probs, k=effective_topk, dim=-1)

            sample_ids = batch["sample_id"]
            for sample_id, target, pred, conf, topk_pred, topk_conf in zip(
                sample_ids, targets, preds, confs, topk_preds, topk_confs
            ):
                # 把 tensor 标量转成原生 Python 类型，便于后续 json 序列化。
                target_int = int(target.item())
                pred_int = int(pred.item())
                conf_float = float(conf.item())
                topk_pred_list = [int(x) for x in topk_pred.tolist()]
                topk_conf_list = [float(x) for x in topk_conf.tolist()]
                topk_hit = target_int in topk_pred_list
                y_true.append(target_int)
                y_pred.append(pred_int)
                confidences.append(conf_float)
                topk_hits += int(topk_hit)
                record = {
                    "sample_id": sample_id,
                    # 数据集中标注的最佳 index
                    "target_best_index": target_int,
                    # 模型预测出的最佳 index
                    "pred_best_index": pred_int,
                    # top-1 概率，表示模型对自己当前预测的确信程度
                    "top1_confidence": conf_float,
                    # top-k 候选及其概率，可以看 label 是否只是没排到第 1。
                    "topk_pred_indices": topk_pred_list,
                    "topk_confidences": topk_conf_list,
                    "topk_hit": topk_hit,
                }
                if topk == 5:
                    record.update(
                        {
                            "top5_pred_indices": topk_pred_list,
                            "top5_confidences": topk_conf_list,
                            "top5_hit": topk_hit,
                        }
                    )
                records.append(record)

    # 下面开始做整体统计，而不是单样本结果。
    total = len(records)
    top1_correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    # 统计真实标签 / 预测标签分别集中在哪些 index 上，
    # 可用于观察数据分布是否极度偏斜，或者模型是否塌缩到少数类别。
    true_counter = Counter(y_true)
    pred_counter = Counter(y_pred)
    # majority baseline: 永远预测训练/评估集里最常见的真实标签时，能拿到的准确率。
    # 如果模型准确率只比这个 baseline 高一点，说明效果可能并不理想。
    majority_label, majority_count = true_counter.most_common(1)[0]
    sorted_conf = sorted(confidences)

    summary = {
        "total": total,
        # 为兼容旧脚本，acc 仍然表示 top-1 accuracy。
        "correct": top1_correct,
        "acc": 100.0 * top1_correct / max(1, total),
        "top1_correct": top1_correct,
        "top1_acc": 100.0 * top1_correct / max(1, total),
        "topk": topk,
        "topk_correct": topk_hits,
        "topk_acc": 100.0 * topk_hits / max(1, total),
        "majority_label": majority_label,
        "majority_baseline_acc": 100.0 * majority_count / max(1, total),
        "num_true_classes": len(true_counter),
        "num_pred_classes": len(pred_counter),
        "mean_confidence": sum(confidences) / max(1, len(confidences)),
        "median_confidence": sorted_conf[len(sorted_conf) // 2] if sorted_conf else 0.0,
        "high_conf_ratio_0_9": sum(c > 0.9 for c in confidences) / max(1, len(confidences)),
    }
    if topk == 5:
        summary["top5_correct"] = topk_hits
        summary["top5_acc"] = 100.0 * topk_hits / max(1, total)

    return {
        "summary": summary,
        # 真实标签分布 / 预测标签分布
        "true_best_index_distribution": dict(sorted(true_counter.items())),
        "pred_best_index_distribution": dict(sorted(pred_counter.items())),
        # 只保留前 10 个最常见的 index，方便快速查看
        "top10_true_best_indices": true_counter.most_common(10),
        "top10_pred_best_indices": pred_counter.most_common(10),
        # 每个样本的明细，便于后续筛错例或按 sample_id 回查
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be >= 1.")
    # 确保输出目录存在，避免写 json 时报错。
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    # 如果用户指定了 cuda 且当前机器可用，就走 GPU；否则自动回退到 CPU。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 评估时要和训练阶段保持一致的图像预处理。
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

    # 从 checkpoint 里恢复模型结构参数和权重。
    # num_candidates 如果 checkpoint 里没存，就默认按 27 处理。
    backbone_name = infer_backbone_name_from_checkpoint(checkpoint)
    state_dict = normalize_policy_checkpoint_state_dict(
        checkpoint["model_state_dict"],
        backbone_name,
    )
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    model = SensorPolicyNetwork(
        num_candidates=checkpoint.get("num_candidates", 27),
        pretrained=False,
        backbone_name=backbone_name,
        input_mode=input_mode,
        num_input_views=infer_num_input_views_from_checkpoint(checkpoint),
    ).to(device)
    model.load_state_dict(state_dict)

    # 跑完整个数据集并生成统计结果。
    result = evaluate_predictions(model, loader, device, args.topk)
    result["config"] = {
        "checkpoint": args.checkpoint,
        "data_json": args.data_json,
        "manifest": args.manifest,
        "image_size": args.image_size,
        "device": str(device),
        "topk": args.topk,
        "checkpoint_input_variant": checkpoint_input_variant,
        "checkpoint_ae_input_variant": checkpoint.get("ae_input_variant"),
        "checkpoint_env_input_variant": checkpoint.get("env_input_variant"),
        "checkpoint_single_input_source": checkpoint.get("single_input_source"),
        "eval_ae_input_variant": ae_input_variant,
        "eval_env_input_variant": env_input_variant,
        "eval_single_input_source": single_input_source,
        "eval_noise_seed": noise_seed,
    }

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    # 终端里打印一个简版摘要，json 里保存完整结果。
    print(json.dumps(result["summary"], indent=2))
    print("Top predicted best indices:", result["top10_pred_best_indices"])
    print(f"Saved analysis to {args.output_json}")


if __name__ == "__main__":
    main()
