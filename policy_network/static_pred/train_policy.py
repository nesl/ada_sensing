from __future__ import annotations
import argparse
import json
import os
import random
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import imagenet_preprocess
from policy_model import SUPPORTED_BACKBONES, SensorPolicyNetwork
from policy_dataset import PolicyDataset


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--val_json", type=str, required=True)
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--manifest_json", type=str, default=None)
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to load model weights from before training.",
    )
    p.add_argument(
        "--input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default="real",
        help="Variant applied to the policy input image before preprocessing.",
    )
    p.add_argument(
        "--ae_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
        help="Optional variant for the auto-exposure/baseline input view.",
    )
    p.add_argument(
        "--env_input_variant",
        type=str,
        choices=["real", "random_noise_per_sample"],
        default=None,
        help="Optional variant for fixed environment option input views.",
    )
    p.add_argument(
        "--single_input_source",
        type=str,
        choices=["baseline", "env"],
        default="baseline",
        help="For input_mode=single, choose baseline AE image or fixed env_option_id image.",
    )
    p.add_argument(
        "--noise_seed",
        type=int,
        default=0,
        help="Base seed used for deterministic noise generation when input_variant uses noise.",
    )

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_candidates", type=int, default=27)
    p.add_argument(
        "--input_mode",
        type=str,
        choices=["single", "dual", "multiview"],
        default="single",
    )
    p.add_argument("--env_option_id", type=int, default=None)
    p.add_argument(
        "--env_option_ids",
        type=str,
        default=None,
        help="Comma-separated option ids for multiview input, e.g. '2,8,24'.",
    )
    p.add_argument(
        "--include_ae_input",
        action="store_true",
        help="For multiview input, prepend the auto-exposure baseline image.",
    )
    p.add_argument(
        "--backbone",
        type=str,
        choices=SUPPORTED_BACKBONES,
        default="mobilenet_v3_small",
    )

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument(
        "--loss_type",
        type=str,
        choices=["auto", "hard_ce", "soft_kl"],
        default="auto",
        help="Training loss. 'auto' uses soft_kl when the dataset provides soft_target.",
    )
    p.add_argument(
        "--trainable_scope",
        type=str,
        choices=["head_only", "partial_unfreeze", "full_finetune"],
        required=True,
        help="Explicit training regime.",
    )

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pretrained_flag(args) -> bool:
    return bool(args.pretrained)


def get_trainable_scope(args) -> str:
    return args.trainable_scope


def parse_env_option_ids(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def get_num_input_views(args) -> int:
    if args.input_mode == "single":
        return 1
    if args.input_mode == "dual":
        return 2
    env_option_ids = parse_env_option_ids(args.env_option_ids)
    return len(env_option_ids) + int(bool(args.include_ae_input))


def build_dataloaders(args):
    tfm = imagenet_preprocess(args.image_size)
    dataset_kwargs = {
        "transform": tfm,
        "manifest_path": args.manifest_json,
        "input_mode": args.input_mode,
        "env_option_id": args.env_option_id,
        "env_option_ids": parse_env_option_ids(args.env_option_ids),
        "include_ae_input": args.include_ae_input,
        "input_variant": args.input_variant,
        "ae_input_variant": args.ae_input_variant,
        "env_input_variant": args.env_input_variant,
        "single_input_source": args.single_input_source,
        "noise_seed": args.noise_seed,
    }

    train_ds = PolicyDataset(args.train_json, **dataset_kwargs)
    val_ds = PolicyDataset(args.val_json, **dataset_kwargs)
    test_ds = PolicyDataset(args.test_json, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def resolve_loss_type(args, train_loader) -> str:
    if args.loss_type != "auto":
        return args.loss_type
    return "soft_kl" if getattr(train_loader.dataset, "has_soft_targets", False) else "hard_ce"


def compute_loss(logits, batch, loss_type: str, device: torch.device):
    hard_targets = batch["target"].to(device, non_blocking=True)

    if loss_type == "hard_ce":
        loss = nn.functional.cross_entropy(logits, hard_targets)
        return loss, hard_targets

    if loss_type == "soft_kl":
        if "soft_target" not in batch:
            raise KeyError("Batch is missing 'soft_target' required for soft_kl.")
        soft_targets = batch["soft_target"].to(device, non_blocking=True)
        per_sample_loss = nn.functional.kl_div(
            nn.functional.log_softmax(logits, dim=-1),
            soft_targets,
            reduction="none",
        ).sum(dim=-1)
        if "sample_weight" in batch:
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            per_sample_loss = per_sample_loss * sample_weight
        loss = per_sample_loss.mean()
        return loss, hard_targets

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def evaluate(model, loader, device, loss_type: str) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            loss, hard_targets = compute_loss(logits, batch, loss_type, device)

            preds = torch.argmax(logits, dim=-1)

            batch_size = hard_targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == hard_targets).sum().item())
            total += batch_size

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    return {
        "loss": avg_loss,
        "acc": acc,
        "total": total,
        "correct": total_correct,
    }


def collect_prediction_distribution(model, loader, device, num_candidates: int) -> Dict[str, Any]:
    model.eval()

    softmax_sum = torch.zeros(num_candidates, dtype=torch.float64)
    argmax_hist = torch.zeros(num_candidates, dtype=torch.long)
    top1_conf_sum_by_pred_index = torch.zeros(num_candidates, dtype=torch.float64)
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            top1_confs, preds = torch.max(probs, dim=-1)

            softmax_sum += probs.sum(dim=0).detach().cpu().to(torch.float64)
            argmax_hist += torch.bincount(
                preds.detach().cpu(),
                minlength=num_candidates,
            )
            top1_conf_sum_by_pred_index += torch.bincount(
                preds.detach().cpu(),
                weights=top1_confs.detach().cpu().to(torch.float64),
                minlength=num_candidates,
            )
            total += probs.shape[0]

    mean_softmax_per_index = (softmax_sum / max(1, total)).tolist()
    argmax_hist_list = argmax_hist.tolist()
    mean_top1_conf_by_pred_index = []
    for index in range(num_candidates):
        count = int(argmax_hist[index].item())
        if count == 0:
            mean_top1_conf_by_pred_index.append(None)
        else:
            mean_top1_conf_by_pred_index.append(
                float(top1_conf_sum_by_pred_index[index].item() / count)
            )

    return {
        "mean_softmax_per_index": mean_softmax_per_index,
        "argmax_hist": argmax_hist_list,
        "mean_top1_confidence_by_pred_index": mean_top1_conf_by_pred_index,
    }


def train_one_epoch(model, loader, optimizer, device, loss_type: str, epoch: int, epochs: int):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss, hard_targets = compute_loss(logits, batch, loss_type, device)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)

        batch_size = hard_targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == hard_targets).sum().item())
        total += batch_size

        running_loss = total_loss / max(1, total)
        running_acc = 100.0 * total_correct / max(1, total)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}")

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    return {
        "loss": avg_loss,
        "acc": acc,
        "total": total,
        "correct": total_correct,
    }


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_acc: float, args, loss_type: str):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "num_candidates": args.num_candidates,
        "image_size": args.image_size,
        "trainable_scope": get_trainable_scope(args),
        "effective_trainable_scope": getattr(args, "effective_trainable_scope", get_trainable_scope(args)),
        "backbone_name": args.backbone,
        "input_mode": args.input_mode,
        "env_option_id": args.env_option_id,
        "env_option_ids": parse_env_option_ids(args.env_option_ids),
        "include_ae_input": args.include_ae_input,
        "num_input_views": get_num_input_views(args),
        "backbone_lr": args.backbone_lr,
        "head_lr": args.lr,
        "loss_type": loss_type,
        "input_variant": args.input_variant,
        "ae_input_variant": args.ae_input_variant,
        "env_input_variant": args.env_input_variant,
        "single_input_source": args.single_input_source,
        "noise_seed": args.noise_seed,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pretrained = get_pretrained_flag(args)
    trainable_scope = get_trainable_scope(args)

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args)
    loss_type = resolve_loss_type(args, train_loader)
    print(f"Using loss_type={loss_type}")
    print(f"Using trainable_scope={trainable_scope}")
    print(f"Using backbone={args.backbone}")
    print(f"Using input_mode={args.input_mode}")
    print(f"Using num_input_views={get_num_input_views(args)}")
    if args.input_mode == "multiview":
        print(f"Using env_option_ids={parse_env_option_ids(args.env_option_ids)}")
        print(f"Using include_ae_input={args.include_ae_input}")
    print(f"Using input_variant={args.input_variant}")
    print(f"Using ae_input_variant={args.ae_input_variant or args.input_variant}")
    print(f"Using env_input_variant={args.env_input_variant or 'real'}")
    if args.input_mode == "single":
        print(f"Using single_input_source={args.single_input_source}")

    print("Building model...")
    model = SensorPolicyNetwork(
        num_candidates=args.num_candidates,
        pretrained=pretrained,
        backbone_name=args.backbone,
        input_mode=args.input_mode,
        num_input_views=get_num_input_views(args),
    ).to(device)
    effective_trainable_scope = trainable_scope
    if model.requires_full_training and trainable_scope != "full_finetune":
        effective_trainable_scope = "full_finetune"
        print(
            f"Backbone {args.backbone} is trained from scratch; "
            f"using effective_trainable_scope={effective_trainable_scope} "
            f"for requested trainable_scope={trainable_scope}."
        )
    args.effective_trainable_scope = effective_trainable_scope

    best_val_acc = -1.0
    start_epoch = 1

    if args.resume_checkpoint:
        print(f"Loading resume checkpoint from {args.resume_checkpoint}")
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device)
        resume_backbone = resume_ckpt.get("backbone_name", "mobilenet_v3_small")
        resume_input_mode = resume_ckpt.get("input_mode", "single")
        if resume_backbone != args.backbone:
            raise ValueError(
                "Resume checkpoint backbone does not match current --backbone: "
                f"{resume_backbone} vs {args.backbone}"
            )
        if resume_input_mode != args.input_mode:
            raise ValueError(
                "Resume checkpoint input_mode does not match current --input_mode: "
                f"{resume_input_mode} vs {args.input_mode}"
            )
        model.load_state_dict(resume_ckpt["model_state_dict"])
        best_val_acc = float(resume_ckpt.get("best_val_acc", -1.0))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed model weights from epoch {resume_ckpt.get('epoch', 0)} "
            f"with best_val_acc={best_val_acc:.2f}"
        )

    if effective_trainable_scope == "head_only":
        model.freeze_backbone()
        optimizer_param_groups = [
            {
                "params": model.policy_head.parameters(),
                "lr": args.lr,
            },
        ]
        print("Backbone frozen. Training policy_head only.")
    elif effective_trainable_scope == "partial_unfreeze":
        model.unfreeze_backbone_tail()
        optimizer_param_groups = [
            {
                "params": model.get_backbone_tail_parameters(),
                "lr": args.backbone_lr,
            },
            {
                "params": model.policy_head.parameters(),
                "lr": args.lr,
            },
        ]
        print(
            "Partially unfroze the configured backbone tail and feature_proj. "
            f"Using backbone_lr={args.backbone_lr} and head_lr={args.lr}."
        )
    elif effective_trainable_scope == "full_finetune":
        model.unfreeze_backbone()
        optimizer_param_groups = [
            {
                "params": model.parameters(),
                "lr": args.lr,
            }
        ]
        print("Backbone unfrozen. Training the full network.")
    else:
        raise ValueError(f"Unsupported trainable_scope: {effective_trainable_scope}")

    optimizer = AdamW(
        optimizer_param_groups,
        weight_decay=args.weight_decay,
    )

    best_ckpt_path = os.path.join(args.save_dir, "best_checkpoint.pth")
    last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pth")
    history = []

    if args.resume_checkpoint and not os.path.exists(best_ckpt_path):
        save_checkpoint(
            path=best_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=start_epoch - 1,
            best_val_acc=best_val_acc,
            args=args,
            loss_type=loss_type,
        )
        print(
            "Saved resumed weights as the initial best checkpoint to "
            f"{best_ckpt_path}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=loss_type,
            epoch=epoch,
            epochs=args.epochs,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_type=loss_type,
        )
        train_distribution = collect_prediction_distribution(
            model=model,
            loader=train_loader,
            device=device,
            num_candidates=args.num_candidates,
        )
        val_distribution = collect_prediction_distribution(
            model=model,
            loader=val_loader,
            device=device,
            num_candidates=args.num_candidates,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f}, "
            f"train_acc={train_stats['acc']:.2f}% | "
            f"val_loss={val_stats['loss']:.4f}, "
            f"val_acc={val_stats['acc']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "loss_type": loss_type,
            "train_mean_softmax_per_index": train_distribution["mean_softmax_per_index"],
            "train_argmax_hist": train_distribution["argmax_hist"],
            "train_mean_top1_confidence_by_pred_index": train_distribution["mean_top1_confidence_by_pred_index"],
            "val_mean_softmax_per_index": val_distribution["mean_softmax_per_index"],
            "val_argmax_hist": val_distribution["argmax_hist"],
            "val_mean_top1_confidence_by_pred_index": val_distribution["mean_top1_confidence_by_pred_index"],
        })

        save_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            args=args,
            loss_type=loss_type,
        )
        print(f"Saved last checkpoint to {last_ckpt_path}")

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                args=args,
                loss_type=loss_type,
            )
            print(f"Saved new best checkpoint to {best_ckpt_path}")

    history_path = os.path.join(args.save_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")

    print("Loading best checkpoint for final test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_stats = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        loss_type=loss_type,
    )

    print(
        f"[Final Test] "
        f"loss={test_stats['loss']:.4f}, "
        f"acc={test_stats['acc']:.2f}% "
        f"({test_stats['correct']}/{test_stats['total']})"
    )

    test_result_path = os.path.join(args.save_dir, "index_test_result.json")
    with open(test_result_path, "w") as f:
        json.dump(test_stats, f, indent=2)

    print(f"Saved test result to {test_result_path}")


if __name__ == "__main__":
    main()
