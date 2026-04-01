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
from policy_model import SensorPolicyNetwork
from policy_dataset import PolicyDataset


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--val_json", type=str, required=True)
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument(
        "--manifest_json",
        type=str,
        default="data/ImageNet-ES-Diverse/manifest_all.json",
        help="Manifest used to recover all candidates for state augmentation.",
    )

    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument(
        "--checkpoint_name",
        type=str,
        default="policy_net_part_freeze_soft.pth",
        help="Filename used for the best checkpoint saved inside save_dir.",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to load model weights from before training.",
    )

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_candidates", type=int, default=27)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument(
        "--train_input_sampling",
        type=str,
        choices=["baseline", "random_candidate", "fixed_option"],
        default="random_candidate",
        help="How to choose the training input image for each scene.",
    )
    p.add_argument(
        "--train_input_option_id",
        type=int,
        default=None,
        help="Candidate option_id used when train_input_sampling='fixed_option'.",
    )
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
        default=None,
        help="Explicit training regime. If omitted, the legacy freeze flags are used.",
    )
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Legacy flag. When trainable_scope is omitted, this maps to partial_unfreeze.",
    )
    p.add_argument(
        "--no_freeze_backbone",
        action="store_true",
        help="Legacy flag. When trainable_scope is omitted, this maps to full_finetune.",
    )

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pretrained_flag(args) -> bool:
    if args.no_pretrained:
        return False
    return True


def get_trainable_scope(args) -> str:
    if args.trainable_scope is not None:
        return args.trainable_scope
    if args.no_freeze_backbone:
        return "full_finetune"
    return "partial_unfreeze"


def build_dataloaders(args):
    tfm = imagenet_preprocess(args.image_size)

    train_ds = PolicyDataset(
        args.train_json,
        transform=tfm,
        manifest_path=args.manifest_json,
        input_sampling=args.train_input_sampling,
        fixed_option_id=args.train_input_option_id,
    )
    val_ds = PolicyDataset(args.val_json, transform=tfm, input_sampling="baseline")
    test_ds = PolicyDataset(args.test_json, transform=tfm, input_sampling="baseline")

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
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(logits, dim=-1),
            soft_targets,
            reduction="batchmean",
        )
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
        "backbone_lr": args.backbone_lr,
        "head_lr": args.lr,
        "loss_type": loss_type,
        "train_input_sampling": args.train_input_sampling,
        "train_input_option_id": args.train_input_option_id,
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
    print(
        f"Using train_input_sampling={args.train_input_sampling}, "
        f"train_input_option_id={args.train_input_option_id}"
    )
    print(f"Using trainable_scope={trainable_scope}")

    print("Building model...")
    model = SensorPolicyNetwork(
        num_candidates=args.num_candidates,
        pretrained=pretrained,
    ).to(device)

    best_val_acc = -1.0
    start_epoch = 1

    if args.resume_checkpoint:
        print(f"Loading resume checkpoint from {args.resume_checkpoint}")
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        best_val_acc = float(resume_ckpt.get("best_val_acc", -1.0))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        print(
            f"Resumed model weights from epoch {resume_ckpt.get('epoch', 0)} "
            f"with best_val_acc={best_val_acc:.2f}"
        )

    if trainable_scope == "head_only":
        model.freeze_backbone()
        optimizer_param_groups = [
            {
                "params": model.policy_head.parameters(),
                "lr": args.lr,
            },
        ]
        print("Backbone frozen. Training policy_head only.")
    elif trainable_scope == "partial_unfreeze":
        model.unfreeze_backbone_tail(start_idx=9)
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
            "Partially unfroze backbone tail (modules 9-12) and feature_proj. "
            f"Using backbone_lr={args.backbone_lr} and head_lr={args.lr}."
        )
    elif trainable_scope == "full_finetune":
        model.unfreeze_backbone()
        optimizer_param_groups = [
            {
                "params": model.parameters(),
                "lr": args.lr,
            }
        ]
        print("Backbone unfrozen. Training the full network.")
    else:
        raise ValueError(f"Unsupported trainable_scope: {trainable_scope}")

    optimizer = AdamW(
        optimizer_param_groups,
        weight_decay=args.weight_decay,
    )

    best_ckpt_path = os.path.join(args.save_dir, args.checkpoint_name)
    history = []

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
        })

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

    test_result_path = os.path.join(args.save_dir, "test_result.json")
    with open(test_result_path, "w") as f:
        json.dump(test_stats, f, indent=2)

    print(f"Saved test result to {test_result_path}")


if __name__ == "__main__":
    main()
