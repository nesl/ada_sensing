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

    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_candidates", type=int, default=27)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_pretrained", action="store_true")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pretrained_flag(args) -> bool:
    if args.no_pretrained:
        return False
    return True


def build_dataloaders(args):
    tfm = imagenet_preprocess(args.image_size)

    train_ds = PolicyDataset(args.train_json, transform=tfm)
    val_ds = PolicyDataset(args.val_json, transform=tfm)
    test_ds = PolicyDataset(args.test_json, transform=tfm)

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


def evaluate(model, loader, criterion, device) -> Dict[str, Any]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            preds = torch.argmax(logits, dim=-1)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == targets).sum().item())
            total += batch_size

    avg_loss = total_loss / max(1, total)
    acc = 100.0 * total_correct / max(1, total)

    return {
        "loss": avg_loss,
        "acc": acc,
        "total": total,
        "correct": total_correct,
    }


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int, epochs: int):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}")

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
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


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_acc: float, args):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "num_candidates": args.num_candidates,
        "image_size": args.image_size,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pretrained = get_pretrained_flag(args)

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(args)

    print("Building model...")
    model = SensorPolicyNetwork(
        num_candidates=args.num_candidates,
        pretrained=pretrained,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "best_policy_net.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
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
        criterion=criterion,
        device=device,
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