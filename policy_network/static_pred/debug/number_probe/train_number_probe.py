from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = ROOT / "data" / "ImageNet-ES-Diverse"
NUM_CANDIDATES = 27
ENCODING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
MAX_PERIOD = 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny MLP that predicts best_option_id from only the "
            "lightning condition number, class number, or both."
        )
    )
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--feature_mode",
        type=str,
        choices=["lightning_class", "lightning", "class"],
        required=True,
    )
    parser.add_argument(
        "--label_kind",
        type=str,
        choices=["oracle", "policy"],
        default="oracle",
        help="Which label JSON set under data/ImageNet-ES-Diverse to use.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_label_paths(label_kind: str) -> Tuple[Path, Path, Path]:
    if label_kind == "oracle":
        label_dir = DEFAULT_DATA_ROOT / "oracle_policy_labels"
        prefix = "oracle_policy"
    elif label_kind == "policy":
        label_dir = DEFAULT_DATA_ROOT / "policy_labels"
        prefix = "policy"
    else:
        raise ValueError(f"Unsupported label_kind={label_kind}")
    return (
        label_dir / f"{prefix}_train_labels.json",
        label_dir / f"{prefix}_val_labels.json",
        label_dir / f"{prefix}_test_labels.json",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def parse_env_number(env: Any) -> int:
    env_str = str(env)
    if not env_str.startswith("l"):
        raise ValueError(f"Expected env like 'l1', got {env_str!r}")
    return int(env_str[1:])


def build_mappings(
    split_items: Iterable[List[Dict[str, Any]]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    records = [record for items in split_items for record in items]
    envs = sorted({str(record["env"]) for record in records}, key=parse_env_number)
    class_ids = sorted({str(record["class_id"]) for record in records})

    env_to_number = {env: parse_env_number(env) for env in envs}
    class_to_number = {class_id: idx for idx, class_id in enumerate(class_ids)}
    return env_to_number, class_to_number


class NumberProbeDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        env_to_number: Dict[str, int],
        class_to_number: Dict[str, int],
    ):
        self.items = items
        self.env_to_number = env_to_number
        self.class_to_number = class_to_number

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        env = str(item["env"])
        class_id = str(item["class_id"])
        return {
            "lightning": torch.tensor(
                float(self.env_to_number[env]), dtype=torch.float32
            ),
            "class_number": torch.tensor(
                float(self.class_to_number[class_id]), dtype=torch.float32
            ),
            "target": torch.tensor(int(item["best_option_id"]), dtype=torch.long),
        }


class SinusoidalScalarEncoder(nn.Module):
    def __init__(self, encoding_dim: int, max_period: float):
        super().__init__()
        if encoding_dim <= 0 or encoding_dim % 2 != 0:
            raise ValueError("--encoding_dim must be a positive even integer.")
        half_dim = encoding_dim // 2
        exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32)
        exponent = exponent / max(1, half_dim - 1)
        self.register_buffer("inv_freq", torch.exp(exponent), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().view(-1, 1)
        phase = x * self.inv_freq.view(1, -1)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class NumberProbeMLP(nn.Module):
    def __init__(
        self,
        feature_mode: str,
        encoding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_candidates: int,
        max_period: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("--num_layers must be >= 1.")
        if feature_mode not in {"lightning_class", "lightning", "class"}:
            raise ValueError(f"Unsupported feature_mode={feature_mode}")

        self.feature_mode = feature_mode
        self.encoder = SinusoidalScalarEncoder(
            encoding_dim=encoding_dim,
            max_period=max_period,
        )
        num_inputs = 2 if feature_mode == "lightning_class" else 1
        input_dim = encoding_dim * num_inputs

        layers: List[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_candidates))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_parts = []
        if self.feature_mode in {"lightning_class", "lightning"}:
            encoded_parts.append(self.encoder(batch["lightning"]))
        if self.feature_mode in {"lightning_class", "class"}:
            encoded_parts.append(self.encoder(batch["class_number"]))
        features = torch.cat(encoded_parts, dim=-1)
        return self.mlp(features)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def macro_f1_from_counts(confusion: torch.Tensor) -> float:
    scores = []
    for index in range(confusion.shape[0]):
        tp = float(confusion[index, index].item())
        fp = float(confusion[:, index].sum().item() - tp)
        fn = float(confusion[index, :].sum().item() - tp)
        denom = (2.0 * tp) + fp + fn
        if denom > 0:
            scores.append((2.0 * tp) / denom)
    return float(sum(scores) / max(1, len(scores)))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_candidates: int,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pred_hist = torch.zeros(num_candidates, dtype=torch.long)
    target_hist = torch.zeros(num_candidates, dtype=torch.long)
    confusion = torch.zeros((num_candidates, num_candidates), dtype=torch.long)

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits = model(batch)
            targets = batch["target"]
            loss = nn.functional.cross_entropy(logits, targets)
            preds = torch.argmax(logits, dim=-1)

            batch_size = targets.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == targets).sum().item())
            total += batch_size

            preds_cpu = preds.detach().cpu()
            targets_cpu = targets.detach().cpu()
            pred_hist += torch.bincount(preds_cpu, minlength=num_candidates)
            target_hist += torch.bincount(targets_cpu, minlength=num_candidates)
            for target, pred in zip(targets_cpu.tolist(), preds_cpu.tolist()):
                confusion[target, pred] += 1

    per_class_acc = []
    for index in range(num_candidates):
        count = int(target_hist[index].item())
        if count == 0:
            per_class_acc.append(None)
        else:
            per_class_acc.append(float(confusion[index, index].item() / count))

    return {
        "loss": total_loss / max(1, total),
        "acc": 100.0 * total_correct / max(1, total),
        "macro_f1": macro_f1_from_counts(confusion),
        "total": total,
        "correct": total_correct,
        "pred_hist": pred_hist.tolist(),
        "target_hist": target_hist.tolist(),
        "per_class_acc": per_class_acc,
        "confusion": confusion.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}")

    for raw_batch in pbar:
        batch = move_batch(raw_batch, device)
        targets = batch["target"]

        optimizer.zero_grad()
        logits = model(batch)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        batch_size = targets.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == targets).sum().item())
        total += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / max(1, total):.4f}",
            acc=f"{100.0 * total_correct / max(1, total):.2f}",
        )

    return {
        "loss": total_loss / max(1, total),
        "acc": 100.0 * total_correct / max(1, total),
        "total": total,
        "correct": total_correct,
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def summarize_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": len(items),
        "env_hist": dict(sorted(Counter(str(item["env"]) for item in items).items())),
        "num_classes": len({str(item["class_id"]) for item in items}),
        "target_hist": dict(
            sorted(Counter(int(item["best_option_id"]) for item in items).items())
        ),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    train_json, val_json, test_json = get_label_paths(args.label_kind)
    train_items = load_json(train_json)
    val_items = load_json(val_json)
    test_items = load_json(test_json)
    env_to_number, class_to_number = build_mappings([train_items, val_items, test_items])

    with open(Path(args.save_dir) / "args.json", "w") as f:
        payload = {
            **vars(args),
            "train_json": str(train_json),
            "val_json": str(val_json),
            "test_json": str(test_json),
            "num_candidates": NUM_CANDIDATES,
            "encoding_dim": ENCODING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "max_period": MAX_PERIOD,
            "lightning_mode": "raw",
            "class_mode": "subset_index",
        }
        json.dump(payload, f, indent=2)
    with open(Path(args.save_dir) / "env_to_number.json", "w") as f:
        json.dump(env_to_number, f, indent=2, sort_keys=True)
    with open(Path(args.save_dir) / "class_to_number.json", "w") as f:
        json.dump(class_to_number, f, indent=2, sort_keys=True)

    dataset_summary = {
        "train": summarize_items(train_items),
        "val": summarize_items(val_items),
        "test": summarize_items(test_items),
    }
    with open(Path(args.save_dir) / "dataset_summary.json", "w") as f:
        json.dump(dataset_summary, f, indent=2)

    train_loader = make_loader(
        NumberProbeDataset(train_items, env_to_number, class_to_number),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = make_loader(
        NumberProbeDataset(val_items, env_to_number, class_to_number),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = make_loader(
        NumberProbeDataset(test_items, env_to_number, class_to_number),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = NumberProbeMLP(
        feature_mode=args.feature_mode,
        encoding_dim=ENCODING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_candidates=NUM_CANDIDATES,
        max_period=MAX_PERIOD,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_test_stats: Dict[str, Any] | None = None
    best_epoch = 0
    history = []
    best_ckpt_path = Path(args.save_dir) / "best_checkpoint.pth"
    last_ckpt_path = Path(args.save_dir) / "last_checkpoint.pth"

    print(f"feature_mode={args.feature_mode}")
    print(f"label_kind={args.label_kind}")
    print(f"env_to_number={env_to_number}")
    print(f"num_classes={len(class_to_number)}")
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )
        val_stats = evaluate(model, val_loader, device, NUM_CANDIDATES)
        test_stats = evaluate(model, test_loader, device, NUM_CANDIDATES)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_stats['loss']:.4f}, "
            f"train_acc={train_stats['acc']:.2f}% | "
            f"val_loss={val_stats['loss']:.4f}, "
            f"val_acc={val_stats['acc']:.2f}% | "
            f"test_acc={test_stats['acc']:.2f}%"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "val_macro_f1": val_stats["macro_f1"],
                "test_loss": test_stats["loss"],
                "test_acc": test_stats["acc"],
                "test_macro_f1": test_stats["macro_f1"],
            }
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "args": vars(args),
            "config": payload,
            "env_to_number": env_to_number,
            "class_to_number": class_to_number,
        }
        torch.save(ckpt, last_ckpt_path)

        if val_stats["acc"] > best_val_acc:
            best_val_acc = float(val_stats["acc"])
            best_test_stats = test_stats
            best_epoch = epoch
            ckpt["best_val_acc"] = best_val_acc
            torch.save(ckpt, best_ckpt_path)

    with open(Path(args.save_dir) / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_train_stats = evaluate(model, train_loader, device, NUM_CANDIDATES)
    final_val_stats = evaluate(model, val_loader, device, NUM_CANDIDATES)
    final_test_stats = evaluate(model, test_loader, device, NUM_CANDIDATES)

    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": None if best_test_stats is None else best_test_stats["acc"],
        "best_test_macro_f1": None
        if best_test_stats is None
        else best_test_stats["macro_f1"],
        "final_train": final_train_stats,
        "final_val": final_val_stats,
        "final_test": final_test_stats,
    }
    with open(Path(args.save_dir) / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(
        "Done. "
        f"best_epoch={best_epoch}, "
        f"best_val_acc={best_val_acc:.2f}%, "
        f"best_test_acc={final_metrics['best_test_acc']:.2f}%"
    )


if __name__ == "__main__":
    main()
