from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
FEATURE_MODES = ("lightning_class", "lightning", "class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val accuracy and loss for number-probe experiments."
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=(
            "Directory containing lightning_class/, lightning/, and class/. "
            "Defaults to policy_network/results_number_probe/{label_kind}."
        ),
    )
    parser.add_argument(
        "--label_kind",
        type=str,
        choices=["oracle", "policy"],
        default="oracle",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to results_root.",
    )
    return parser.parse_args()


def load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing train history: {path}")
    with open(path, "r") as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"Empty train history: {path}")
    return history


def plot_metric(
    histories: Dict[str, List[Dict[str, Any]]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False)

    for ax, feature_mode in zip(axes, FEATURE_MODES):
        history = histories[feature_mode]
        epochs = [int(row["epoch"]) for row in history]
        train_values = [float(row[train_key]) for row in history]
        val_values = [float(row[val_key]) for row in history]

        ax.plot(epochs, train_values, label="train", linewidth=2)
        ax.plot(epochs, val_values, label="val", linewidth=2)
        ax.set_title(feature_mode)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    results_root = (
        Path(args.results_root)
        if args.results_root is not None
        else ROOT / "policy_network" / "results_number_probe" / args.label_kind
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    histories = {
        feature_mode: load_history(results_root / feature_mode / "train_history.json")
        for feature_mode in FEATURE_MODES
    }

    acc_path = output_dir / "number_probe_train_val_acc.png"
    loss_path = output_dir / "number_probe_train_val_loss.png"

    acc_path = plot_metric(
        histories=histories,
        train_key="train_acc",
        val_key="val_acc",
        ylabel="accuracy (%)",
        title="Number Probe Train/Val Accuracy",
        output_path=acc_path,
    )
    loss_path = plot_metric(
        histories=histories,
        train_key="train_loss",
        val_key="val_loss",
        ylabel="loss",
        title="Number Probe Train/Val Loss",
        output_path=loss_path,
    )

    print(f"Saved {acc_path}")
    print(f"Saved {loss_path}")


if __name__ == "__main__":
    main()
