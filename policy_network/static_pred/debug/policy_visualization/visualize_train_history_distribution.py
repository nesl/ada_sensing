from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HISTORY = (
    ROOT
    / "policy_network"
    / "results_fixed_input13"
    / "B_lens_head_hard"
    / "train_history.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "policy_network"
    / "results_fixed_input13"
    / "B_lens_head_hard"
    / "train_history_distribution_summary.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize training-history distribution signals saved in train_history.json, "
            "including mean softmax per index and argmax histogram per epoch."
        )
    )
    parser.add_argument("--history_json", type=str, default=str(DEFAULT_HISTORY))
    parser.add_argument("--output_png", type=str, default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def to_matrix(history: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([item[key] for item in history], dtype=float)


def to_zero_filled_matrix(history: List[Dict[str, Any]], key: str) -> np.ndarray:
    rows = []
    for item in history:
        row = [0.0 if value is None else float(value) for value in item[key]]
        rows.append(row)
    return np.array(rows, dtype=float)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)

    history = load_history(args.history_json)
    if not history:
        raise ValueError(f"History file is empty: {args.history_json}")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    train_argmax = normalize_rows(to_matrix(history, "train_argmax_hist"))
    val_argmax = normalize_rows(to_matrix(history, "val_argmax_hist"))
    train_top1_conf_by_pred_index = to_zero_filled_matrix(
        history,
        "train_mean_top1_confidence_by_pred_index",
    )
    val_top1_conf_by_pred_index = to_zero_filled_matrix(
        history,
        "val_mean_top1_confidence_by_pred_index",
    )

    num_indices = train_argmax.shape[1]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker="o", label="train_loss")
    ax.plot(epochs, val_loss, marker="o", label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, train_acc, marker="o", label="train_acc")
    ax.plot(epochs, val_acc, marker="o", label="val_acc")
    ax.set_title("Index Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    im = axes[1, 0].imshow(
        train_argmax.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[1, 0].set_title("Train Argmax Distribution\nColor = fraction of samples predicted as this index")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Index")
    axes[1, 0].set_xticks(range(len(epochs)))
    axes[1, 0].set_xticklabels(epochs)
    axes[1, 0].set_yticks(range(num_indices))
    cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Argmax fraction")

    im = axes[1, 1].imshow(
        val_argmax.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes[1, 1].set_title("Val Argmax Distribution\nColor = fraction of samples predicted as this index")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Index")
    axes[1, 1].set_xticks(range(len(epochs)))
    axes[1, 1].set_xticklabels(epochs)
    axes[1, 1].set_yticks(range(num_indices))
    cbar = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Argmax fraction")

    conf_values = np.concatenate(
        [train_top1_conf_by_pred_index.ravel(), val_top1_conf_by_pred_index.ravel()]
    )
    conf_vmin = float(np.min(conf_values))
    conf_vmax = float(np.max(conf_values))

    im = axes[2, 0].imshow(
        train_top1_conf_by_pred_index.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=conf_vmin,
        vmax=conf_vmax,
    )
    axes[2, 0].set_title(
        "Train Mean Top1 Confidence By Pred Index\n"
        "Color = mean max-softmax; 0 means no sample predicted this index"
    )
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Index")
    axes[2, 0].set_xticks(range(len(epochs)))
    axes[2, 0].set_xticklabels(epochs)
    axes[2, 0].set_yticks(range(num_indices))
    cbar = fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Mean top1 confidence")

    im = axes[2, 1].imshow(
        val_top1_conf_by_pred_index.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=conf_vmin,
        vmax=conf_vmax,
    )
    axes[2, 1].set_title(
        "Val Mean Top1 Confidence By Pred Index\n"
        "Color = mean max-softmax; 0 means no sample predicted this index"
    )
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Index")
    axes[2, 1].set_xticks(range(len(epochs)))
    axes[2, 1].set_xticklabels(epochs)
    axes[2, 1].set_yticks(range(num_indices))
    cbar = fig.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Mean top1 confidence")

    fig.suptitle(
        f"Training Distribution Summary\n{args.history_json}",
        fontsize=14,
    )
    fig.savefig(args.output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output_png}")


if __name__ == "__main__":
    main()
