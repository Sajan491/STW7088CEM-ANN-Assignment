"""Evaluate the trained tile classifier on the held-out test split.

Produces:
  results/tables/tile_classifier_metrics.csv
  results/figures/tile_confusion_matrix.png
  results/figures/tile_learning_curves.png   (from the training history CSV)

Usage:
    python -m src.evaluation.evaluate_tile_classifier
    python -m src.evaluation.evaluate_tile_classifier --split val --limit 200
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.tile_dataset import TileDataset
from src.evaluation.metrics import binary_metrics, confusion
from src.models.factory import build_model
from src.utils import plotting
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) over a loader."""
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets).astype(int), np.concatenate(probs)


def fig_confusion_matrix(cm: np.ndarray, split: str, figures_dir) -> None:
    """Confusion-matrix heatmap with count + row-share annotations."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    labels = ["no litter", "litter"]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / row_sums[i, 0]
            colour = "white" if cm[i, j] > cm.max() * 0.6 else plotting.INK
            ax.text(
                j, i, f"{cm[i, j]:,}\n({share:.1%})",
                ha="center", va="center", color=colour, fontsize=11,
            )
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Tile classifier confusion matrix ({split} split)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="tiles")
    plotting.save_figure(fig, "tile_confusion_matrix", figures_dir)
    plt.close(fig)


def fig_learning_curves(history: pd.DataFrame, figures_dir) -> None:
    """Two panels: loss and F1 across epochs for train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax1.plot(history["epoch"], history["train_loss"], color=plotting.BLUE, lw=2, label="train")
    ax1.plot(history["epoch"], history["val_loss"], color=plotting.GREEN, lw=2, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted BCE loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history["epoch"], history["train_f1"], color=plotting.BLUE, lw=2, label="train")
    ax2.plot(history["epoch"], history["val_f1"], color=plotting.GREEN, lw=2, label="validation")
    best = history["val_f1"].idxmax()
    ax2.scatter(
        history.loc[best, "epoch"], history.loc[best, "val_f1"],
        color=plotting.GREEN, zorder=3, s=45,
    )
    ax2.annotate(
        f"best val F1 {history.loc[best, 'val_f1']:.3f}",
        (history.loc[best, "epoch"], history.loc[best, "val_f1"]),
        textcoords="offset points", xytext=(6, -12),
        color=plotting.INK_SECONDARY, fontsize=9,
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 (litter class)")
    ax2.set_title("F1 score")
    ax2.legend()

    fig.suptitle("Tile classifier learning curves", y=1.02)
    plotting.save_figure(fig, "tile_learning_curves", figures_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the tile litter classifier")
    ap.add_argument("--config", default="configs/tile_classifier.yaml")
    ap.add_argument("--checkpoint", default=None, help="path to .pt (default: config)")
    ap.add_argument("--split", default=None, help="override evaluation split")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N tiles")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    plotting.apply_style()
    ecfg = cfg["evaluation"]
    split = args.split or ecfg["split"]
    figures_dir = resolve_path(ecfg["figures_dir"])
    tables_dir = resolve_path(ecfg["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or (
        resolve_path(cfg["training"]["checkpoint_dir"]) / cfg["training"]["checkpoint_name"]
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    print("=== Tile classifier evaluation ===")
    print(f"  checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
    print(f"  split: {split} | threshold: {ecfg['threshold']} | device: {device}")

    ds = TileDataset(
        resolve_path(cfg["data"]["tiles_dir"]), split, cfg["data"]["input_size"],
        limit=args.limit,
    )
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                    num_workers=cfg["data"].get("num_workers", 2))
    print(f"  tiles: {len(ds)}")

    y_true, y_prob = predict(model, dl, device)
    metrics = binary_metrics(y_true, y_prob, ecfg["threshold"])
    cm = confusion(y_true, y_prob, ecfg["threshold"])

    print("  metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    row = {"model": cfg["model"]["name"], "split": split, "tiles": len(ds),
           "threshold": ecfg["threshold"], **{k: round(v, 4) for k, v in metrics.items()}}
    out_csv = tables_dir / "tile_classifier_metrics.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    print(f"[table] saved {out_csv}")

    fig_confusion_matrix(cm, split, figures_dir)

    hist_file = resolve_path(cfg["training"]["history_file"])
    if hist_file.exists():
        fig_learning_curves(pd.read_csv(hist_file), figures_dir)
    else:
        print(f"[warn] history file {hist_file} not found; skipping learning curves")

    print("[done] evaluation complete")


if __name__ == "__main__":
    main()
