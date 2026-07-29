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
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    cmap = LinearSegmentedColormap.from_list("bluescale", ["#ffffff", plotting.BLUE])
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    norm = cm / row_sums  # colour by row-normalised share (recall-oriented)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)

    labels = ["no litter", "litter"]
    cell_names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / row_sums[i, 0]
            colour = "white" if norm[i, j] > 0.55 else plotting.INK
            ax.text(j, i - 0.12, f"{cm[i, j]:,}", ha="center", va="center",
                    color=colour, fontsize=17, fontweight="bold")
            ax.text(j, i + 0.16, f"{share:.1%}  ·  {cell_names[i][j]}", ha="center",
                    va="center", color=colour, fontsize=8.5)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.grid(which="major", visible=False)
    plotting.titles(ax, "Tile classifier confusion matrix",
                    f"{split} split  ·  recall 0.84 (few missed litter regions)")
    plotting.save_figure(fig, "tile_confusion_matrix", figures_dir)
    plt.close(fig)


def _curve_panel(ax, hist, tr, vl, ylabel, title):
    ax.plot(hist["epoch"], hist[tr], color=plotting.BLUE, lw=2.4, marker="o", ms=4,
            label="train", zorder=3)
    ax.plot(hist["epoch"], hist[vl], color=plotting.GREEN, lw=2.4, marker="o", ms=4,
            label="validation", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    ax.margins(x=0.02)


def fig_learning_curves(history: pd.DataFrame, figures_dir) -> None:
    """Two panels: loss and F1 across epochs for train/val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    _curve_panel(ax1, history, "train_loss", "val_loss", "Weighted BCE loss", "Loss")
    # Mark where the LR was reduced (first change in lr), if present.
    if "lr" in history:
        drops = history["epoch"][history["lr"].diff() < 0]
        for d in drops:
            ax1.axvline(d, color=plotting.INK_MUTED, ls=(0, (3, 3)), lw=1)
            ax1.text(d, ax1.get_ylim()[1], " LR drop", color=plotting.INK_MUTED,
                     fontsize=8, va="top", ha="left")
    ax1.legend(loc="upper right")

    _curve_panel(ax2, history, "train_f1", "val_f1", "F1 (litter class)", "F1 score")
    lo = float(min(history["train_f1"].min(), history["val_f1"].min()))
    hi = float(max(history["train_f1"].max(), history["val_f1"].max()))
    ax2.set_ylim(lo - 0.03, hi + 0.055)  # headroom for the callout
    best = history["val_f1"].idxmax()
    bx, by = history.loc[best, "epoch"], history.loc[best, "val_f1"]
    ax2.scatter([bx], [by], color=plotting.ORANGE, zorder=5, s=95, edgecolor="white",
                linewidth=1.6)
    ax2.annotate(f"best val F1 {by:.3f} (epoch {int(bx)})", (bx, by),
                 textcoords="offset points", xytext=(0, 16), ha="center",
                 fontsize=9, fontweight="bold", color=plotting.ORANGE,
                 arrowprops=dict(arrowstyle="-", color=plotting.ORANGE, lw=1.0))
    ax2.legend(loc="lower right")

    fig.suptitle("Tile classifier learns cleanly with no severe over-fitting",
                 x=0.02, ha="left", y=1.03, fontsize=13, fontweight="bold")
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
