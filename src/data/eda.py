"""Exploratory data analysis for UAVVaste.

Produces the report's dataset figures and tables from the COCO annotation
file alone (image width/height are stored in the annotations, so the 2.9 GB
image archive is NOT required):

  results/figures/object_size_distribution.png
  results/figures/annotations_per_image.png
  results/figures/tile_class_balance.png
  results/tables/dataset_summary.csv
  results/tables/tile_class_balance.csv

Usage:
    python -m src.data.eda
    python -m src.data.eda --limit 50   # smoke test (tile stats on 50 images)
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.coco_parser import CocoDataset
from src.data.splits import make_splits
from src.data.tiles import build_tile_index, class_balance
from src.utils import plotting
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def _callout(ax, x, y, text, color=plotting.INK):
    """A soft rounded annotation box."""
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=plotting.PANEL,
                      edgecolor=plotting.GRID, linewidth=1))


def fig_object_sizes(fracs: np.ndarray, figures_dir) -> None:
    """Histogram of bbox area as a fraction of image area (log scale)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    pct = fracs * 100
    bins = np.logspace(np.log10(pct.min()), np.log10(pct.max()), 42)
    ax.hist(pct, bins=bins, color=plotting.BLUE, edgecolor="white", linewidth=0.5, alpha=0.95)
    ax.set_xscale("log")
    ymax = ax.get_ylim()[1]

    # Shade the "smaller than 1% of image" region — the whole challenge.
    ax.axvspan(pct.min(), 1.0, color=plotting.BLUE_SOFT, alpha=0.18, zorder=0)
    median = np.median(pct)
    ax.axvline(median, color=plotting.ORANGE, linewidth=1.8)
    ax.text(median * 1.12, ymax * 0.95, f"median\n{median:.3f}%", color=plotting.ORANGE,
            fontsize=9, fontweight="bold", va="top")
    ax.axvline(1.0, color=plotting.INK_SECONDARY, linewidth=1.3, linestyle=(0, (4, 3)))
    ax.text(1.15, ymax * 0.6, "1% of\nimage area", color=plotting.INK_SECONDARY, fontsize=9)

    frac_under = float((pct < 1.0).mean())
    _callout(ax, 0.015, 0.97,
             f"{frac_under:.1%} of objects are\nsmaller than 1% of the frame")

    ax.set_xlabel("Bounding-box area as % of image area  (log scale)")
    ax.set_ylabel("Number of annotations")
    plotting.titles(ax, "Litter objects are tiny relative to the frame",
                    "UAVVaste bounding-box size distribution")
    plotting.save_figure(fig, "object_size_distribution", figures_dir)
    plt.close(fig)


def fig_annotations_per_image(counts: np.ndarray, figures_dir) -> None:
    """Histogram of annotations per image."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.arange(0, counts.max() + 2) - 0.5
    ax.hist(counts, bins=bins, color=plotting.BLUE, edgecolor="white", linewidth=0.6, alpha=0.95)

    mean, median = counts.mean(), np.median(counts)
    ax.axvline(mean, color=plotting.ORANGE, linewidth=1.8)
    ax.text(mean + 0.8, ax.get_ylim()[1] * 0.9, f"mean {mean:.1f}", color=plotting.ORANGE,
            fontsize=9.5, fontweight="bold")
    _callout(ax, 0.5, 0.97,
             f"median {median:.0f}  ·  max {int(counts.max())}\nevery image contains litter")

    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Number of images")
    ax.set_xlim(-0.5, min(counts.max() + 1, 30))
    plotting.titles(ax, "Most images hold only a few litter items",
                    "Litter annotations per UAVVaste image")
    plotting.save_figure(fig, "annotations_per_image", figures_dir)
    plt.close(fig)


def fig_tile_balance(stats: dict, figures_dir) -> None:
    """Grouped bars: litter vs no-litter tile counts per split."""
    splits = [s for s in ("train", "val", "test") if s in stats]
    pos = [stats[s]["pos"] for s in splits]
    neg = [stats[s]["neg"] for s in splits]

    x = np.arange(len(splits))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - width / 2, neg, width, label="no litter", color=plotting.GREEN)
    b2 = ax.bar(x + width / 2, pos, width, label="litter", color=plotting.BLUE)
    plotting.bar_labels(ax, b1, fmt="{:,.0f}", fontsize=8.5)
    plotting.bar_labels(ax, b2, fmt="{:,.0f}", fontsize=8.5)

    labels = []
    for s in splits:
        n = stats[s]["pos"] + stats[s]["neg"]
        labels.append(f"{s.capitalize()}\n{stats[s]['pos_frac']:.1%} litter  ·  {n:,} tiles")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Number of tiles")
    ax.set_ylim(0, max(neg) * 1.16)
    total_pos = stats.get("total", {}).get("pos_frac", stats[splits[0]]["pos_frac"])
    plotting.titles(ax, "Tiles are strongly imbalanced toward 'no litter'",
                    f"512 px tiles per split  ·  ~{total_pos:.0%} positive overall (image-level splits)")
    ax.legend(loc="upper right", ncol=2)
    plotting.save_figure(fig, "tile_class_balance", figures_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="UAVVaste exploratory data analysis")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: tile stats on N images")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    plotting.apply_style()
    figures_dir = resolve_path(cfg["eda"]["figures_dir"])
    tables_dir = resolve_path(cfg["eda"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=== UAVVaste EDA ===")
    ds = CocoDataset(resolve_path(cfg["dataset"]["annotations_file"]))
    summary = ds.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(
        tables_dir / "dataset_summary.csv"
    )
    print(f"[table] saved {tables_dir / 'dataset_summary.csv'}")

    fracs = np.array(ds.bbox_area_fractions())
    counts = np.array(ds.ann_counts_per_image())
    fig_object_sizes(fracs, figures_dir)
    fig_annotations_per_image(counts, figures_dir)

    # Tile class balance: reuse the saved splits when present, otherwise
    # derive them deterministically (same seed -> same assignment).
    splits_file = resolve_path(cfg["splits"]["file"])
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]
        print(f"[info] using existing splits from {splits_file}")
    else:
        splits = make_splits(sorted(ds.file_names()), cfg["splits"]["ratios"], cfg["seed"])
        print("[info] splits file not found; derived splits in memory (same seed)")

    tcfg = cfg["tiles"]
    index = build_tile_index(
        ds,
        splits,
        size=tcfg["size"],
        stride=tcfg["stride"],
        min_overlap_frac=tcfg["min_overlap_frac"],
        min_overlap_px=tcfg["min_overlap_px"],
        limit=args.limit,
    )
    stats = class_balance(index)
    for split, s in stats.items():
        print(f"  tiles[{split}]: {s['pos']} pos / {s['neg']} neg ({s['pos_frac']:.1%} positive)")

    rows = [{"split": k, **v} for k, v in stats.items()]
    pd.DataFrame(rows).to_csv(tables_dir / "tile_class_balance.csv", index=False)
    print(f"[table] saved {tables_dir / 'tile_class_balance.csv'}")

    fig_tile_balance(stats, figures_dir)
    print("[done] EDA complete")


if __name__ == "__main__":
    main()
