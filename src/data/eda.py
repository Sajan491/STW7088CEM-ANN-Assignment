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


def fig_object_sizes(fracs: np.ndarray, figures_dir) -> None:
    """Histogram of bbox area as a fraction of image area (log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pct = fracs * 100
    bins = np.logspace(np.log10(pct.min()), np.log10(pct.max()), 40)
    ax.hist(pct, bins=bins, color=plotting.BLUE, edgecolor=plotting.SURFACE, linewidth=0.4)
    ax.set_xscale("log")

    median = np.median(pct)
    ax.axvline(median, color=plotting.INK_SECONDARY, linewidth=1.2, linestyle="--")
    ax.text(
        median * 1.15,
        ax.get_ylim()[1] * 0.92,
        f"median {median:.3f}%",
        color=plotting.INK_SECONDARY,
        fontsize=9,
    )
    ax.axvline(1.0, color=plotting.INK_MUTED, linewidth=1.0, linestyle=":")
    ax.text(1.1, ax.get_ylim()[1] * 0.78, "1% of image", color=plotting.INK_MUTED, fontsize=9)

    ax.set_xlabel("Bounding-box area as % of image area (log scale)")
    ax.set_ylabel("Number of annotations")
    ax.set_title("UAVVaste litter objects are extremely small relative to the frame")
    plotting.save_figure(fig, "object_size_distribution", figures_dir)
    plt.close(fig)


def fig_annotations_per_image(counts: np.ndarray, figures_dir) -> None:
    """Histogram of annotations per image."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.arange(0, counts.max() + 2) - 0.5
    ax.hist(counts, bins=bins, color=plotting.BLUE, edgecolor=plotting.SURFACE, linewidth=0.4)

    mean, median = counts.mean(), np.median(counts)
    ax.axvline(mean, color=plotting.INK_SECONDARY, linewidth=1.2, linestyle="--")
    ax.text(
        mean + 0.6,
        ax.get_ylim()[1] * 0.92,
        f"mean {mean:.1f} / median {median:.0f}",
        color=plotting.INK_SECONDARY,
        fontsize=9,
    )
    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Litter annotations per UAVVaste image")
    plotting.save_figure(fig, "annotations_per_image", figures_dir)
    plt.close(fig)


def fig_tile_balance(stats: dict, figures_dir) -> None:
    """Grouped bars: litter vs no-litter tile counts per split."""
    splits = [s for s in ("train", "val", "test") if s in stats]
    pos = [stats[s]["pos"] for s in splits]
    neg = [stats[s]["neg"] for s in splits]

    x = np.arange(len(splits))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - width / 2, pos, width, label="litter", color=plotting.BLUE)
    b2 = ax.bar(x + width / 2, neg, width, label="no litter", color=plotting.GREEN)

    for bars in (b1, b2):
        for rect in bars:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{int(rect.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=plotting.INK_SECONDARY,
            )
    for i, s in enumerate(splits):
        ax.text(
            i,
            -0.14,
            f"{stats[s]['pos_frac']:.1%} positive",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=plotting.INK_MUTED,
        )

    ax.set_xticks(x, [s.capitalize() for s in splits])
    ax.set_ylabel("Number of tiles")
    ax.set_title("Tile class balance per split (image-level splits)")
    ax.legend()
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
