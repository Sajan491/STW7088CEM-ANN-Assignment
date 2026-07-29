"""Consolidate final results across both tasks.

Gathers the headline metrics produced by the earlier phases into one summary
table and a cross-task overview figure:

    Task 1 — tile classification (custom CNN):   accuracy, precision, recall, F1
    Task 2 — detection (baseline vs optimised):  mAP@0.5, mAP@0.5:0.95

Outputs:
    results/tables/final_summary.csv
    results/figures/final_summary.png

Runs on CPU. Missing inputs are skipped with a warning.

Usage:
    python -m src.evaluation.consolidate
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import plotting
from src.utils.config import resolve_path

TILE_CSV = "results/tables/tile_classifier_metrics.csv"
YOLO_BASE_CSV = "results/tables/yolo_baseline_metrics.csv"
YOLO_OPT_CSV = "results/tables/yolo_optimized_metrics.csv"


def _read(path: str):
    p = resolve_path(path)
    if not p.exists():
        print(f"[warn] {p} missing")
        return None
    return pd.read_csv(p).iloc[0]


def build_summary() -> pd.DataFrame:
    """Tidy long-form summary of both tasks' headline metrics."""
    rows = []
    tile = _read(TILE_CSV)
    if tile is not None:
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if m in tile:
                rows.append({"task": "Task 1 — tile CNN", "config": "custom CNN",
                             "metric": m, "value": round(float(tile[m]), 4)})
    for label, path in [("baseline (640)", YOLO_BASE_CSV), ("optimised (1024)", YOLO_OPT_CSV)]:
        r = _read(path)
        if r is not None:
            for m in ["mAP50", "mAP50_95", "precision", "recall"]:
                if m in r:
                    rows.append({"task": "Task 2 — YOLO", "config": label,
                                 "metric": m, "value": round(float(r[m]), 4)})
    return pd.DataFrame(rows)


def fig_summary(figures_dir) -> None:
    """Two-panel cross-task overview."""
    tile = _read(TILE_CSV)
    base = _read(YOLO_BASE_CSV)
    opt = _read(YOLO_OPT_CSV)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Task 1 — classification metrics
    if tile is not None:
        metrics = ["accuracy", "precision", "recall", "f1"]
        vals = [float(tile[m]) for m in metrics]
        colours = [plotting.BLUE if m != "accuracy" else plotting.BLUE_SOFT for m in metrics]
        bars = ax1.bar([m.capitalize() for m in metrics], vals, color=colours, zorder=3)
        plotting.bar_labels(ax1, bars, fmt="{:.3f}", fontsize=9.5)
        # Majority-class baseline reference — accuracy is misleading without it.
        ax1.axhline(0.835, color=plotting.RED, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
        ax1.text(3.4, 0.845, "majority-class\nbaseline 0.835", color=plotting.RED,
                 fontsize=8.5, ha="right", va="bottom")
        ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("score")
    plotting.titles(ax1, "Task 1 — tile classifier",
                    "custom CNN from scratch, test split")

    # Task 2 — baseline vs optimised mAP
    if base is not None and opt is not None:
        labels = ["mAP@0.5", "mAP@0.5:0.95"]
        base_v = [float(base["mAP50"]), float(base["mAP50_95"])]
        opt_v = [float(opt["mAP50"]), float(opt["mAP50_95"])]
        x = list(range(len(labels)))
        width = 0.38
        b1 = ax2.bar([i - width / 2 for i in x], base_v, width, label="baseline (640 px)",
                     color=plotting.BLUE_SOFT, zorder=3)
        b2 = ax2.bar([i + width / 2 for i in x], opt_v, width, label="optimised (1024 px + aug)",
                     color=plotting.GREEN, zorder=3)
        plotting.bar_labels(ax2, b1, fmt="{:.3f}", fontsize=9.5)
        plotting.bar_labels(ax2, b2, fmt="{:.3f}", fontsize=9.5)
        # Improvement arrows.
        for i, (bv, ov) in enumerate(zip(base_v, opt_v)):
            ax2.annotate(f"+{ov - bv:.3f}", (i + width / 2, ov),
                         textcoords="offset points", xytext=(0, 16), ha="center",
                         fontsize=8.5, color=plotting.GREEN, fontweight="bold")
        ax2.set_xticks(x, labels)
        ax2.set_ylim(0, 1.02)
        ax2.legend(loc="upper right")
    ax2.set_ylabel("mAP")
    plotting.titles(ax2, "Task 2 — detection",
                    "baseline vs optimised YOLOv8s, test split")

    fig.suptitle("Cross-task results overview", x=0.02, ha="left", y=1.02,
                 fontsize=14, fontweight="bold")
    plotting.save_figure(fig, "final_summary", figures_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate final cross-task results")
    ap.add_argument("--figures-dir", default="results/figures")
    ap.add_argument("--tables-dir", default="results/tables")
    args = ap.parse_args()

    plotting.apply_style()
    df = build_summary()
    if df.empty:
        print("[error] no metrics found; run the task evaluations first")
        return

    tables_dir = resolve_path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "final_summary.csv"
    df.to_csv(out, index=False)

    print("=== Final cross-task summary ===")
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
    print(f"[table] {out}")

    fig_summary(resolve_path(args.figures_dir))
    print("[done] consolidation complete")


if __name__ == "__main__":
    main()
