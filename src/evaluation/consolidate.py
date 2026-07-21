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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Task 1 — classification metrics
    if tile is not None:
        metrics = ["accuracy", "precision", "recall", "f1"]
        vals = [float(tile[m]) for m in metrics]
        bars = ax1.bar(metrics, vals, color=plotting.BLUE)
        for rect in bars:
            ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                     f"{rect.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                     color=plotting.INK_SECONDARY)
        ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("score")
    ax1.set_title("Task 1 — tile classifier (custom CNN, test)")

    # Task 2 — baseline vs optimised mAP
    if base is not None and opt is not None:
        labels = ["mAP@0.5", "mAP@0.5:0.95"]
        base_v = [float(base["mAP50"]), float(base["mAP50_95"])]
        opt_v = [float(opt["mAP50"]), float(opt["mAP50_95"])]
        x = range(len(labels))
        width = 0.38
        b1 = ax2.bar([i - width / 2 for i in x], base_v, width, label="baseline (640)",
                     color=plotting.INK_MUTED)
        b2 = ax2.bar([i + width / 2 for i in x], opt_v, width, label="optimised (1024+aug)",
                     color=plotting.GREEN)
        for bars in (b1, b2):
            for rect in bars:
                ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.008,
                         f"{rect.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                         color=plotting.INK_SECONDARY)
        ax2.set_xticks(list(x), labels)
        ax2.set_ylim(0, 1.0)
        ax2.legend()
    ax2.set_ylabel("mAP")
    ax2.set_title("Task 2 — detection (baseline vs optimised, test)")

    fig.suptitle("Cross-task results overview", y=1.02, fontsize=13)
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
