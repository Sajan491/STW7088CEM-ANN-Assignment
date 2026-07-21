"""Consolidate the Task 2 ablation into one table and a comparison figure.

Reads the per-arm metric CSVs produced by the evaluation scripts and assembles
the ablation the report needs:

    baseline (640) -> +resolution (1024) -> +augmentation   [Ultralytics eval]
    +augmentation full-image -> +SAHI sliced                [COCO / pycocotools eval]

The first three arms are the cumulative *training* progression, scored with
Ultralytics. SAHI is a *test-time* method scored with pycocotools, so its fair
reference is the same model on full images through the same pipeline
(the "full-image, COCO-eval" control) — not the Ultralytics number. Both eval
backends are labelled in the table so the comparison is transparent.

Outputs:
    results/tables/yolo_ablation.csv       (metrics + deltas vs baseline)
    results/figures/yolo_ablation.png      (grouped mAP bar chart)

Runs on CPU — it only reads CSVs. Any missing arm is skipped with a warning.

Usage:
    python -m src.evaluation.ablation
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import plotting
from src.utils.config import resolve_path

# arm label -> (csv path, eval backend). Order is the presentation order.
ARMS = [
    ("baseline (640)", "results/tables/yolo_baseline_metrics.csv", "ultralytics"),
    ("+resolution (1024)", "results/tables/yolo_res1024_metrics.csv", "ultralytics"),
    ("+augmentation (1024)", "results/tables/yolo_optimized_metrics.csv", "ultralytics"),
    ("+aug, full-image (COCO)", "results/tables/yolo_sahi_metrics_noslice.csv", "pycocotools"),
    ("+SAHI, sliced (COCO)", "results/tables/yolo_sahi_metrics.csv", "pycocotools"),
]
COLS = ["mAP50", "mAP50_95", "precision", "recall"]


def collect(arms) -> pd.DataFrame:
    """Read each arm's CSV (first row) into a single ordered DataFrame."""
    rows = []
    for label, path, backend in arms:
        p = resolve_path(path)
        if not p.exists():
            print(f"[warn] {p} missing; skipping '{label}'")
            continue
        r = pd.read_csv(p).iloc[0]
        rows.append({
            "arm": label,
            "eval": backend,
            "imgsz": int(r.get("imgsz", 0)),
            **{c: round(float(r[c]), 4) for c in COLS if c in r},
        })
    return pd.DataFrame(rows)


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Append change vs the baseline (first row) for each metric."""
    if df.empty:
        return df
    base = df.iloc[0]
    for c in COLS:
        if c in df:
            df[f"d_{c}"] = (df[c] - base[c]).round(4)
    return df


def fig_ablation(df: pd.DataFrame, figures_dir) -> None:
    """Grouped bar chart of mAP@0.5 and mAP@0.5:0.95 across arms.

    A divider separates the Ultralytics-scored training arms from the
    pycocotools-scored SAHI comparison so the two eval backends aren't
    compared directly by eye.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.38
    b1 = ax.bar([i - width / 2 for i in x], df["mAP50"], width,
                label="mAP@0.5", color=plotting.BLUE)
    b2 = ax.bar([i + width / 2 for i in x], df["mAP50_95"], width,
                label="mAP@0.5:0.95", color=plotting.GREEN)
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.006,
                    f"{rect.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color=plotting.INK_SECONDARY)

    # Divider between Ultralytics arms and the pycocotools SAHI comparison.
    n_ul = int((df["eval"] == "ultralytics").sum())
    if 0 < n_ul < len(df):
        ax.axvline(n_ul - 0.5, color=plotting.INK_MUTED, linestyle=":", linewidth=1)
        ax.text(n_ul / 2 - 0.5, 1.03, "training (Ultralytics eval)",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                color=plotting.INK_MUTED)
        ax.text((n_ul + len(df)) / 2 - 0.5, 1.03, "test-time SAHI (COCO eval)",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                color=plotting.INK_MUTED)

    ax.set_xticks(list(x), df["arm"], rotation=18, ha="right")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, max(df["mAP50"].max(), df["mAP50_95"].max()) * 1.22)
    ax.set_title("Task 2 ablation: contribution of each modification", pad=34)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.97))
    plotting.save_figure(fig, "yolo_ablation", figures_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the Task 2 ablation table + figure")
    ap.add_argument("--figures-dir", default="results/figures")
    ap.add_argument("--tables-dir", default="results/tables")
    args = ap.parse_args()

    plotting.apply_style()
    df = collect(ARMS)
    if df.empty:
        print("[error] no arm metrics found; run the evaluations first")
        return
    df = add_deltas(df)

    tables_dir = resolve_path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "yolo_ablation.csv"
    df.to_csv(out_csv, index=False)

    print("=== Task 2 ablation ===")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(df.to_string(index=False))
    print(f"[table] {out_csv}")

    fig_ablation(df, resolve_path(args.figures_dir))
    print("[done] ablation consolidated")


if __name__ == "__main__":
    main()
