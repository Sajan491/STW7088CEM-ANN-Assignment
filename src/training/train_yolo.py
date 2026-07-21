"""Fine-tune a pretrained Ultralytics YOLO detector on UAVVaste.

Config-driven (configs/yolo_baseline.yaml). Trains from pretrained weights,
copies the best checkpoint into checkpoints/ and the Ultralytics training
plots (results curve, confusion matrix, PR curve) into results/figures/, and
records the validation metrics.

Usage:
    python -m src.training.train_yolo
    python -m src.training.train_yolo --config configs/yolo_baseline.yaml
    python -m src.training.train_yolo --epochs 3 --limit-note smoke   # quick run
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

# Ultralytics plot files worth keeping as report evidence. Curve files are
# named "PR_curve.png" in older Ultralytics and "BoxPR_curve.png" in newer
# releases, so both spellings are listed.
PLOT_FILES = {
    "results.png": "results",
    "confusion_matrix.png": "confusion_matrix",
    "confusion_matrix_normalized.png": "confusion_matrix_normalized",
    "PR_curve.png": "pr_curve",
    "BoxPR_curve.png": "pr_curve",
    "F1_curve.png": "f1_curve",
    "BoxF1_curve.png": "f1_curve",
    "labels.jpg": "labels",
}


def copy_outputs(run_dir: Path, prefix: str, figures_dir: Path, ckpt_dir: Path,
                 ckpt_name: str) -> None:
    """Copy Ultralytics run plots + best weights into the repo's results/checkpoints."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_stem in PLOT_FILES.items():
        src = run_dir / src_name
        if src.exists():
            shutil.copy2(src, figures_dir / f"{prefix}_{dst_stem}{src.suffix}")
            print(f"[fig] {figures_dir / f'{prefix}_{dst_stem}{src.suffix}'}")
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, ckpt_dir / ckpt_name)
        print(f"[ckpt] {ckpt_dir / ckpt_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO on UAVVaste")
    ap.add_argument("--config", default="configs/yolo_baseline.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default=None, help="cuda device or 'cpu'")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tcfg = cfg["train"]
    epochs = args.epochs or tcfg["epochs"]

    print("=== YOLO training ===")
    print(f"  model: {cfg['model']['weights']} | imgsz: {tcfg['imgsz']} | epochs: {epochs}")

    model = YOLO(cfg["model"]["weights"])

    train_kwargs = dict(
        data=str(resolve_path(cfg["data"]["data_yaml"])),
        imgsz=tcfg["imgsz"],
        epochs=epochs,
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        optimizer=tcfg.get("optimizer", "auto"),
        lr0=tcfg.get("lr0", 0.01),
        seed=tcfg.get("seed", cfg["seed"]),
        project=tcfg["project"],
        name=tcfg["name"],
        exist_ok=True,
        verbose=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    # Optional augmentation overrides (empty for baseline).
    train_kwargs.update(tcfg.get("augment") or {})

    results = model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    print(f"[done] training complete -> {run_dir}")

    ecfg = cfg["eval"]
    copy_outputs(
        run_dir,
        prefix=tcfg["name"],
        figures_dir=resolve_path(ecfg["figures_dir"]),
        ckpt_dir=resolve_path(ecfg["checkpoint_dir"]),
        ckpt_name=ecfg["checkpoint_name"],
    )

    # Persist the best-epoch validation metrics.
    metrics = {
        "model": cfg["model"]["weights"],
        "imgsz": tcfg["imgsz"],
        "epochs_run": epochs,
        "split": "val",
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
    }
    tables_dir = resolve_path(ecfg["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = tables_dir / "yolo_baseline_val_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(f"[table] {out}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
