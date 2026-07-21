"""Evaluate a fine-tuned YOLO detector on the held-out UAVVaste test split.

Reports the detection metrics required by the brief — mAP@0.5, mAP@0.5:0.95,
precision, recall — and writes them to results/tables/. Uses the checkpoint
saved by training unless another is given.

Usage:
    python -m src.evaluation.evaluate_yolo
    python -m src.evaluation.evaluate_yolo --checkpoint checkpoints/yolo_baseline.pt
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate YOLO on the UAVVaste test split")
    ap.add_argument("--config", default="configs/yolo_baseline.yaml")
    ap.add_argument("--checkpoint", default=None, help="path to .pt (default: config checkpoint)")
    ap.add_argument("--split", default=None, help="override eval split (default: config)")
    ap.add_argument("--device", default=None, help="cuda device or 'cpu'")
    ap.add_argument("--tag", default="baseline", help="label used in the metrics row/file")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ecfg = cfg["eval"]
    split = args.split or ecfg["split"]

    ckpt = args.checkpoint or (
        resolve_path(ecfg["checkpoint_dir"]) / ecfg["checkpoint_name"]
    )
    print("=== YOLO evaluation ===")
    print(f"  checkpoint: {ckpt} | split: {split} | imgsz: {cfg['train']['imgsz']}")

    model = YOLO(str(ckpt))
    val_kwargs = dict(
        data=str(resolve_path(cfg["data"]["data_yaml"])),
        split=split,
        imgsz=cfg["train"]["imgsz"],
        project=cfg["train"]["project"],
        name=f"{cfg['train']['name']}_eval_{split}",
        exist_ok=True,
        verbose=True,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)

    row = {
        "config": args.tag,
        "model": cfg["model"]["weights"],
        "imgsz": cfg["train"]["imgsz"],
        "split": split,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }

    tables_dir = resolve_path(ecfg["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = resolve_path(ecfg["metrics_file"])
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[table] {out}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
