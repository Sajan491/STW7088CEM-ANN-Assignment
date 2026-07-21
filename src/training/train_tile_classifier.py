"""Train the custom CNN tile classifier.

Handles the ~15%-positive class imbalance with a weighted BCE loss (or a
balanced sampler), reduces the learning rate on plateau, stops early when
the monitored validation metric stops improving, and checkpoints the best
model. Per-epoch history is written to CSV for the learning-curve figure.

Usage:
    python -m src.training.train_tile_classifier
    python -m src.training.train_tile_classifier --limit 400 --epochs 2   # smoke test
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.tile_dataset import TileDataset
from src.evaluation.metrics import binary_metrics
from src.models.factory import build_model
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def make_loaders(cfg: dict, limit: int | None) -> tuple[DataLoader, DataLoader, TileDataset]:
    dcfg = cfg["data"]
    tiles_dir = resolve_path(dcfg["tiles_dir"])
    train_ds = TileDataset(
        tiles_dir, "train", dcfg["input_size"], augment=dcfg.get("augment"), limit=limit
    )
    val_ds = TileDataset(tiles_dir, "val", dcfg["input_size"], limit=limit)

    sampler = None
    shuffle = True
    if cfg["training"]["imbalance"] == "balanced_sampler":
        sampler = train_ds.balanced_sampler()
        shuffle = False  # sampler and shuffle are mutually exclusive

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=dcfg.get("num_workers", 2),
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=dcfg.get("num_workers", 2),
        pin_memory=True,
    )
    return train_dl, val_dl, train_ds


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict]:
    """One pass over a loader. With an optimizer -> train, else eval.

    Returns (mean loss, metrics dict).
    """
    training = optimizer is not None
    model.train(training)
    losses, probs, targets = [], [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(y.cpu().numpy())
    metrics = binary_metrics(np.concatenate(targets), np.concatenate(probs))
    return float(np.mean(losses)), metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tile litter classifier")
    ap.add_argument("--config", default="configs/tile_classifier.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N tiles per split")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tcfg = cfg["training"]
    epochs = args.epochs or tcfg["epochs"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=== Tile classifier training ===")
    print(f"  device: {device}")

    train_dl, val_dl, train_ds = make_loaders(cfg, args.limit)
    labels = train_ds.labels()
    print(f"  train tiles: {len(train_ds)} ({sum(labels)} pos / {len(labels) - sum(labels)} neg)")
    print(f"  val tiles:   {len(val_dl.dataset)}")

    model = build_model(cfg["model"]).to(device)
    print(f"  model: {cfg['model']['name']} ({model.num_parameters():,} trainable params)")

    if tcfg["imbalance"] == "weighted_loss":
        pos_weight = train_ds.pos_weight().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  imbalance: weighted BCE loss (pos_weight={pos_weight.item():.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("  imbalance: balanced sampler")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    monitor = tcfg["early_stopping"]["monitor"]  # val_f1 | val_loss
    mode_max = monitor != "val_loss"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if mode_max else "min",
        factor=tcfg["scheduler"]["factor"],
        patience=tcfg["scheduler"]["patience"],
        min_lr=tcfg["scheduler"]["min_lr"],
    )

    ckpt_dir = resolve_path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / tcfg["checkpoint_name"]

    best_value = -np.inf if mode_max else np.inf
    best_epoch = 0
    patience = tcfg["early_stopping"]["patience"]
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_dl, criterion, device, optimizer)
        val_loss, val_m = run_epoch(model, val_dl, criterion, device)

        value = val_loss if monitor == "val_loss" else val_m["f1"]
        scheduler.step(value)
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_m["accuracy"],
            "train_f1": train_m["f1"],
            "val_loss": val_loss,
            "val_acc": val_m["accuracy"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"  epoch {epoch:3d}/{epochs} | lr {lr:.1e} | "
            f"train loss {train_loss:.4f} f1 {train_m['f1']:.3f} | "
            f"val loss {val_loss:.4f} f1 {val_m['f1']:.3f} "
            f"prec {val_m['precision']:.3f} rec {val_m['recall']:.3f} "
            f"({row['seconds']}s)"
        )

        improved = value > best_value if mode_max else value < best_value
        if improved:
            best_value, best_epoch = value, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": cfg["model"],
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "seed": cfg["seed"],
                },
                ckpt_path,
            )
            print(f"    [ckpt] best {monitor}={best_value:.4f} -> {ckpt_path}")
        elif epoch - best_epoch >= patience:
            print(f"  [early stop] no {monitor} improvement for {patience} epochs")
            break

    hist_file = resolve_path(tcfg["history_file"])
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_file, index=False)
    print(f"[done] best {monitor}={best_value:.4f} at epoch {best_epoch}")
    print(f"[done] history -> {hist_file}")
    print(f"[done] checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
