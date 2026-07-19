"""Leakage-free train/val/test splits at IMAGE level.

Splitting happens on whole images before any tiling, so every tile cut from
an image inherits that image's split — tiles from one image can never appear
in two splits. The assignment is a deterministic seeded shuffle, written to
JSON together with the seed and ratios that produced it.

Usage:
    python -m src.data.splits
    python -m src.data.splits --limit 50   # smoke test on the first 50 images
"""

import argparse
import json
import random
from pathlib import Path

from src.data.coco_parser import CocoDataset
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def make_splits(
    file_names: list[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Deterministically partition file names into train/val/test."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "split ratios must sum to 1"
    names = sorted(file_names)  # fixed order before shuffling -> reproducible
    rng = random.Random(seed)
    rng.shuffle(names)

    n = len(names)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    return {
        "train": sorted(names[:n_train]),
        "val": sorted(names[n_train : n_train + n_val]),
        "test": sorted(names[n_train + n_val :]),
    }


def verify_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise if any image appears in more than one split."""
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    overlaps = (train & val) | (train & test) | (val & test)
    if overlaps:
        raise ValueError(f"split leakage: {len(overlaps)} images in multiple splits")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate image-level dataset splits")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use first N images")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)

    ds = CocoDataset(resolve_path(cfg["dataset"]["annotations_file"]))
    names = sorted(ds.file_names())
    if args.limit:
        names = names[: args.limit]
        print(f"[smoke] limited to first {len(names)} images")

    splits = make_splits(names, cfg["splits"]["ratios"], seed)
    verify_disjoint(splits)

    out = resolve_path(cfg["splits"]["file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "ratios": cfg["splits"]["ratios"],
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Image-level splits ===")
    total = sum(len(v) for v in splits.values())
    for name, files in splits.items():
        print(f"  {name}: {len(files)} images ({len(files) / total:.1%})")
    print(f"  disjoint: OK")
    print(f"[done] written to {out}")


if __name__ == "__main__":
    main()
