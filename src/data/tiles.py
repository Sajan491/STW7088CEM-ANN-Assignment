"""Grid tiling of aerial images with litter/no-litter labels.

Each image is cut into square tiles on a regular grid. A tile is labelled
positive (litter) when any annotation bbox overlaps it "enough":

    intersection_area / bbox_area >= min_overlap_frac
    OR intersection_area >= min_overlap_px

The fractional rule captures small objects (the UAVVaste norm) that fall
inside a tile; the absolute rule keeps a tile positive when a larger object
crosses it with a substantial chunk even if that chunk is a small fraction
of the object. Tiles inherit the IMAGE-level split of their source image.

Usage:
    python -m src.data.tiles                   # full run (requires images)
    python -m src.data.tiles --limit 10        # smoke test on 10 images
    python -m src.data.tiles --dry-run         # labels/stats only, no image IO
"""

import argparse
import json
from pathlib import Path

from src.data.coco_parser import CocoDataset
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def tile_grid(width: int, height: int, size: int, stride: int) -> list[tuple[int, int]]:
    """Tile origins (x0, y0) covering the image.

    The grid steps by `stride`; if the image edge is not hit exactly, one
    final tile is anchored flush with the edge so every pixel is covered.
    Images smaller than `size` yield no tiles.
    """
    if width < size or height < size:
        return []

    def axis_origins(dim: int) -> list[int]:
        origins = list(range(0, dim - size + 1, stride))
        if origins[-1] + size < dim:
            origins.append(dim - size)
        return origins

    return [(x, y) for y in axis_origins(height) for x in axis_origins(width)]


def intersection_area(
    tile: tuple[int, int, int, int], bbox: list[float]
) -> float:
    """Intersection area between tile (x0, y0, x1, y1) and COCO bbox [x, y, w, h]."""
    tx0, ty0, tx1, ty1 = tile
    bx, by, bw, bh = bbox
    ix = max(0.0, min(tx1, bx + bw) - max(tx0, bx))
    iy = max(0.0, min(ty1, by + bh) - max(ty0, by))
    return ix * iy


def label_tile(
    tile: tuple[int, int, int, int],
    bboxes: list[list[float]],
    min_overlap_frac: float,
    min_overlap_px: float,
) -> bool:
    """True (litter) if any bbox overlaps the tile beyond either threshold."""
    for bbox in bboxes:
        inter = intersection_area(tile, bbox)
        if inter <= 0:
            continue
        area = bbox[2] * bbox[3]
        if area > 0 and inter / area >= min_overlap_frac:
            return True
        if inter >= min_overlap_px:
            return True
    return False


def build_tile_index(
    ds: CocoDataset,
    splits: dict[str, list[str]],
    size: int,
    stride: int,
    min_overlap_frac: float,
    min_overlap_px: float,
    limit: int | None = None,
) -> list[dict]:
    """Compute every tile's position, label and split (no image IO needed)."""
    split_of = {
        name: split for split, names in splits.items() for name in names
    }
    index = []
    images = sorted(ds.images, key=lambda im: im["file_name"])
    if limit:
        images = images[:limit]
    for im in images:
        split = split_of.get(im["file_name"])
        if split is None:
            continue  # image not in any split (e.g. smoke-test splits)
        bboxes = ds.bboxes_for_image(im["id"])
        for x0, y0 in tile_grid(im["width"], im["height"], size, stride):
            tile_box = (x0, y0, x0 + size, y0 + size)
            positive = label_tile(tile_box, bboxes, min_overlap_frac, min_overlap_px)
            index.append(
                {
                    "image": im["file_name"],
                    "split": split,
                    "x": x0,
                    "y": y0,
                    "size": size,
                    "label": int(positive),
                }
            )
    return index


def write_tile_crops(index: list[dict], images_dir: Path, out_dir: Path) -> int:
    """Cut and save tile crops as out_dir/<split>/<pos|neg>/<stem>_<x>_<y>.jpg."""
    import cv2

    written = 0
    by_image: dict[str, list[dict]] = {}
    for t in index:
        by_image.setdefault(t["image"], []).append(t)

    from tqdm import tqdm

    for name, tiles in tqdm(by_image.items(), desc="writing tile crops"):
        # IMREAD_IGNORE_ORIENTATION: COCO annotations (and image sizes) are in
        # RAW pixel space; some UAVVaste images carry EXIF rotation tags that
        # cv2 would otherwise apply, swapping width/height vs the annotations.
        img = cv2.imread(
            str(images_dir / name), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        if img is None:
            print(f"[warn] missing image {name}; skipped {len(tiles)} tiles")
            continue
        stem = Path(name).stem
        for t in tiles:
            crop = img[t["y"] : t["y"] + t["size"], t["x"] : t["x"] + t["size"]]
            if crop.shape[0] != t["size"] or crop.shape[1] != t["size"]:
                print(f"[warn] {name}: tile ({t['x']},{t['y']}) out of bounds; skipped")
                continue
            cls = "pos" if t["label"] else "neg"
            dest = out_dir / t["split"] / cls / f"{stem}_{t['x']}_{t['y']}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    return written


def class_balance(index: list[dict]) -> dict:
    """Positive/negative tile counts, overall and per split."""
    stats: dict = {"total": {"pos": 0, "neg": 0}}
    for t in index:
        stats.setdefault(t["split"], {"pos": 0, "neg": 0})
        key = "pos" if t["label"] else "neg"
        stats[t["split"]][key] += 1
        stats["total"][key] += 1
    for s in stats.values():
        n = s["pos"] + s["neg"]
        s["pos_frac"] = round(s["pos"] / n, 4) if n else 0.0
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate labelled tiles from UAVVaste")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute tile labels and stats only; do not read or write images",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    tcfg = cfg["tiles"]

    ds = CocoDataset(resolve_path(cfg["dataset"]["annotations_file"]))
    with open(resolve_path(cfg["splits"]["file"]), "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    print("=== Tile generation ===")
    print(f"  tile size {tcfg['size']}, stride {tcfg['stride']}, "
          f"overlap frac >= {tcfg['min_overlap_frac']} or px >= {tcfg['min_overlap_px']}")

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
        print(f"  {split}: {s['pos']} pos / {s['neg']} neg ({s['pos_frac']:.1%} positive)")

    out_dir = resolve_path(tcfg["out_dir"])
    index_file = resolve_path(tcfg["index_file"])
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump({"config": tcfg, "stats": stats, "tiles": index}, f)
    print(f"[done] tile index ({len(index)} tiles) written to {index_file}")

    if args.dry_run:
        print("[info] --dry-run set; skipping crop writing")
        return

    written = write_tile_crops(index, resolve_path(cfg["dataset"]["images_dir"]), out_dir)
    print(f"[done] {written} tile crops written to {out_dir}")


if __name__ == "__main__":
    main()
