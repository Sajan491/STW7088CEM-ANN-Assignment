"""Convert UAVVaste COCO annotations to Ultralytics YOLO format.

Builds the directory layout Ultralytics expects, reusing the leakage-free
IMAGE-level splits from Phase 1:

    data/yolo/
      images/{train,val,test}/*.jpg
      labels/{train,val,test}/*.txt   # "0 xc yc w h" (normalised)
      data.yaml

EXIF handling (important): the COCO boxes are in RAW pixel space, but
Ultralytics applies EXIF orientation when loading an image. 29 UAVVaste
images carry rotation tags, which would misalign their boxes. We therefore
write every image into the YOLO set in raw-pixel space with the orientation
tag stripped (via OpenCV's IMREAD_IGNORE_ORIENTATION), so pixels and labels
share one coordinate system. Images with no rotation are copied as-is
(lossless) for speed.

Usage:
    python -m src.data.coco_to_yolo
    python -m src.data.coco_to_yolo --limit 20   # smoke test on 20 images
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

from src.data.coco_parser import CocoDataset
from src.utils.config import load_config, resolve_path

EXIF_ORIENTATION_TAG = 274
ROTATED = {3, 5, 6, 7, 8}  # orientations that change displayed pixels


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """COCO [x, y, w, h] (raw px) -> YOLO (xc, yc, w, h) normalised to [0, 1]."""
    x, y, bw, bh = bbox
    xc = (x + bw / 2) / img_w
    yc = (y + bh / 2) / img_h
    return xc, yc, bw / img_w, bh / img_h


def _needs_orientation_fix(path: Path) -> bool:
    """True if the image carries an EXIF rotation that OpenCV/PIL would apply."""
    try:
        with Image.open(path) as im:
            return im.getexif().get(EXIF_ORIENTATION_TAG, 1) in ROTATED
    except Exception:
        return False


def write_image(src: Path, dst: Path) -> None:
    """Copy an image into the YOLO set in raw-pixel space (no EXIF rotation)."""
    if _needs_orientation_fix(src):
        # Re-encode raw pixels without the orientation tag.
        img = cv2.imread(str(src), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        shutil.copy2(src, dst)


def convert(cfg: dict, limit: int | None = None) -> dict:
    """Build the YOLO dataset. Returns per-split image/label counts."""
    ds = CocoDataset(resolve_path(cfg["data"]["annotations_file"]))
    images_dir = resolve_path(cfg["data"]["images_dir"])
    yolo_dir = resolve_path(cfg["data"]["yolo_dir"])
    with open(resolve_path(cfg["data"]["splits_file"]), "r", encoding="utf-8") as f:
        splits = json.load(f)["splits"]

    img_by_name = {im["file_name"]: im for im in ds.images}
    counts: dict = {}

    for split, names in splits.items():
        img_out = yolo_dir / "images" / split
        lbl_out = yolo_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        names = sorted(names)
        if limit:
            names = names[:limit]
        n_img = n_box = 0
        for name in tqdm(names, desc=f"{split:>5}"):
            im = img_by_name[name]
            src = images_dir / name
            if not src.exists():
                print(f"[warn] missing image {name}; skipped")
                continue
            write_image(src, img_out / name)

            lines = []
            for bbox in ds.bboxes_for_image(im["id"]):
                xc, yc, w, h = coco_bbox_to_yolo(bbox, im["width"], im["height"])
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (lbl_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
        counts[split] = {"images": n_img, "boxes": n_box}
        print(f"  {split}: {n_img} images, {n_box} boxes")

    write_data_yaml(cfg, yolo_dir)
    return counts


def write_data_yaml(cfg: dict, yolo_dir: Path) -> Path:
    """Write the Ultralytics dataset descriptor (data.yaml)."""
    out = resolve_path(cfg["data"]["data_yaml"])
    # Ultralytics resolves train/val/test relative to `path`.
    content = (
        f"path: {yolo_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n  0: {cfg['data']['class_name']}\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print(f"[done] data.yaml -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert UAVVaste COCO to YOLO format")
    ap.add_argument("--config", default="configs/yolo_baseline.yaml")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images per split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("=== COCO -> YOLO conversion ===")
    convert(cfg, limit=args.limit)
    print("[done] YOLO dataset ready")


if __name__ == "__main__":
    main()
