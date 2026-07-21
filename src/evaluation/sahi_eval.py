"""Evaluate a YOLO detector on the test split using SAHI sliced inference.

SAHI (Slicing Aided Hyper Inference) cuts each large aerial image into
overlapping windows, runs the detector on each window, and merges the results.
For small objects this recovers detail lost when a whole 4000x3000 frame is
squashed to the network input size.

Detections are scored with pycocotools (COCO mAP@0.5 and mAP@0.5:0.95) so the
numbers are computed the same way regardless of slicing; precision/recall are
reported at a fixed confidence/IoU operating point via a simple greedy matcher.
Run with --no-slice to evaluate the same model on full images through the SAME
pipeline — the delta between the two isolates SAHI's contribution.

Usage:
    python -m src.evaluation.sahi_eval --config configs/yolo_optimized.yaml
    python -m src.evaluation.sahi_eval --config configs/yolo_optimized.yaml --no-slice
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.coco_parser import CocoDataset
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def iou_xywh(a: list[float], b: list[float]) -> float:
    """IoU between two [x, y, w, h] boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def precision_recall_at(
    gt_by_img: dict[int, list], dt_by_img: dict[int, list],
    conf: float, iou_thr: float = 0.5,
) -> tuple[float, float]:
    """Greedy TP/FP/FN matching at a confidence + IoU threshold -> (precision, recall)."""
    tp = fp = 0
    n_gt = sum(len(v) for v in gt_by_img.values())
    for img_id, dts in dt_by_img.items():
        dts = sorted([d for d in dts if d["score"] >= conf], key=lambda d: -d["score"])
        gts = gt_by_img.get(img_id, [])
        matched = set()
        for d in dts:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                i = iou_xywh(d["bbox"], g)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= iou_thr and best_j >= 0:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return precision, recall


def build_coco_gt(ds: CocoDataset, names: list[str], images_dir: Path) -> tuple[dict, dict, list]:
    """Build a COCO ground-truth dict for the given image names.

    Returns (coco_gt_dict, gt_boxes_by_img_id, ordered_image_records) where
    image ids are dense 0..N-1 and boxes are [x, y, w, h] in raw pixel space.
    """
    by_name = {im["file_name"]: im for im in ds.images}
    images, annotations, gt_by_img, records = [], [], {}, []
    ann_id = 0
    for img_id, name in enumerate(sorted(names)):
        im = by_name[name]
        images.append({"id": img_id, "file_name": name,
                       "width": im["width"], "height": im["height"]})
        records.append({"id": img_id, "name": name, "path": images_dir / name})
        boxes = ds.bboxes_for_image(im["id"])
        gt_by_img[img_id] = boxes
        for bbox in boxes:
            x, y, w, h = bbox
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "rubbish"}],
    }
    return coco_gt, gt_by_img, records


def coco_map(coco_gt: dict, detections: list[dict]) -> tuple[float, float]:
    """mAP@0.5:0.95 and mAP@0.5 via pycocotools (silenced)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        if not detections:
            return 0.0, 0.0
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])  # mAP@.5:.95, mAP@.5


def main() -> None:
    ap = argparse.ArgumentParser(description="SAHI sliced-inference evaluation")
    ap.add_argument("--config", default="configs/yolo_optimized.yaml")
    ap.add_argument("--checkpoint", default=None, help="path to .pt (default: config)")
    ap.add_argument("--no-slice", action="store_true",
                    help="full-image inference through the same pipeline (control)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N test images")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    scfg = cfg["sahi"]
    sliced = not args.no_slice
    tag = "+SAHI" if sliced else "+augmentation (coco-eval)"

    ckpt = args.checkpoint or (
        resolve_path(cfg["eval"]["checkpoint_dir"]) / cfg["eval"]["checkpoint_name"]
    )
    # Test images in raw-pixel space (EXIF already normalised in the YOLO set).
    images_dir = resolve_path(cfg["data"]["yolo_dir"]) / "images" / "test"
    ds = CocoDataset(resolve_path(cfg["data"]["annotations_file"]))
    with open(resolve_path(cfg["data"]["splits_file"]), "r", encoding="utf-8") as f:
        names = json.load(f)["splits"]["test"]
    if args.limit:
        names = sorted(names)[: args.limit]

    device = args.device or "cuda:0"
    print("=== SAHI evaluation ===")
    print(f"  checkpoint: {ckpt} | sliced: {sliced} | images: {len(names)} | device: {device}")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(ckpt),
        confidence_threshold=scfg["confidence_threshold"],
        device=device,
    )

    coco_gt, gt_by_img, records = build_coco_gt(ds, names, images_dir)
    detections, dt_by_img = [], {}
    from tqdm import tqdm

    for rec in tqdm(records, desc="predict"):
        if sliced:
            result = get_sliced_prediction(
                str(rec["path"]), model,
                slice_height=scfg["slice_height"], slice_width=scfg["slice_width"],
                overlap_height_ratio=scfg["overlap_height_ratio"],
                overlap_width_ratio=scfg["overlap_width_ratio"],
                verbose=0,
            )
        else:
            result = get_prediction(str(rec["path"]), model)
        preds = result.to_coco_predictions(image_id=rec["id"])
        for p in preds:
            p["category_id"] = 1  # single class
            detections.append(p)
            dt_by_img.setdefault(rec["id"], []).append(
                {"bbox": p["bbox"], "score": p["score"]}
            )

    map5095, map50 = coco_map(coco_gt, detections)
    precision, recall = precision_recall_at(
        gt_by_img, dt_by_img, conf=scfg["confidence_threshold"], iou_thr=0.5
    )

    row = {
        "config": tag,
        "model": cfg["model"]["weights"],
        "imgsz": cfg["train"]["imgsz"],
        "split": "test",
        "sahi": sliced,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }
    out = resolve_path(scfg["metrics_file"])
    if args.no_slice:
        out = out.with_name(out.stem + "_noslice.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[table] {out}")
    for k, v in row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
