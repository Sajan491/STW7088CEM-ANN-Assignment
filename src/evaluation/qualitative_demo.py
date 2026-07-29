"""Qualitative demonstration on unseen (test-split) aerial images.

For each selected image, produces a three-panel figure:

    ground truth  |  YOLO detections  |  tile-classifier litter heatmap

This shows the two tasks working together — the detector localises individual
litter items while the classifier flags litter-bearing regions — the two-stage
deployment pattern from the proposal.

Both models are loaded from their checkpoints (not committed). Runs on CPU.

Usage:
    python -m src.evaluation.qualitative_demo --num-images 4
    python -m src.evaluation.qualitative_demo --images BATCH_d07_img_580.jpg
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.coco_parser import CocoDataset
from src.evaluation.heatmap import tile_litter_heatmap
from src.models.factory import build_model
from src.utils import plotting
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def load_tile_model(tile_cfg: dict, device):
    ckpt_path = (resolve_path(tile_cfg["training"]["checkpoint_dir"])
                 / tile_cfg["training"]["checkpoint_name"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt_path


def select_images(ds: CocoDataset, test_names: list[str], n: int) -> list[str]:
    """Pick the n test images with the most annotations (visually informative)."""
    by_name = {im["file_name"]: im for im in ds.images}
    counts = [(name, len(ds.anns_for_image(by_name[name]["id"]))) for name in test_names]
    counts.sort(key=lambda t: (-t[1], t[0]))
    return [name for name, _ in counts[:n]]


def read_raw(path: Path) -> np.ndarray:
    """Read an image in raw-pixel (EXIF-ignored) space as RGB, matching training."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(ax, image, boxes, color, title, subtitle):
    """Show the image with boxes; each box has a subtle white halo for contrast."""
    import matplotlib.patheffects as pe

    ax.imshow(image)
    halo = [pe.Stroke(linewidth=3.2, foreground="white", alpha=0.7), pe.Normal()]
    for (x, y, w, h) in boxes:
        r = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.8)
        r.set_path_effects(halo)
        ax.add_patch(r)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=color, pad=22)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom", fontsize=9.5,
                color=plotting.INK_MUTED)
    ax.axis("off")


def demo_image(name, ds, images_dir, tile_model, yolo_model, cfg_tile, cfg_yolo,
               tile_px, conf, heatmap_stride, device, out_dir):
    by_name = {im["file_name"]: im for im in ds.images}
    im_meta = by_name[name]
    image = read_raw(images_dir / name)
    gt_boxes = ds.bboxes_for_image(im_meta["id"])

    # YOLO detections (raw-space image, optimised model resolution)
    res = yolo_model.predict(image[:, :, ::-1], imgsz=cfg_yolo["train"]["imgsz"],
                             conf=conf, verbose=False)[0]
    det_boxes = []
    for b in res.boxes.xyxy.cpu().numpy():
        det_boxes.append((b[0], b[1], b[2] - b[0], b[3] - b[1]))

    # Tile-classifier heatmap (slide the trained 512 px tiles, overlapping)
    heat = tile_litter_heatmap(
        image, tile_model,
        tile_size=tile_px,
        stride=heatmap_stride,
        input_size=cfg_tile["data"]["input_size"],
        device=device,
    )

    aspect = image.shape[0] / image.shape[1]  # keep the figure close to the image shape
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 3.2 + 4.2 * aspect))
    draw_boxes(axes[0], image, gt_boxes, plotting.GREEN,
               "Ground truth", f"{len(gt_boxes)} labelled litter items")
    draw_boxes(axes[1], image, det_boxes, plotting.BLUE,
               "YOLO detections", f"{len(det_boxes)} found at confidence {conf}")
    axes[2].imshow(image)
    hm = axes[2].imshow(heat, cmap="turbo", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Tile-classifier heatmap", loc="left", fontsize=12.5,
                      fontweight="bold", color=plotting.ORANGE, pad=22)
    axes[2].annotate("region-level litter probability", xy=(0, 1), xycoords="axes fraction",
                     xytext=(0, 5), textcoords="offset points", ha="left", va="bottom",
                     fontsize=9.5, color=plotting.INK_MUTED)
    axes[2].axis("off")
    cbar = fig.colorbar(hm, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.set_label("litter probability", fontsize=9)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Two-stage pipeline on an unseen image  —  the detector localises items, "
                 f"the classifier flags regions", y=0.99, fontsize=13,
                 fontweight="bold", color=plotting.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = plotting.save_figure(fig, f"demo_{Path(name).stem}", out_dir)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitative demo: detections + tile heatmap")
    ap.add_argument("--tile-config", default="configs/tile_classifier.yaml")
    ap.add_argument("--yolo-config", default="configs/yolo_optimized.yaml")
    ap.add_argument("--num-images", type=int, default=4)
    ap.add_argument("--images", nargs="*", default=None, help="explicit image file names")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--heatmap-stride", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg_tile = load_config(args.tile_config)
    cfg_yolo = load_config(args.yolo_config)
    cfg_data = load_config("configs/data.yaml")
    set_seed(cfg_tile["seed"])
    plotting.apply_style()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tile_px = cfg_data["tiles"]["size"]  # physical tile size the classifier was trained on

    ds = CocoDataset(resolve_path(cfg_data["dataset"]["annotations_file"]))
    images_dir = resolve_path(cfg_data["dataset"]["images_dir"])
    with open(resolve_path(cfg_data["splits"]["file"]), "r", encoding="utf-8") as f:
        test_names = json.load(f)["splits"]["test"]

    names = args.images or select_images(ds, test_names, args.num_images)
    print("=== Qualitative demo ===")
    tile_model, tile_ckpt = load_tile_model(cfg_tile, device)
    yolo_ckpt = resolve_path(cfg_yolo["eval"]["checkpoint_dir"]) / cfg_yolo["eval"]["checkpoint_name"]
    yolo_model = YOLO(str(yolo_ckpt))
    print(f"  tile model: {tile_ckpt}")
    print(f"  yolo model: {yolo_ckpt}")
    print(f"  images: {names}")

    out_dir = resolve_path("results/figures/qualitative")
    for name in names:
        out = demo_image(name, ds, images_dir, tile_model, yolo_model, cfg_tile,
                         cfg_yolo, tile_px, args.conf, args.heatmap_stride, device, out_dir)
        print(f"  [demo] {name} -> {out}")
    print("[done] qualitative demo complete")


if __name__ == "__main__":
    main()
