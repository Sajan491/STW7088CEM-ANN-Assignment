"""Interactive litter-detection dashboard (demo product for the video).

Two pages, selectable from the sidebar:

  • Live demo — upload or pick an aerial image and watch the project's two neural
    networks run side by side, in real time, on CPU:
        original / ground truth  |  YOLO detections  |  tile-classifier heatmap
  • Results & visualisations — a control page collecting every headline figure
    and table from the report (EDA, tile CNN, YOLO ablation, qualitative demos)
    so the whole story can be presented from the dashboard.

Nothing is retrained here — the app loads the committed checkpoints and reuses
the exact inference code from ``src`` (``qualitative_demo``, ``heatmap``), and
reads the generated figures/tables straight from ``results/``, so what the
viewer sees matches the report.

Run from the repository root:

    pip install -r requirements.txt          # includes streamlit
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- make the project importable no matter where streamlit is launched from ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.coco_parser import CocoDataset  # noqa: E402
from src.evaluation.heatmap import tile_litter_heatmap  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

# Palette borrowed from the report styling so the app and figures agree.
GREEN, BLUE, ORANGE, INK_MUTED = "#0f8a4d", "#2f6fd0", "#e8833a", "#8a8a8a"

YOLO_ARMS = {
    "Optimised (1024 px)": "configs/yolo_optimized.yaml",
    "Baseline (640 px)": "configs/yolo_baseline.yaml",
}


# --------------------------------------------------------------------------- #
# Cached loaders — models and dataset load once and stay warm across reruns.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the tile-classifier CNN…")
def load_tile_model(tile_config: str):
    import torch

    from src.models.factory import build_model

    cfg = load_config(tile_config)
    ckpt_path = (resolve_path(cfg["training"]["checkpoint_dir"])
                 / cfg["training"]["checkpoint_name"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@st.cache_resource(show_spinner="Loading the YOLO detector…")
def load_yolo_model(yolo_config: str):
    from ultralytics import YOLO

    cfg = load_config(yolo_config)
    ckpt = (resolve_path(cfg["eval"]["checkpoint_dir"])
            / cfg["eval"]["checkpoint_name"])
    return YOLO(str(ckpt)), cfg


@st.cache_resource(show_spinner="Reading dataset annotations…")
def load_dataset():
    cfg_data = load_config("configs/data.yaml")
    ann = resolve_path(cfg_data["dataset"]["annotations_file"])
    images_dir = resolve_path(cfg_data["dataset"]["images_dir"])
    splits = resolve_path(cfg_data["splits"]["file"])
    ds = CocoDataset(ann) if ann.exists() else None
    test_names: list[str] = []
    if splits.exists():
        with open(splits, "r", encoding="utf-8") as f:
            test_names = json.load(f)["splits"]["test"]
    tile_px = cfg_data["tiles"]["size"]
    return ds, images_dir, test_names, tile_px


@st.cache_data(show_spinner=False)
def load_headline_metrics():
    path = resolve_path("results/tables/final_summary.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Image + inference helpers (raw-pixel space, matching training).
# --------------------------------------------------------------------------- #
def read_raw_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB array, ignoring EXIF (matches training)."""
    import cv2

    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_raw_path(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_yolo(model, cfg_yolo, image_rgb, conf):
    t0 = time.perf_counter()
    res = model.predict(image_rgb[:, :, ::-1], imgsz=cfg_yolo["train"]["imgsz"],
                        conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    boxes, scores = [], []
    for b, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
        boxes.append((b[0], b[1], b[2] - b[0], b[3] - b[1]))
        scores.append(float(s))
    return boxes, scores, elapsed_ms


def run_heatmap(model, cfg_tile, image_rgb, tile_px, stride):
    import torch

    t0 = time.perf_counter()
    heat = tile_litter_heatmap(
        image_rgb, model,
        tile_size=tile_px,
        stride=stride,
        input_size=cfg_tile["data"]["input_size"],
        device=torch.device("cpu"),
    )
    return heat, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# Rendering — one matplotlib panel per stage (kept close to the report figure).
# --------------------------------------------------------------------------- #
def panel_boxes(image, boxes, color, scores=None):
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    aspect = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(6, 6 * aspect))
    ax.imshow(image)
    halo = [pe.Stroke(linewidth=3.0, foreground="white", alpha=0.7), pe.Normal()]
    for i, (x, y, w, h) in enumerate(boxes):
        r = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.8)
        r.set_path_effects(halo)
        ax.add_patch(r)
        if scores is not None:
            ax.text(x, max(y - 4, 2), f"{scores[i]:.2f}", color="white", fontsize=7,
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="none", alpha=0.85))
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def panel_heatmap(image, heat):
    import matplotlib.pyplot as plt

    aspect = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(figsize=(6, 6 * aspect))
    ax.imshow(image)
    hm = ax.imshow(heat, cmap="turbo", alpha=0.5, vmin=0, vmax=1)
    ax.axis("off")
    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("litter probability", fontsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig


# --------------------------------------------------------------------------- #
# Results-page helpers — show a generated figure/table, skip gracefully if absent.
# --------------------------------------------------------------------------- #
# Display-width presets for the results page (px; None = fill the page width).
FIG_SIZES = {"Medium": 640, "Large": 900, "Extra-large": 1200, "Fit width": None}


def show_figure(rel: str, caption: str | None = None, width: int | None = 900):
    """Show a generated figure at a fixed pixel width (crisp, consistent).

    ``width=None`` fills the available width. A fixed width avoids upscaling
    small PNGs and keeps square plots (e.g. confusion matrices) legible.
    """
    p = resolve_path(f"results/figures/{rel}")
    if not p.exists():
        st.info(f"Figure not generated yet: `{rel}` — run the evaluation notebooks/scripts.")
        return
    if width is None:
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.image(str(p), caption=caption, width=width)


def show_table(rel: str, caption: str | None = None, **kwargs):
    p = resolve_path(f"results/tables/{rel}")
    if p.exists():
        st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True, **kwargs)
        if caption:
            st.caption(caption)
    else:
        st.info(f"Table not generated yet: `{rel}`.")


# --------------------------------------------------------------------------- #
# PAGE 1 — Live demo
# --------------------------------------------------------------------------- #
def page_live_demo():
    st.title("🛰️ Aerial Litter Detection — live demo")
    st.caption(
        "Two neural networks on UAV imagery, running on CPU: a **custom CNN** flags "
        "litter-bearing regions, a fine-tuned **YOLO** detector localises individual "
        "items. UAVVaste dataset · ST7088CEM ANN coursework."
    )

    metrics = load_headline_metrics()
    if metrics is not None:
        def _val(task_frag, metric):
            row = metrics[(metrics["task"].str.contains(task_frag)) &
                          (metrics["metric"] == metric)]
            return float(row["value"].iloc[-1]) if len(row) else None

        c1, c2, c3, c4 = st.columns(4)
        cnn_f1 = _val("tile", "f1")
        cnn_auc = _val("tile", "roc_auc")
        base = metrics[(metrics["config"].str.contains("baseline")) & (metrics["metric"] == "mAP50")]
        opt = metrics[(metrics["config"].str.contains("optimised")) & (metrics["metric"] == "mAP50")]
        if cnn_f1 is not None:
            c1.metric("Tile CNN — F1", f"{cnn_f1:.3f}")
        if cnn_auc is not None:
            c2.metric("Tile CNN — ROC-AUC", f"{cnn_auc:.3f}")
        if len(base):
            c3.metric("YOLO baseline — mAP@50", f"{float(base['value'].iloc[-1]):.3f}")
        if len(opt) and len(base):
            delta = float(opt["value"].iloc[-1]) - float(base["value"].iloc[-1])
            c4.metric("YOLO optimised — mAP@50", f"{float(opt['value'].iloc[-1]):.3f}",
                      delta=f"+{delta:.3f} vs baseline")

    st.divider()

    ds, images_dir, test_names, tile_px = load_dataset()

    # ------------------------------- sidebar ------------------------------- #
    with st.sidebar:
        st.header("Controls")

        arm_label = st.radio("YOLO detector", list(YOLO_ARMS.keys()), index=0,
                             help="The optimised arm adds 1024 px input + aerial "
                                  "augmentation (+7 mAP@50 over baseline).")

        st.subheader("Image source")
        source = st.radio("Source", ["Test-set image", "Upload your own"],
                          label_visibility="collapsed")

        uploaded = None
        chosen_name = None

        if source == "Upload your own":
            uploaded = st.file_uploader("Aerial image", type=["jpg", "jpeg", "png"])
        else:
            if not test_names:
                st.warning("No splits.json found — upload an image instead.")
            else:
                if ds is not None:
                    by_name = {im["file_name"]: im for im in ds.images}
                    ordered = sorted(
                        [n for n in test_names if n in by_name],
                        key=lambda n: -len(ds.anns_for_image(by_name[n]["id"])),
                    )
                else:
                    ordered = sorted(test_names)
                if "img_idx" not in st.session_state:
                    st.session_state.img_idx = 0
                if st.button("🎲 Random test image", use_container_width=True):
                    st.session_state.img_idx = int(np.random.randint(len(ordered)))
                chosen_name = st.selectbox("Held-out test image", ordered,
                                           index=st.session_state.img_idx)

        st.subheader("Inference settings")
        conf = st.slider("YOLO confidence", 0.05, 0.60, 0.25, 0.05,
                         help="Lower = more (but noisier) detections. Watch the "
                              "precision/recall trade-off live.")
        stride = st.select_slider("Heatmap stride (px)", options=[128, 256, 512], value=256,
                                  help="Smaller stride = finer, slower heatmap.")
        show_gt = st.checkbox("Show ground-truth boxes", value=True,
                              help="Only available for held-out test images.")

    # ---------------------------- resolve the image ------------------------ #
    image_rgb = None
    gt_boxes: list = []
    title = ""

    if source == "Upload your own":
        if uploaded is not None:
            image_rgb = read_raw_bytes(uploaded.getvalue())
            title = uploaded.name
    elif chosen_name is not None:
        path = images_dir / chosen_name
        if path.exists():
            image_rgb = read_raw_path(path)
            title = chosen_name
            if ds is not None:
                by_name = {im["file_name"]: im for im in ds.images}
                gt_boxes = ds.bboxes_for_image(by_name[chosen_name]["id"])
        else:
            st.error(f"Image not found on disk: {path}. Place the UAVVaste images "
                     f"under data/images/ (see README).")

    if image_rgb is None:
        st.info("⬅️ Pick a held-out test image or upload an aerial photo to run both "
                "models.")
        st.stop()

    # ------------------------------- run models ---------------------------- #
    yolo_model, cfg_yolo = load_yolo_model(YOLO_ARMS[arm_label])
    tile_model, cfg_tile = load_tile_model("configs/tile_classifier.yaml")

    with st.spinner("Running YOLO detector…"):
        det_boxes, det_scores, yolo_ms = run_yolo(yolo_model, cfg_yolo, image_rgb, conf)
    with st.spinner("Sliding the tile classifier…"):
        heat, heat_ms = run_heatmap(tile_model, cfg_tile, image_rgb, tile_px, stride)

    # ------------------------------- three panels -------------------------- #
    h, w = image_rgb.shape[:2]
    st.markdown(f"**{title}** — {w}×{h}px")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### :green[Original / ground truth]")
        if show_gt and gt_boxes:
            st.pyplot(panel_boxes(image_rgb, gt_boxes, GREEN), use_container_width=True)
            st.caption(f"{len(gt_boxes)} labelled litter items")
        else:
            st.image(image_rgb, use_container_width=True)
            st.caption("no ground truth (uploaded image)" if not gt_boxes
                       else "ground-truth boxes hidden")

    with col2:
        st.markdown("##### :blue[YOLO detections]")
        st.pyplot(panel_boxes(image_rgb, det_boxes, BLUE, det_scores),
                  use_container_width=True)
        st.caption(f"{len(det_boxes)} found @ conf {conf:.2f} · {yolo_ms:.0f} ms · {arm_label}")

    with col3:
        st.markdown("##### :orange[Tile-classifier heatmap]")
        st.pyplot(panel_heatmap(image_rgb, heat), use_container_width=True)
        peak = float(heat.max()) if heat.size else 0.0
        st.caption(f"region-level litter probability · peak {peak:.2f} · {heat_ms:.0f} ms")

    # ------------------------- summary + accuracy check -------------------- #
    st.divider()
    s1, s2, s3 = st.columns(3)
    s1.metric("Detections", len(det_boxes))
    s2.metric("Detector latency", f"{yolo_ms:.0f} ms")
    s3.metric("Heatmap latency", f"{heat_ms:.0f} ms")

    if gt_boxes:
        st.caption(
            f"On this held-out image the detector proposed **{len(det_boxes)}** boxes "
            f"against **{len(gt_boxes)}** ground-truth items — a like-for-like look at "
            f"the mAP numbers above, on an image the model never saw in training."
        )

    with st.expander("How the two-stage pipeline works"):
        st.markdown(
            """
            - **Tile classifier (Task 1)** — a CNN trained *from scratch* on 512 px
              tiles labelled litter / no-litter. Slid across the whole image with
              overlapping windows, its per-tile probabilities average into the
              region-level **heatmap** on the right. It answers *“where should we look?”*
            - **YOLO detector (Task 2)** — a pretrained detector fine-tuned on UAVVaste
              (single `rubbish` class). The **optimised** arm adds 1024 px input and
              domain-motivated aerial augmentation, lifting mAP@50 from 0.79 to 0.86.
              It answers *“exactly which pixels are litter?”*
            - **Together** they form the deployment pattern from the proposal: the
              classifier flags litter-bearing regions cheaply, the detector localises
              individual items precisely. Everything here runs on **CPU**.
            """
        )


# --------------------------------------------------------------------------- #
# PAGE 2 — Results & visualisations (control page)
# --------------------------------------------------------------------------- #
def page_results():
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.title("📊 Results & visualisations")
        st.caption(
            "Every headline figure and table from the report, read straight from "
            "`results/` — the full story to present from the dashboard."
        )
    with head_r:
        size_label = st.radio("Figure size", list(FIG_SIZES.keys()), index=1,
                              help="Display width for the figures below. Use the ⤢ icon "
                                   "on any figure for full-screen.")
    W = FIG_SIZES[size_label]           # chosen display width (px, or None = fit)

    tab_data, tab_cnn, tab_yolo, tab_cross = st.tabs(
        ["🗺️ Dataset & EDA", "🧠 Task 1 · Tile CNN",
         "🎯 Task 2 · YOLO detection", "🔀 Cross-task & qualitative"]
    )

    # --- Dataset & EDA ----------------------------------------------------- #
    with tab_data:
        st.subheader("UAVVaste dataset")
        st.markdown(
            "772 drone images · 3,716 hand-labelled litter items · single `rubbish` "
            "class. The core challenge: litter is **tiny** relative to the frame."
        )
        show_table("dataset_summary.csv")
        show_figure("annotations_per_image.png",
                    "Litter items per image — most frames are sparsely littered.", width=W)
        show_figure("object_size_distribution.png",
                    "Object size distribution — the small-object problem, quantified.", width=W)
        st.subheader("Tile labelling (Task 1 inputs)")
        show_table("tile_class_balance.csv")
        show_figure("tile_class_balance.png",
                    "Litter / no-litter tile balance after gridding.", width=W)

    # --- Task 1 · Tile CNN ------------------------------------------------- #
    with tab_cnn:
        st.subheader("Custom CNN — trained from scratch on 512 px tiles")
        show_table("tile_classifier_metrics.csv",
                   "Held-out test metrics for the tile classifier.")
        show_figure("tile_learning_curves.png",
                    "Training / validation curves (early-stopped on val F1).", width=W)
        show_figure("tile_confusion_matrix.png",
                    "Confusion matrix on the held-out test tiles.", width=W)

    # --- Task 2 · YOLO ----------------------------------------------------- #
    with tab_yolo:
        st.subheader("Ablation — what each modification contributed")
        show_figure("yolo_ablation.png",
                    "Baseline → +resolution → +augmentation → +SAHI, each arm's contribution.",
                    width=W)
        show_table("yolo_ablation.csv",
                   "Note SAHI's sliced inference *reduced* precision here — an honest "
                   "negative result kept in the analysis.")
        st.divider()
        st.subheader("Per-configuration detail")
        arm = st.selectbox(
            "Configuration",
            ["optimized", "baseline", "res1024"],
            format_func={"optimized": "Optimised (1024 px, +aug)",
                         "baseline": "Baseline (640 px)",
                         "res1024": "+Resolution (1024 px)"}.get,
        )
        show_table(f"yolo_{arm}_metrics.csv", "Test-split metrics.")
        show_figure(f"yolo_{arm}_results.png", "Training results.", width=W)
        show_figure(f"yolo_{arm}_pr_curve.png", "Precision–recall curve.", width=W)
        show_figure(f"yolo_{arm}_f1_curve.png", "F1–confidence curve.", width=W)
        show_figure(f"yolo_{arm}_confusion_matrix.png", "Confusion matrix.", width=W)
        with st.expander("SAHI sliced-inference tables"):
            cc1, cc2 = st.columns(2)
            with cc1:
                show_table("yolo_sahi_metrics.csv", "With slicing.")
            with cc2:
                show_table("yolo_sahi_metrics_noslice.csv", "Control (no slicing).")

    # --- Cross-task & qualitative ------------------------------------------ #
    with tab_cross:
        st.subheader("Both tasks, side by side")
        show_table("final_summary.csv", "Consolidated headline results across both tasks.")
        show_figure("final_summary.png", width=W)
        st.divider()
        st.subheader("Qualitative demos on unseen images")
        st.caption("Three panels each: ground truth · YOLO detections · tile-classifier "
                   "heatmap — the same view the Live demo page produces on demand. These "
                   "are wide panels, so they fill the page width for clarity.")
        quals = sorted(resolve_path("results/figures/qualitative").glob("demo_*.png")) \
            if resolve_path("results/figures/qualitative").exists() else []
        if quals:
            for fig_path in quals:
                st.image(str(fig_path), caption=fig_path.stem.replace("demo_", ""),
                         use_container_width=True)
        else:
            st.info("No qualitative demo figures yet — run "
                    "`python -m src.evaluation.qualitative_demo`.")


# --------------------------------------------------------------------------- #
# App entry — page config, global styling, navigation.
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Aerial Litter Detection — demo",
                   page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 1rem;}
      [data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stImage"] img {border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

nav = st.navigation([
    st.Page(page_live_demo, title="Live demo", icon="🛰️", default=True),
    st.Page(page_results, title="Results & visualisations", icon="📊"),
])
nav.run()
