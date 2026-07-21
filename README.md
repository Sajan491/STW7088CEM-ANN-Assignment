# CNN-Based Aerial Litter Detection for Sustainable Trail and Environmental Cleanup

ST7088CEM Artificial Neural Networks — Individual Coursework
Softwarica College of IT & E-Commerce / Coventry University

## Overview

Automated litter detection in aerial (UAV) imagery, motivated by drone-supported
cleanup programmes on Nepal's trekking trails. Litter items in drone footage are
extremely small relative to the frame (typically under 1% of image area), which makes
this a challenging small-object problem. The project tackles it with two complementary
neural-network tasks on the [UAVVaste dataset](https://github.com/PUTvision/UAVVaste):

- **Task 1 — Tile-based litter classification.** Aerial images are sliced into grid
  tiles, each labelled litter / no-litter from annotation overlap. A custom CNN
  (stacked convolution–pooling blocks with a dense head) is designed and trained
  from scratch. Metrics: accuracy, precision, recall, F1, confusion matrix,
  learning curves.
- **Task 2 — Litter object detection.** A pretrained Ultralytics YOLO detector is
  fine-tuned on UAVVaste (single class `rubbish`). A baseline configuration at
  standard settings is established first; an optimised configuration then adds
  SAHI sliced inference, higher input resolution, augmentation, and hyperparameter
  tuning, with each modification's contribution quantified against the baseline.
  Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall.

## Dataset

[UAVVaste](https://github.com/PUTvision/UAVVaste) — 772 drone images, 3,716
hand-labelled litter annotations (COCO format, single `rubbish` class).

Download manually and place under `data/` (the folder is gitignored — the
dataset is **never** committed):

- Annotations: [annotations.json](https://raw.githubusercontent.com/PUTvision/UAVVaste/main/annotations/annotations.json)
  → save to `data/annotations/annotations.json`
- Images: [UAVVasteDataset.zip from Zenodo](https://zenodo.org/records/8214061)
  (~2.9 GB) → extract all images (flat) into `data/images/`

```
data/
├── annotations/
│   └── annotations.json
└── images/
    ├── BATCH_d07_img_580.jpg
    ├── batch_01_frame_0.jpg
    └── ... (772 .jpg files)
```

## Repository structure

```
├── configs/            # YAML configs (tile size, splits, model & training settings)
├── docs/               # Project plan and documentation
├── notebooks/          # One notebook per task (evidence runs)
├── results/
│   ├── figures/        # Generated figures (learning curves, EDA plots, ...)
│   └── tables/         # Generated metrics tables
├── src/
│   ├── data/           # Download, COCO parsing, splits, tile generation
│   ├── models/         # Custom CNN and model factory
│   ├── training/       # Training loops, schedules, early stopping
│   ├── evaluation/     # Metrics, ablations, qualitative demos
│   └── utils/          # Seeding, config loading, common helpers
└── tests/              # Unit tests (tile labelling, split leakage)
```

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

A CUDA GPU (e.g. Google Colab / Kaggle free tier) is assumed for training;
CPU works for the smoke-test modes.

## Usage

### Phase 1 — Data pipeline

All scripts read `configs/data.yaml` (tile size, split ratios, thresholds, seed)
and support a smoke-test mode for quick verification.

First place the dataset under `data/` as described in the Dataset section
(the EDA and split steps only need `annotations.json`; tile-crop writing and
detector training also need the images).

```bash
# 1. Leakage-free image-level splits (70/15/15, seeded)
python -m src.data.splits

# 2. Exploratory data analysis -> results/figures/ + results/tables/
python -m src.data.eda

# 3. Tile generation with overlap-based litter/no-litter labels
python -m src.data.tiles --dry-run               # labels + stats only (no images needed)
python -m src.data.tiles                         # write tile crops (requires images)
python -m src.data.tiles --limit 10              # smoke test on 10 images

# Unit tests (tile labelling correctness, split leakage)
python -m pytest tests/ -v
```

Notebook version: [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) runs the whole
phase top-to-bottom and displays the generated figures and tables.

### Phase 2 — Tile classifier (Task 1)

A custom CNN (stacked conv–pool blocks + dense head, defined in
`src/models/tile_cnn.py`) is trained from scratch on the labelled tiles.
All hyperparameters live in `configs/tile_classifier.yaml`. Class imbalance is
handled with a weighted BCE loss (or a balanced sampler — config switch);
training uses ReduceLROnPlateau and early stopping on validation F1.

```bash
# Train (GPU recommended; checkpoints/ and history CSV are written)
python -m src.training.train_tile_classifier
python -m src.training.train_tile_classifier --limit 400 --epochs 2   # smoke test

# Evaluate the best checkpoint on the held-out test split
# -> metrics table, confusion matrix + learning-curve figures
python -m src.evaluation.evaluate_tile_classifier
```

Notebook version: [notebooks/02_tile_classifier.ipynb](notebooks/02_tile_classifier.ipynb)
(tile generation, sample-tile grid, training, evaluation, results).

### Phase 3 — YOLO detection baseline (Task 2)

A pretrained Ultralytics YOLO detector is fine-tuned on UAVVaste (single class
`rubbish`) at standard 640 px settings, establishing the baseline the optimised
configuration (Phase 4) is measured against. Config: `configs/yolo_baseline.yaml`.
The COCO→YOLO converter reuses the Phase 1 image-level splits and writes the 29
EXIF-rotated images in raw-pixel space so labels and pixels stay aligned
(Ultralytics auto-applies EXIF orientation otherwise).

```bash
# 1. Convert COCO -> YOLO format (images/ + labels/ + data.yaml)
python -m src.data.coco_to_yolo
python -m src.data.coco_to_yolo --limit 20     # smoke test

# 2. Fine-tune the baseline detector (GPU; needs `pip install ultralytics`)
python -m src.training.train_yolo

# 3. Evaluate on the held-out test split -> results/tables/yolo_baseline_metrics.csv
python -m src.evaluation.evaluate_yolo
```

Notebook version: [notebooks/03_yolo_baseline.ipynb](notebooks/03_yolo_baseline.ipynb).

### Phase 4 — YOLO optimisation & ablation (Task 2)

Three modifications are added cumulatively to the baseline and each one's
contribution is quantified: **+resolution** (1024 px), **+augmentation**
(domain-motivated: vertical flips + rotation for nadir aerial, scale jitter,
copy-paste, brightness), and **+SAHI** (sliced inference). Configs:
`configs/yolo_res1024.yaml`, `configs/yolo_optimized.yaml`.

```bash
# Train the two 1024 px arms (GPU; needs ultralytics, sahi, pycocotools)
python -m src.training.train_yolo --config configs/yolo_res1024.yaml
python -m src.training.train_yolo --config configs/yolo_optimized.yaml

# Evaluate each arm on the test split
python -m src.evaluation.evaluate_yolo --config configs/yolo_res1024.yaml --tag "+resolution"
python -m src.evaluation.evaluate_yolo --config configs/yolo_optimized.yaml --tag "+augmentation"
python -m src.evaluation.sahi_eval    --config configs/yolo_optimized.yaml            # sliced
python -m src.evaluation.sahi_eval    --config configs/yolo_optimized.yaml --no-slice # control

# Consolidate baseline + arms -> results/tables/yolo_ablation.csv + figure
python -m src.evaluation.ablation
```

Notebook version: [notebooks/04_yolo_optimized.ipynb](notebooks/04_yolo_optimized.ipynb).

## Reproducibility

All scripts are config-driven (see `configs/`), set global seeds, and support a
`--limit N` smoke-test mode for quick verification before full runs. Dataset splits
are performed at image level so tiles from one image never cross splits.
