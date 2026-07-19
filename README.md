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
The dataset is downloaded by script and is **not** committed to this repository.

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

Each phase adds runnable scripts; usage instructions are added to this section
as the corresponding phase lands.

## Reproducibility

All scripts are config-driven (see `configs/`), set global seeds, and support a
`--limit N` smoke-test mode for quick verification before full runs. Dataset splits
are performed at image level so tiles from one image never cross splits.
