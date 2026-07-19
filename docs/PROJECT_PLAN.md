# Project Plan

CNN-Based Aerial Litter Detection for Sustainable Trail and Environmental Cleanup
(ST7088CEM Artificial Neural Networks coursework)

Two neural-network tasks on the UAVVaste aerial litter dataset
(772 images, 3,716 COCO annotations, single `rubbish` class).

## Phases

### Phase 1 — Data pipeline (`feature/data-pipeline`)
- Manual dataset placement under `data/` (direct links documented in README)
- COCO annotation parsing utilities
- Exploratory data analysis: object-size distribution, annotations per image,
  class-balance statistics → figures/tables saved to `results/`
- Image-level train/val/test splits (no tile ever crosses splits)
- Tile generation with overlap-based litter/no-litter labelling
- Unit tests: tile labelling correctness, split leakage

### Phase 2 — Tile classifier (`feature/tile-classifier`)
- Custom CNN built from scratch (stacked conv–pool blocks + dense head),
  model factory so alternative backbones can be added later
- Training loop: weighted loss / balanced sampling for class imbalance,
  LR schedule, early stopping, checkpointing
- Evaluation: accuracy, precision, recall, F1, confusion matrix, learning curves
- Figures saved to `results/figures/`

### Phase 3 — YOLO baseline (`feature/yolo-baseline`)
- COCO → YOLO label conversion (image-level splits reused)
- Baseline fine-tune of pretrained Ultralytics YOLO at standard settings (~640 px)
- Evaluation: mAP@0.5, mAP@0.5:0.95, precision, recall
- Baseline metrics table saved to `results/tables/`

### Phase 4 — YOLO optimisation (`feature/yolo-optimization`)
- Higher input resolution (1024+)
- Augmentation configuration (mosaic, scale, flips, brightness)
- SAHI sliced inference
- Ablation runner producing a table:
  baseline vs +resolution vs +augmentation vs +SAHI

### Phase 5 — Evaluation & demo (`feature/evaluation`)
- Cross-task comparison figures
- Qualitative demo: annotated detections + tile heatmap side by side
  on unseen images
- Final consolidated metrics


