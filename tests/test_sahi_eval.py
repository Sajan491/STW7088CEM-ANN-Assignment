"""Unit tests for the SAHI-evaluation helper functions (no GPU/SAHI needed)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.coco_parser import CocoDataset
from src.evaluation.sahi_eval import build_coco_gt, iou_xywh, precision_recall_at


class TestIoU:
    def test_identical_boxes(self):
        assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_disjoint_boxes(self):
        assert iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0

    def test_half_overlap(self):
        # two 10x10 boxes overlapping in a 5x10 strip -> inter 50, union 150
        assert iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


class TestPrecisionRecall:
    def test_perfect_detection(self):
        gt = {0: [[0, 0, 10, 10]]}
        dt = {0: [{"bbox": [0, 0, 10, 10], "score": 0.9}]}
        p, r = precision_recall_at(gt, dt, conf=0.25)
        assert (p, r) == (1.0, 1.0)

    def test_false_positive(self):
        gt = {0: [[0, 0, 10, 10]]}
        dt = {0: [{"bbox": [0, 0, 10, 10], "score": 0.9},
                  {"bbox": [500, 500, 10, 10], "score": 0.8}]}
        p, r = precision_recall_at(gt, dt, conf=0.25)
        assert r == 1.0 and p == pytest.approx(0.5)

    def test_missed_detection(self):
        gt = {0: [[0, 0, 10, 10], [50, 50, 10, 10]]}
        dt = {0: [{"bbox": [0, 0, 10, 10], "score": 0.9}]}
        p, r = precision_recall_at(gt, dt, conf=0.25)
        assert p == 1.0 and r == pytest.approx(0.5)

    def test_confidence_filter(self):
        gt = {0: [[0, 0, 10, 10]]}
        dt = {0: [{"bbox": [0, 0, 10, 10], "score": 0.1}]}  # below conf
        p, r = precision_recall_at(gt, dt, conf=0.25)
        assert (p, r) == (0.0, 0.0)

    def test_one_prediction_matches_one_gt_only(self):
        # a single prediction cannot count as two true positives
        gt = {0: [[0, 0, 10, 10], [0, 0, 10, 10]]}
        dt = {0: [{"bbox": [0, 0, 10, 10], "score": 0.9}]}
        p, r = precision_recall_at(gt, dt, conf=0.25)
        assert p == 1.0 and r == pytest.approx(0.5)


class TestBuildCocoGt:
    def _dataset(self):
        raw = {
            "images": [
                {"id": 7, "file_name": "b.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "a.jpg", "width": 200, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 7, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 3, "bbox": [0, 0, 50, 50]},
                {"id": 3, "image_id": 3, "bbox": [60, 10, 10, 10]},
            ],
            "categories": [{"id": 1, "name": "rubbish"}],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ann.json"
            p.write_text(json.dumps(raw))
            return CocoDataset(p)

    def test_dense_ids_and_sorted_names(self):
        ds = self._dataset()
        coco_gt, gt_by_img, records = build_coco_gt(ds, ["a.jpg", "b.jpg"], Path("imgs"))
        # image ids are dense 0..N-1 in sorted-name order
        assert [im["id"] for im in coco_gt["images"]] == [0, 1]
        assert [im["file_name"] for im in coco_gt["images"]] == ["a.jpg", "b.jpg"]

    def test_annotations_reindexed_to_dense_image_ids(self):
        ds = self._dataset()
        coco_gt, gt_by_img, _ = build_coco_gt(ds, ["a.jpg", "b.jpg"], Path("imgs"))
        # a.jpg (dense id 0) has 2 boxes, b.jpg (dense id 1) has 1 box
        assert len(gt_by_img[0]) == 2
        assert len(gt_by_img[1]) == 1
        assert all(a["category_id"] == 1 for a in coco_gt["annotations"])
        assert all(a["iscrowd"] == 0 for a in coco_gt["annotations"])
