"""Unit tests for COCO -> YOLO conversion."""

import pytest

from src.data.coco_to_yolo import coco_bbox_to_yolo


class TestBboxConversion:
    def test_centre_of_image(self):
        # 100x100 box centred in a 200x200 image
        xc, yc, w, h = coco_bbox_to_yolo([50, 50, 100, 100], 200, 200)
        assert (xc, yc, w, h) == (0.5, 0.5, 0.5, 0.5)

    def test_top_left_corner(self):
        # 20x20 box in the top-left of a 100x100 image -> centre at (10,10)
        xc, yc, w, h = coco_bbox_to_yolo([0, 0, 20, 20], 100, 100)
        assert (xc, yc, w, h) == (0.1, 0.1, 0.2, 0.2)

    def test_non_square_image(self):
        xc, yc, w, h = coco_bbox_to_yolo([100, 50, 40, 30], 400, 200)
        assert xc == pytest.approx((100 + 20) / 400)
        assert yc == pytest.approx((50 + 15) / 200)
        assert w == pytest.approx(40 / 400)
        assert h == pytest.approx(30 / 200)

    def test_values_within_unit_range(self):
        # a small aerial-litter box stays well inside [0, 1]
        xc, yc, w, h = coco_bbox_to_yolo([1000, 800, 30, 25], 4000, 3000)
        assert all(0.0 <= v <= 1.0 for v in (xc, yc, w, h))

    def test_full_image_box(self):
        xc, yc, w, h = coco_bbox_to_yolo([0, 0, 640, 480], 640, 480)
        assert (xc, yc, w, h) == (0.5, 0.5, 1.0, 1.0)
