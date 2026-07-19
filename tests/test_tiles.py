"""Unit tests for the tile grid and overlap-based labelling logic."""

from src.data.tiles import intersection_area, label_tile, tile_grid

FRAC = 0.25   # min_overlap_frac used across tests
PX = 1024     # min_overlap_px used across tests


class TestTileGrid:
    def test_exact_grid(self):
        # 1024x1024 image, 512 tiles, stride 512 -> 2x2 grid
        origins = tile_grid(1024, 1024, size=512, stride=512)
        assert origins == [(0, 0), (512, 0), (0, 512), (512, 512)]

    def test_edge_anchored_remainder(self):
        # 1300 wide: origins 0, 512 then a final tile flush with the edge (788)
        origins = tile_grid(1300, 512, size=512, stride=512)
        xs = [x for x, _ in origins]
        assert xs == [0, 512, 788]

    def test_full_coverage(self):
        # every pixel of a 4000x3000 image is inside at least one tile
        size = 512
        origins = tile_grid(4000, 3000, size=size, stride=size)
        assert max(x for x, _ in origins) + size == 4000
        assert max(y for _, y in origins) + size == 3000

    def test_image_smaller_than_tile(self):
        assert tile_grid(300, 300, size=512, stride=512) == []


class TestIntersectionArea:
    def test_bbox_fully_inside(self):
        assert intersection_area((0, 0, 512, 512), [100, 100, 50, 40]) == 50 * 40

    def test_no_overlap(self):
        assert intersection_area((0, 0, 512, 512), [600, 600, 50, 50]) == 0

    def test_partial_overlap(self):
        # bbox sticks 20px into the tile horizontally, fully inside vertically
        assert intersection_area((0, 0, 512, 512), [492, 100, 60, 30]) == 20 * 30


class TestLabelTile:
    def test_small_object_inside_is_positive(self):
        # typical UAVVaste object: tiny bbox fully inside -> frac = 1.0
        assert label_tile((0, 0, 512, 512), [[200, 200, 40, 40]], FRAC, PX)

    def test_no_objects_is_negative(self):
        assert not label_tile((0, 0, 512, 512), [], FRAC, PX)

    def test_object_outside_is_negative(self):
        assert not label_tile((0, 0, 512, 512), [[900, 900, 40, 40]], FRAC, PX)

    def test_sliver_overlap_below_both_thresholds_is_negative(self):
        # 2px of a 40px-wide bbox inside: frac = 0.05, area = 2*40 = 80 px^2
        bbox = [510, 100, 40, 40]
        assert not label_tile((0, 0, 512, 512), [bbox], FRAC, PX)

    def test_fraction_threshold_boundary(self):
        # exactly 25% of the bbox inside -> positive (>= comparison)
        bbox = [502, 100, 40, 40]  # 10px inside of 40 -> frac 0.25
        assert label_tile((0, 0, 512, 512), [bbox], FRAC, PX)

    def test_large_crossing_object_absolute_rule(self):
        # big object: only ~11% inside the tile (below frac) but 3200 px^2 >= PX
        bbox = [480, 0, 300, 100]  # 32px inside horizontally, 100 tall
        inter = 32 * 100
        assert inter / (300 * 100) < FRAC and inter >= PX
        assert label_tile((0, 0, 512, 512), [bbox], FRAC, PX)

    def test_any_of_many_boxes_triggers_positive(self):
        boxes = [[900, 900, 40, 40], [100, 100, 30, 30]]
        assert label_tile((0, 0, 512, 512), boxes, FRAC, PX)

    def test_zero_area_bbox_ignored(self):
        assert not label_tile((0, 0, 512, 512), [[100, 100, 0, 0]], FRAC, PX)
