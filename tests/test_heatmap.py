"""Unit tests for the tile-heatmap accumulation logic."""

import numpy as np

from src.evaluation.heatmap import accumulate_heatmap


class TestAccumulateHeatmap:
    def test_single_tile_fills_region(self):
        hm = accumulate_heatmap([(0, 0)], [0.8], image_shape=(10, 10), tile_size=10)
        assert hm.shape == (10, 10)
        assert np.allclose(hm, 0.8)

    def test_uncovered_pixels_are_zero(self):
        # one 5x5 tile in a 10x10 image -> the rest stays 0
        hm = accumulate_heatmap([(0, 0)], [1.0], image_shape=(10, 10), tile_size=5)
        assert np.allclose(hm[:5, :5], 1.0)
        assert hm[9, 9] == 0.0

    def test_overlap_is_averaged(self):
        # two 6-wide tiles overlapping in columns 4-5 -> that strip is the mean
        hm = accumulate_heatmap([(0, 0), (4, 0)], [1.0, 0.0],
                                image_shape=(6, 10), tile_size=6)
        assert hm[0, 0] == 1.0          # only tile 1
        assert hm[0, 9] == 0.0          # only tile 2
        assert hm[0, 4] == 0.5          # both tiles -> mean(1.0, 0.0)

    def test_values_stay_in_unit_range(self):
        hm = accumulate_heatmap([(0, 0), (2, 0)], [0.3, 0.9],
                                image_shape=(4, 6), tile_size=4)
        assert hm.min() >= 0.0 and hm.max() <= 1.0
