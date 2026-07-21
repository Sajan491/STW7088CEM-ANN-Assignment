"""Unit tests for the ablation table assembly."""

import pandas as pd

from src.evaluation.ablation import add_deltas


class TestAddDeltas:
    def _df(self):
        return pd.DataFrame([
            {"arm": "baseline (640)", "imgsz": 640, "mAP50": 0.79, "mAP50_95": 0.44,
             "precision": 0.81, "recall": 0.75},
            {"arm": "+resolution (1024)", "imgsz": 1024, "mAP50": 0.83, "mAP50_95": 0.50,
             "precision": 0.83, "recall": 0.78},
        ])

    def test_baseline_delta_is_zero(self):
        df = add_deltas(self._df())
        assert df.iloc[0]["d_mAP50"] == 0.0
        assert df.iloc[0]["d_mAP50_95"] == 0.0

    def test_improvement_delta_positive(self):
        df = add_deltas(self._df())
        assert df.iloc[1]["d_mAP50"] == round(0.83 - 0.79, 4)
        assert df.iloc[1]["d_mAP50_95"] == round(0.50 - 0.44, 4)

    def test_empty_frame_is_safe(self):
        assert add_deltas(pd.DataFrame()).empty
