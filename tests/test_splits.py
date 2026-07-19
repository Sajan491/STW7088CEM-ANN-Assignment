"""Unit tests for image-level splitting and leakage prevention."""

import pytest

from src.data.splits import make_splits, verify_disjoint

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def fake_names(n: int) -> list[str]:
    return [f"img_{i:04d}.jpg" for i in range(n)]


class TestMakeSplits:
    def test_partition_is_complete_and_disjoint(self):
        names = fake_names(772)
        splits = make_splits(names, RATIOS, seed=42)
        verify_disjoint(splits)  # must not raise
        combined = splits["train"] + splits["val"] + splits["test"]
        assert sorted(combined) == sorted(names)

    def test_ratios_respected(self):
        splits = make_splits(fake_names(1000), RATIOS, seed=42)
        assert len(splits["train"]) == 700
        assert len(splits["val"]) == 150
        assert len(splits["test"]) == 150

    def test_deterministic_for_same_seed(self):
        a = make_splits(fake_names(500), RATIOS, seed=42)
        b = make_splits(fake_names(500), RATIOS, seed=42)
        assert a == b

    def test_different_seed_differs(self):
        a = make_splits(fake_names(500), RATIOS, seed=42)
        b = make_splits(fake_names(500), RATIOS, seed=43)
        assert a != b

    def test_input_order_irrelevant(self):
        names = fake_names(300)
        a = make_splits(names, RATIOS, seed=42)
        b = make_splits(list(reversed(names)), RATIOS, seed=42)
        assert a == b

    def test_bad_ratios_rejected(self):
        with pytest.raises(AssertionError):
            make_splits(fake_names(10), {"train": 0.8, "val": 0.1, "test": 0.2}, seed=42)


class TestVerifyDisjoint:
    def test_leakage_detected(self):
        splits = {
            "train": ["a.jpg", "b.jpg"],
            "val": ["b.jpg"],  # leaked
            "test": ["c.jpg"],
        }
        with pytest.raises(ValueError, match="leakage"):
            verify_disjoint(splits)


class TestTileSplitInheritance:
    """Tiles must inherit their source image's split — no tile-level leakage."""

    def test_tiles_never_cross_splits(self):
        from src.data.coco_parser import CocoDataset
        from src.data.tiles import build_tile_index

        # synthetic 2-image dataset written through the parser's expected schema
        import json
        import tempfile
        from pathlib import Path

        raw = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 1024, "height": 1024},
                {"id": 2, "file_name": "b.jpg", "width": 1024, "height": 1024},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "bbox": [10, 10, 40, 40]},
            ],
            "categories": [{"id": 1, "name": "rubbish"}],
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ann.json"
            p.write_text(json.dumps(raw), encoding="utf-8")
            ds = CocoDataset(p)

        splits = {"train": ["a.jpg"], "val": ["b.jpg"], "test": []}
        index = build_tile_index(
            ds, splits, size=512, stride=512, min_overlap_frac=0.25, min_overlap_px=1024
        )
        split_of_image = {"a.jpg": "train", "b.jpg": "val"}
        assert index, "expected tiles to be generated"
        for tile in index:
            assert tile["split"] == split_of_image[tile["image"]]
