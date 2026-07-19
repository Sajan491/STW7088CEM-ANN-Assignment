"""Lightweight parser for the UAVVaste COCO annotation file.

Wraps the raw JSON into convenient lookups (images, annotations per image,
bounding boxes) without requiring the images themselves — every field used
here (including image width/height) lives in the annotation file.
"""

import json
from collections import defaultdict
from pathlib import Path


class CocoDataset:
    """In-memory view of a COCO annotation file (single 'rubbish' class)."""

    def __init__(self, annotations_file: str | Path):
        with open(annotations_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.images: list[dict] = raw["images"]
        self.annotations: list[dict] = raw["annotations"]
        self.categories: list[dict] = raw.get("categories", [])

        self._img_by_id: dict[int, dict] = {im["id"]: im for im in self.images}
        self._anns_by_img: dict[int, list[dict]] = defaultdict(list)
        for ann in self.annotations:
            self._anns_by_img[ann["image_id"]].append(ann)

    # --- lookups -----------------------------------------------------------
    def image(self, image_id: int) -> dict:
        return self._img_by_id[image_id]

    def anns_for_image(self, image_id: int) -> list[dict]:
        return self._anns_by_img.get(image_id, [])

    def bboxes_for_image(self, image_id: int) -> list[list[float]]:
        """Bounding boxes as [x, y, w, h] in pixels for one image."""
        return [ann["bbox"] for ann in self.anns_for_image(image_id)]

    def image_ids(self) -> list[int]:
        return [im["id"] for im in self.images]

    def file_names(self) -> list[str]:
        return [im["file_name"] for im in self.images]

    # --- statistics --------------------------------------------------------
    def ann_counts_per_image(self) -> list[int]:
        """Number of annotations for every image (0 included)."""
        return [len(self._anns_by_img.get(i, [])) for i in self._img_by_id]

    def bbox_area_fractions(self) -> list[float]:
        """Each annotation's bbox area as a fraction of its image area."""
        fracs = []
        for ann in self.annotations:
            im = self._img_by_id[ann["image_id"]]
            img_area = im["width"] * im["height"]
            _, _, w, h = ann["bbox"]
            fracs.append((w * h) / img_area)
        return fracs

    def summary(self) -> dict:
        """Headline dataset statistics used in the report."""
        import numpy as np

        counts = np.array(self.ann_counts_per_image())
        fracs = np.array(self.bbox_area_fractions())
        sizes = {(im["width"], im["height"]) for im in self.images}
        return {
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": [c["name"] for c in self.categories],
            "anns_per_image_mean": float(counts.mean()),
            "anns_per_image_median": float(np.median(counts)),
            "anns_per_image_max": int(counts.max()),
            "images_without_annotations": int((counts == 0).sum()),
            "bbox_area_frac_mean": float(fracs.mean()),
            "bbox_area_frac_median": float(np.median(fracs)),
            "bbox_under_1pct_of_image": float((fracs < 0.01).mean()),
            "unique_resolutions": len(sizes),
        }


def main() -> None:
    """Quick check: print dataset summary from the annotation file."""
    import argparse

    from src.utils.config import load_config, resolve_path

    ap = argparse.ArgumentParser(description="Print UAVVaste annotation summary")
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = CocoDataset(resolve_path(cfg["dataset"]["annotations_file"]))
    print("=== UAVVaste annotation summary ===")
    for k, v in ds.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
