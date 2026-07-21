"""PyTorch dataset for the labelled litter tiles.

Reads the crops written by ``src.data.tiles`` from
``data/processed/tiles/<split>/{pos,neg}/*.jpg``. Labels come from the
directory name (pos=1, neg=0). Train-time augmentation is limited to
orientation flips and light colour jitter — aerial tiles have no canonical
orientation, so flips are label-preserving.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


class TileDataset(Dataset):
    """Litter/no-litter tile crops for one split."""

    def __init__(
        self,
        tiles_dir: str | Path,
        split: str,
        input_size: int,
        augment: dict | None = None,
        limit: int | None = None,
    ):
        self.split = split
        root = Path(tiles_dir) / split
        if not root.exists():
            raise FileNotFoundError(
                f"tile split directory {root} not found — run `python -m src.data.tiles` first"
            )
        pos = sorted((root / "pos").glob("*.jpg"))
        neg = sorted((root / "neg").glob("*.jpg"))
        self.samples: list[tuple[Path, int]] = [(p, 1) for p in pos] + [(p, 0) for p in neg]
        # deterministic interleave-by-name so --limit keeps both classes
        self.samples.sort(key=lambda s: s[0].name)
        if limit:
            self.samples = self.samples[:limit]

        ops: list = [transforms.Resize((input_size, input_size))]
        if augment:
            if augment.get("hflip"):
                ops.append(transforms.RandomHorizontalFlip())
            if augment.get("vflip"):
                ops.append(transforms.RandomVerticalFlip())
            jitter = augment.get("color_jitter", 0)
            if jitter:
                ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(float(label))

    # --- class-imbalance helpers ------------------------------------------
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def pos_weight(self) -> torch.Tensor:
        """BCE pos_weight = negatives / positives on this split."""
        labels = self.labels()
        pos = sum(labels)
        neg = len(labels) - pos
        return torch.tensor(neg / max(pos, 1), dtype=torch.float32)

    def balanced_sampler(self) -> WeightedRandomSampler:
        """Sampler drawing both classes with equal probability."""
        labels = self.labels()
        pos = sum(labels)
        neg = len(labels) - pos
        class_w = {1: 1.0 / max(pos, 1), 0: 1.0 / max(neg, 1)}
        weights = [class_w[l] for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
