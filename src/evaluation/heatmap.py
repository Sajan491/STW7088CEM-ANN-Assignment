"""Tile-classifier litter heatmap over a whole aerial image.

The custom CNN (Task 1) is a tile-level classifier; sliding it across an image
with overlapping windows and averaging the per-tile litter probabilities yields
a region-level litter heatmap — the "coarse litter map" from the proposal, and
the left half of the two-stage deployment story (classifier flags regions, the
detector localises within them).
"""

from pathlib import Path

import numpy as np

from src.data.tiles import tile_grid


def accumulate_heatmap(
    positions: list[tuple[int, int]],
    probs: list[float],
    image_shape: tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    """Average per-tile probabilities into a pixel heatmap.

    positions are (x, y) tile origins; probs the matching litter probabilities.
    Overlapping tiles are averaged. Returns an (H, W) float array in [0, 1].
    """
    h, w = image_shape
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    for (x, y), p in zip(positions, probs):
        acc[y : y + tile_size, x : x + tile_size] += p
        cnt[y : y + tile_size, x : x + tile_size] += 1.0
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def tile_litter_heatmap(
    image: np.ndarray,
    model,
    tile_size: int,
    stride: int,
    input_size: int,
    device,
) -> np.ndarray:
    """Slide the tile classifier over an image -> (H, W) litter-probability map.

    `image` is an RGB uint8 array. Tiles are taken on a `stride`-spaced grid,
    resized to the classifier input, and scored; probabilities are averaged
    back over overlaps.
    """
    import torch
    from torchvision import transforms

    h, w = image.shape[:2]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    positions = tile_grid(w, h, size=tile_size, stride=stride)
    if not positions:
        return np.zeros((h, w), dtype=np.float32)

    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        batch, batch_pos = [], []
        for (x, y) in positions:
            crop = image[y : y + tile_size, x : x + tile_size]
            batch.append(tf(crop))
            batch_pos.append((x, y))
            if len(batch) == 32:
                logits = model(torch.stack(batch).to(device))
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                batch = []
        if batch:
            logits = model(torch.stack(batch).to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    return accumulate_heatmap(positions, probs, (h, w), tile_size)
