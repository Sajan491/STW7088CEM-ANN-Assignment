"""Model factory: build classifiers by config name.

Keeping construction behind a registry means an alternative backbone
(e.g. a pretrained MobileNetV2) can be added later by registering one
builder function — training and evaluation code stay untouched.
"""

from torch import nn

from src.models.tile_cnn import TileCNN


def _build_tile_cnn(cfg: dict) -> nn.Module:
    return TileCNN(
        conv_channels=list(cfg.get("conv_channels", (32, 64, 128, 256, 256))),
        dense_units=cfg.get("dense_units", 128),
        dropout=cfg.get("dropout", 0.3),
    )


_REGISTRY = {
    "tile_cnn": _build_tile_cnn,
}


def build_model(cfg: dict) -> nn.Module:
    """Build a model from a config dict with a ``name`` key."""
    name = cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"unknown model '{name}'; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
