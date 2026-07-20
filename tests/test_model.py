"""Unit tests for the custom CNN and the model factory."""

import pytest
import torch

from src.models.factory import build_model
from src.models.tile_cnn import TileCNN

CFG = {"name": "tile_cnn", "conv_channels": [16, 32, 64], "dense_units": 32, "dropout": 0.2}


class TestFactory:
    def test_builds_registered_model(self):
        model = build_model(CFG)
        assert isinstance(model, TileCNN)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            build_model({"name": "resnet_9000"})


class TestTileCNN:
    def test_forward_shape(self):
        model = build_model(CFG)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4,)

    def test_input_size_tolerant(self):
        # global average pooling makes the head input-size independent
        model = build_model(CFG)
        out = model(torch.randn(2, 3, 128, 128))
        assert out.shape == (2,)

    def test_has_trainable_parameters(self):
        model = build_model(CFG)
        assert model.num_parameters() > 0

    def test_gradients_flow(self):
        model = build_model(CFG)
        x = torch.randn(2, 3, 64, 64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), torch.tensor([1.0, 0.0])
        )
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_deterministic_under_seed(self):
        torch.manual_seed(7)
        a = build_model(CFG)
        torch.manual_seed(7)
        b = build_model(CFG)
        x = torch.randn(1, 3, 64, 64)
        a.eval(), b.eval()
        assert torch.allclose(a(x), b(x))
