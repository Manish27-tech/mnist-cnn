"""Model tests."""

from __future__ import annotations

import torch

from mnistvision.model import build_model, count_parameters


def test_forward_shape():
    model = build_model("cnn")
    logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_parameter_count():
    model = build_model("cnn")
    assert 50_000 < count_parameters(model) < 500_000


def test_gradcam_hook_exists():
    model = build_model("cnn")
    assert model.gradcam_layer is not None
    assert model.head().out_features == 10
