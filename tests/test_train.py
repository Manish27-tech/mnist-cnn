"""Training smoke tests."""

from __future__ import annotations

from mnistvision.data import build_loaders
from mnistvision.model import build_model
from mnistvision.train import fit


def test_fit_one_epoch(tiny_dataset, tmp_path):
    train_loader, val_loader, _ = build_loaders(tiny_dataset, batch_size=32)
    model = build_model("cnn")
    result = fit(model, train_loader, val_loader, epochs=1, checkpoint_path=tmp_path / "model.pt", verbose=False)
    assert result.epochs_run == 1
    assert len(result.history) == 1
    assert (tmp_path / "model.pt").exists()
