"""Data tests."""

from __future__ import annotations

from mnistvision.data import stratified_split


def test_stratified_split_is_disjoint(tiny_arrays):
    _, labels = tiny_arrays
    train_idx, val_idx = stratified_split(labels, 0.2, 0)
    assert len(train_idx) + len(val_idx) == len(labels)
    assert len(val_idx) == 40
    assert not set(train_idx) & set(val_idx)


def test_loaders_shapes(tiny_dataset):
    from mnistvision.data import build_loaders

    train_loader, val_loader, test_loader = build_loaders(tiny_dataset, batch_size=16)
    images, targets = next(iter(train_loader))
    assert images.shape == (16, 1, 28, 28)
    assert targets.shape == (16,)
    assert len(test_loader.dataset) == 40
