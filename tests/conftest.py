"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tiny_arrays():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (200, 28, 28), dtype=np.uint8)
    labels = np.repeat(np.arange(10), 20)
    return images, labels


@pytest.fixture
def tiny_dataset(tiny_arrays):
    from mnistvision.data import Dataset

    images, labels = tiny_arrays
    return Dataset(images[:160], labels[:160], images[160:], labels[160:])
