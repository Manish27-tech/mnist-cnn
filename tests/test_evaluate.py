"""Evaluation tests."""

from __future__ import annotations

import numpy as np

from mnistvision.evaluate import (
    build_prediction_frame,
    expected_calibration_error,
    selective_accuracy,
    temperature_scale,
)


def test_temperature_scale_in_range():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(100, 10))
    targets = rng.integers(0, 10, 100)
    temperature = temperature_scale(logits, targets)
    assert 0.5 <= temperature <= 5.0


def test_ece_is_bounded():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(10), size=100)
    targets = rng.integers(0, 10, 100)
    ece, mce, bins = expected_calibration_error(probs, targets)
    assert 0 <= ece <= 1
    assert 0 <= mce <= 1
    assert len(bins) == 15


def test_prediction_frame_and_selective():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(10), size=50)
    targets = rng.integers(0, 10, 50)
    predictions = build_prediction_frame(probs, targets)
    assert len(predictions.frame) == 50
    assert predictions.frame.columns.tolist() == [
        "image_index", "true_index", "true_name", "predicted_index",
        "predicted_name", "confidence", "entropy", "correct",
    ]
    rows = selective_accuracy(predictions)
    assert {row["threshold"] for row in rows} == {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}
