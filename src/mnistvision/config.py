"""Project configuration."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
ARTIFACTS = ROOT / "artifacts"
DASHBOARD_FEED = ARTIFACTS / "dashboard"
CHECKPOINT = ARTIFACTS / "model.pt"

CLASS_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
NUM_CLASSES = 10
IMAGE_SIZE = 28
CHANNELS = 1

VAL_FRACTION = 0.1
SEED = 42
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
NUM_THREADS = 12
NUM_WORKERS = 0

PIXEL_MEAN = 0.1307
PIXEL_STD = 0.3081

ROBUSTNESS_SAMPLE = 1000
DATASET_SOURCE = "http://yann.lecun.com/exdb/mnist/"
DATASET_LICENSE = "CC BY-SA 3.0"


def ensure_directories() -> None:
    for path in (DATA, REPORTS, FIGURES, ARTIFACTS, DASHBOARD_FEED):
        path.mkdir(parents=True, exist_ok=True)
