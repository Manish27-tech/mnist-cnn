"""Cached access to the committed dashboard feed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "artifacts" / "dashboard"
FIGURES = ROOT / "reports" / "figures"


def available() -> bool:
    return (FEED / "kpis.json").exists()


@st.cache_data(show_spinner=False)
def kpis() -> dict | None:
    path = FEED / "kpis.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feed() -> dict:
    bundle = {}
    for name in ("history", "per_class", "robustness", "predictions", "confidence_buckets"):
        bundle[name] = pd.read_csv(FEED / f"{name}.csv")
    bundle["selective"] = pd.read_csv(FEED / "selective_accuracy.csv")
    with np.load(FEED / "samples.npz") as z:
        bundle["samples"] = {"images": z["images"], "labels": z["labels"], "indices": z["indices"]}
    with np.load(FEED / "gradcam.npz") as z:
        bundle["gradcam"] = {"images": z["images"], "heatmaps": z["heatmaps"]}
    return bundle


def figure(name: str) -> Path | None:
    path = FIGURES / name
    return path if path.exists() else None
