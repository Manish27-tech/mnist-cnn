"""Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402
from dashboard.views import about, demo, performance  # noqa: E402

st.set_page_config(page_title="MNIST CNN", page_icon="🔢", layout="wide")

kpis = data.kpis()
if kpis is None:
    st.title("MNIST CNN")
    st.error("Run `python run.py all` first to build the dashboard feed.")
    st.stop()

feed = data.load_feed()

with st.sidebar:
    st.title("MNIST CNN")
    page = st.radio("Page", ["Try it", "Performance", "About"], label_visibility="collapsed")
    st.metric("Accuracy", f"{kpis['accuracy']:.2%}")
    st.metric("Macro F1", f"{kpis['macro_f1']:.3f}")
    st.caption("A compact CNN with calibration, Grad-CAM and robustness checks.")

st.title("MNIST CNN")
{"Try it": demo.render, "Performance": performance.render, "About": about.render}[page](feed, kpis)
