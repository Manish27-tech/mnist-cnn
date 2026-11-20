"""About page."""

from __future__ import annotations

import streamlit as st

from dashboard import data


def render(feed: dict, kpis: dict) -> None:
    st.subheader("About this model")
    st.markdown(
        f"""
**Dataset.** MNIST handwritten digits, {kpis.get('test_rows', 10000):,} test images, 10 classes.

**Architecture.** {kpis['architecture']} with {kpis['parameters']:,} parameters.

**Training.** 3 epochs, AdamW with cosine decay and label smoothing.

**Calibration.** Temperature scaling fitted on the validation split,
ECE {kpis['ece_before']:.4f} to {kpis['ece_after']:.4f}.
"""
    )
    fig = data.figure("02_sample_grid.png")
    if fig:
        st.image(str(fig), width="stretch")

    st.subheader("Reproduce")
    st.code("python run.py all\npython run.py check\npython run.py dashboard", language="bash")
