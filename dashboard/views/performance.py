"""Performance page."""

from __future__ import annotations

import streamlit as st

from dashboard import data


def render(feed: dict, kpis: dict) -> None:
    st.subheader("Headline metrics")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{kpis['accuracy']:.2%}")
    cols[1].metric("Macro F1", f"{kpis['macro_f1']:.3f}")
    cols[2].metric("Parameters", f"{kpis['parameters']:,}")
    cols[3].metric("ECE (calibrated)", f"{kpis['ece_after']:.4f}")

    st.subheader("Per-class scores")
    st.bar_chart(feed["per_class"].set_index("class_name")["f1"])
    st.dataframe(feed["per_class"], hide_index=True, width="stretch")

    st.subheader("Confusion matrix")
    fig = data.figure("04_confusion_matrix.png")
    if fig:
        st.image(str(fig), width="stretch")

    st.subheader("Confidence vs accuracy")
    fig = data.figure("06_reliability_diagram.png")
    if fig:
        st.image(str(fig), width="stretch")

    st.subheader("Robustness")
    st.dataframe(feed["robustness"], hide_index=True, width="stretch")
    fig = data.figure("07_robustness.png")
    if fig:
        st.image(str(fig), width="stretch")
