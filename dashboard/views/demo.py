"""Interactive demo page."""

from __future__ import annotations

import streamlit as st

from dashboard import data


def render(feed: dict, kpis: dict) -> None:
    st.subheader("Inspect a test digit")
    samples = feed["samples"]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        idx = st.slider("Sample", 0, len(samples["images"]) - 1, 0)
        image = samples["images"][idx]
        true_label = int(samples["labels"][idx])
        st.image(image, caption=f"True label: {kpis['class_names'][true_label]}", width=160, clamp=True)

    row = feed["predictions"]
    row = row[row["image_index"] == int(samples["indices"][idx])]
    with col_b:
        if not row.empty:
            record = row.iloc[0]
            st.metric("Prediction", record["predicted_name"])
            st.metric("Confidence", f"{record['confidence']:.2%}")
            st.progress(float(record["confidence"]))
            if record["correct"]:
                st.success("Correct")
            else:
                st.error(f"Wrong, expected {record['true_name']}")

    st.divider()
    st.subheader("Grad-CAM")
    st.caption("Warm colours show the pixels that pushed the prediction.")
    fig = data.figure("08_gradcam.png")
    if fig:
        st.image(str(fig), width="stretch")
    else:
        st.info("Run the pipeline to render Grad-CAM panels.")
