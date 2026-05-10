"""Streamlit app entry point. Pages auto-discovered from app/pages/."""
from pathlib import Path
import streamlit as st
from app.components import inject_styles, render_disclaimer

st.set_page_config(
    page_title="CKD Patient Clustering",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

with st.sidebar:
    st.markdown("# 🩺 CKD Patient Clustering")
    st.markdown("**Clinical decision-support dashboard**")
    st.markdown("---")
    st.markdown(
        "Unsupervised clustering of UCI CKD data. Discovers patient subgroups "
        "and assigns Low/Medium/High risk tiers."
    )
    st.markdown("---")
    st.markdown(
        "[GitHub](https://github.com/Homd11/health-care)  ·  AIE323 Data Mining"
    )

st.title("CKD Patient Clustering & Medical Risk Grouping")
render_disclaimer()
st.markdown(
    "Use the sidebar to navigate the dashboard pages: **Population Overview**, "
    "**Cluster Explorer**, **Patient Risk Lookup**, **Feature Distributions**, "
    "**Batch Risk Scoring**."
)
st.info(
    "📊 The clustering model was fit on a 9-feature clinical core matrix "
    "(eGFR, creatinine, hemoglobin, glucose, multimorbidity, anemia severity, "
    "CV risk, age, blood pressure) and discovered **3 patient subgroups**."
)
