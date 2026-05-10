import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Page 1 — Population Overview."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from app.components import inject_styles, render_disclaimer, kpi_card

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/patients_clustered.csv"
PROFILES_PATH = ROOT / "models/cluster_profiles.json"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_profiles():
    with open(PROFILES_PATH) as f:
        return json.load(f)


inject_styles()
st.title("📊 Population Overview")
render_disclaimer()

df = load_data()
profiles = load_profiles()

n_total = len(df)
n_clusters = df["cluster_id"].nunique()
pct_high_risk = (df["risk_tier"] == "High").mean() * 100
pct_ckd = (df["classification"] == 1).mean() * 100

c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Total Patients", n_total)
with c2: kpi_card("Clusters", n_clusters)
with c3: kpi_card("% High-Risk", f"{pct_high_risk:.1f}%", color="#DC2626")
with c4: kpi_card("% CKD-positive", f"{pct_ckd:.1f}%", color="#0F766E")

st.markdown("### Cluster Sizes")
cluster_counts = (
    df.groupby(["cluster_id", "cluster_name", "risk_tier"]).size()
    .reset_index(name="patients")
)
fig_bar = px.bar(
    cluster_counts, x="cluster_name", y="patients", color="risk_tier",
    color_discrete_map={"Low": "#0F766E", "Medium": "#F59E0B", "High": "#DC2626"},
    text="patients", labels={"cluster_name": "Cluster", "patients": "Patient count"},
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(height=400, showlegend=True)
st.plotly_chart(fig_bar, width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Risk Tier Distribution")
    risk_counts = df["risk_tier"].value_counts().reset_index()
    risk_counts.columns = ["risk_tier", "patients"]
    fig_donut = px.pie(
        risk_counts, values="patients", names="risk_tier", hole=0.5,
        color="risk_tier",
        color_discrete_map={"Low": "#0F766E", "Medium": "#F59E0B", "High": "#DC2626"},
    )
    fig_donut.update_layout(height=400)
    st.plotly_chart(fig_donut, width="stretch")

with col2:
    st.markdown("### Cluster Profiles")
    profile_table = pd.DataFrame([
        {
            "Cluster": cid,
            "Name": p["name"],
            "Risk": p["risk_tier"],
            "Size": p["size"],
            "Mean eGFR": f"{p['feature_means'].get('egfr', 0):.1f}",
            "Mean Multimorbidity": f"{p['feature_means'].get('multimorbidity', 0):.2f}",
            "GMM confidence": f"{p['gmm_proba_mean']:.3f}",
        }
        for cid, p in profiles.items()
    ])
    st.dataframe(profile_table, hide_index=True, width="stretch")
