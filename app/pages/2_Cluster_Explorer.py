"""Page 2 — Cluster Explorer: interactive 2D scatter + side panel + radar."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app.components import inject_styles, render_disclaimer, risk_badge

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/processed/patients_clustered.csv")


@st.cache_resource
def load_profiles():
    with open(ROOT / "models/cluster_profiles.json") as f:
        return json.load(f)


inject_styles()
st.title("🔍 Cluster Explorer")
render_disclaimer()

df = load_data()
profiles = load_profiles()

projection = st.selectbox("2D projection", ["PCA", "UMAP", "t-SNE"], index=1)
proj_cols = {"PCA": ("pca1_2d", "pca2_2d"),
             "UMAP": ("umap1", "umap2"),
             "t-SNE": ("tsne1", "tsne2")}[projection]
df_plot = df.copy()
df_plot["risk_tier"] = df_plot["risk_tier"].astype(str)

fig = px.scatter(
    df_plot, x=proj_cols[0], y=proj_cols[1],
    color="cluster_name", symbol="risk_tier",
    hover_data={"cluster_id": True, "risk_tier": True, "gmm_confidence": ":.3f"},
    labels={proj_cols[0]: f"{projection}-1", proj_cols[1]: f"{projection}-2"},
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
fig.update_layout(height=500, legend_title_text="Cluster")
st.plotly_chart(fig, width="stretch")

st.markdown("### Cluster Detail")
cluster_options = sorted(df["cluster_id"].unique())
selected_cid = st.selectbox(
    "Select a cluster to inspect",
    cluster_options,
    format_func=lambda c: f"{c}: {profiles[str(c)]['name']} ({profiles[str(c)]['risk_tier']})",
)

p = profiles[str(selected_cid)]
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(f"**Name:** {p['name']}", unsafe_allow_html=True)
    st.markdown(f"**Risk tier:** {risk_badge(p['risk_tier'])}", unsafe_allow_html=True)
    st.markdown(f"**Size:** {p['size']} patients")
    st.markdown(f"**Mean GMM confidence:** {p['gmm_proba_mean']:.3f}")

with col2:
    radar_features = ["egfr", "sc", "hemo", "bgr", "multimorbidity", "anemia_severity", "cv_risk"]
    means_mat = np.array([
        [profiles[str(c)]["feature_means"].get(f, 0.0) for f in radar_features]
        for c in cluster_options
    ])
    mn, mx = means_mat.min(0), means_mat.max(0)
    norm = np.where(mx > mn, (means_mat - mn) / (mx - mn + 1e-9), 0.5)
    idx = cluster_options.index(selected_cid)
    radar_vals = list(norm[idx]) + [norm[idx][0]]
    radar_axes = radar_features + [radar_features[0]]
    fig_radar = go.Figure(go.Scatterpolar(r=radar_vals, theta=radar_axes, fill="toself",
                                          name=p["name"], line_color="#0F766E"))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                            showlegend=False, height=350,
                            title=f"{p['name']} (vs cluster min/max)")
    st.plotly_chart(fig_radar, width="stretch")

st.markdown("### Feature Profile (vs population mean)")
feature_table = pd.DataFrame([
    {
        "Feature": k,
        "Cluster mean": f"{p['feature_means'][k]:.2f}",
        "Δ vs population": f"{p['feature_deltas'][k]:+.2f}",
    }
    for k in p["feature_means"]
])
st.dataframe(feature_table, hide_index=True, width="stretch")
