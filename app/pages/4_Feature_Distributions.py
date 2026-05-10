import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Page 4 — Per-feature violin/box plots per cluster + ANOVA significance."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from app.components import inject_styles, render_disclaimer

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/processed/patients_clustered.csv")


inject_styles()
st.title("📈 Feature Distributions per Cluster")
render_disclaimer()

df = load_data()
raw_cols = [c for c in df.columns if c.startswith("raw_")]
feature_options = sorted([c.replace("raw_", "") for c in raw_cols])
selected = st.selectbox("Feature", feature_options, index=feature_options.index("egfr") if "egfr" in feature_options else 0)
col_key = f"raw_{selected}"

groups = [df.loc[df["cluster_id"] == cid, col_key].dropna().values
          for cid in sorted(df["cluster_id"].unique())]
if all(len(g) > 1 for g in groups):
    f_stat, p_val = stats.f_oneway(*groups)
else:
    f_stat, p_val = float("nan"), float("nan")

c1, c2 = st.columns(2)
with c1:
    fig_v = px.violin(df, x="cluster_name", y=col_key, color="risk_tier", box=True, points="all",
                      color_discrete_map={"Low": "#0F766E", "Medium": "#F59E0B", "High": "#DC2626"},
                      labels={"cluster_name": "Cluster", col_key: selected})
    fig_v.update_layout(height=450, title=f"{selected} (violin) — ANOVA F={f_stat:.2f}, p={p_val:.2g}")
    st.plotly_chart(fig_v, width="stretch")
with c2:
    fig_b = px.box(df, x="cluster_name", y=col_key, color="risk_tier",
                   color_discrete_map={"Low": "#0F766E", "Medium": "#F59E0B", "High": "#DC2626"},
                   labels={"cluster_name": "Cluster", col_key: selected})
    fig_b.update_layout(height=450, title=f"{selected} (box)")
    st.plotly_chart(fig_b, width="stretch")

st.markdown("### Per-cluster summary")
summary = df.groupby(["cluster_id", "cluster_name", "risk_tier"])[col_key].agg(["mean", "median", "std", "min", "max"]).reset_index()
st.dataframe(summary, hide_index=True, width="stretch")
