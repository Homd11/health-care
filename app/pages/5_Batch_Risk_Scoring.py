"""Page 5 — Upload CSV of patients, get cluster + risk + confidence."""
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from app.components import inject_styles, render_disclaimer
from src.inference import predict_patient

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_COLS = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

inject_styles()
st.title("📂 Batch Risk Scoring")
render_disclaimer()

st.markdown(
    "Upload a CSV of patient records. Each row is scored independently. "
    f"**Required columns:** `{', '.join(TEMPLATE_COLS)}`."
)

template_df = pd.DataFrame(columns=TEMPLATE_COLS)
template_buf = io.StringIO()
template_df.to_csv(template_buf, index=False)
st.download_button("⬇️ Download empty template",
                   template_buf.getvalue(), file_name="ckd_template.csv", mime="text/csv")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded is not None:
    df_in = pd.read_csv(uploaded)
    missing = set(TEMPLATE_COLS) - set(df_in.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
    else:
        with st.spinner(f"Scoring {len(df_in)} patients..."):
            rows = []
            for _, r in df_in.iterrows():
                try:
                    res = predict_patient(r[TEMPLATE_COLS].to_dict())
                    rows.append({
                        "cluster_id": res["cluster_id"],
                        "cluster_name": res["cluster_name"],
                        "risk_tier": res["risk_tier"],
                        "gmm_confidence": round(res["confidence"], 3),
                        "top_feature_1": res["top_features"][0][0] if res["top_features"] else "",
                        "top_feature_2": res["top_features"][1][0] if len(res["top_features"]) > 1 else "",
                        "top_feature_3": res["top_features"][2][0] if len(res["top_features"]) > 2 else "",
                    })
                except Exception as e:
                    rows.append({"cluster_id": None, "cluster_name": "ERROR", "risk_tier": "",
                                 "gmm_confidence": 0.0, "top_feature_1": str(e),
                                 "top_feature_2": "", "top_feature_3": ""})
        results = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Patients scored", len(results))
        with c2:
            st.metric("% High-Risk", f"{(results['risk_tier'] == 'High').mean() * 100:.1f}%")
        with c3:
            st.metric("Mean confidence", f"{results['gmm_confidence'].mean():.3f}")

        st.dataframe(results, hide_index=True, width="stretch")

        out_buf = io.StringIO()
        results.to_csv(out_buf, index=False)
        st.download_button("⬇️ Download scored results", out_buf.getvalue(),
                           file_name="scored_patients.csv", mime="text/csv")
