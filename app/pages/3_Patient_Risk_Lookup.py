"""Page 3 — Patient Risk Lookup (existing index OR new patient form)."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.components import inject_styles, render_disclaimer, risk_badge, confidence_bar
from src.inference import predict_patient

ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data/processed/patients_clustered.csv")


@st.cache_resource
def load_profiles():
    with open(ROOT / "models/cluster_profiles.json") as f:
        return json.load(f)


inject_styles()
st.title("🩺 Patient Risk Lookup")
render_disclaimer()

df = load_data()
profiles = load_profiles()
mode = st.radio("Mode", ["Existing patient", "New patient form"], horizontal=True)

if mode == "Existing patient":
    idx = st.number_input("Patient index", min_value=0, max_value=len(df) - 1, value=0)
    row = df.iloc[int(idx)]
    cid = int(row["cluster_id"])
    p = profiles[str(cid)]
    confidence = float(row["gmm_confidence"])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"### Cluster: {p['name']}")
        st.markdown(f"**Risk tier:** {risk_badge(p['risk_tier'])}", unsafe_allow_html=True)
        confidence_bar(confidence)
    with col2:
        raw_cols = [c for c in row.index if c.startswith("raw_")]
        if raw_cols:
            patient_table = pd.DataFrame({
                "Feature": [c.replace("raw_", "") for c in raw_cols],
                "Value": [round(float(row[c]), 2) for c in raw_cols],
            })
            st.markdown("**Patient values:**")
            st.dataframe(patient_table, hide_index=True, width="stretch")

else:
    st.markdown("Fill all fields. Defaults are population means.")
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            bp = st.number_input("Blood pressure (mmHg)", min_value=40, max_value=250, value=120)
            sg = st.number_input("Specific gravity", min_value=1.000, max_value=1.030, value=1.020, step=0.005, format="%.3f")
            al = st.number_input("Albumin (0-5)", min_value=0, max_value=5, value=0)
            su = st.number_input("Sugar (0-5)", min_value=0, max_value=5, value=0)
            bgr = st.number_input("Blood glucose (mg/dL)", min_value=20, max_value=600, value=120)
            bu = st.number_input("Blood urea (mg/dL)", min_value=0, max_value=400, value=40)
            sc = st.number_input("Serum creatinine (mg/dL)", min_value=0.0, max_value=80.0, value=1.0, step=0.1, format="%.2f")
        with c2:
            sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=170, value=140)
            pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=12.0, value=4.5, step=0.1)
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=14.0, step=0.1)
            pcv = st.number_input("Packed cell volume (%)", min_value=10, max_value=60, value=42)
            wc = st.number_input("WBC count (cells/cumm)", min_value=2000, max_value=30000, value=8000)
            rc = st.number_input("RBC count (millions/cumm)", min_value=2.0, max_value=8.0, value=5.0, step=0.1)
            rbc = st.selectbox("RBC appearance", ["normal", "abnormal"])
            pc = st.selectbox("Pus cells", ["normal", "abnormal"])
        with c3:
            pcc = st.selectbox("Pus cell clumps", ["notpresent", "present"])
            ba = st.selectbox("Bacteria", ["notpresent", "present"])
            htn = st.selectbox("Hypertension", ["no", "yes"])
            dm = st.selectbox("Diabetes mellitus", ["no", "yes"])
            cad = st.selectbox("Coronary artery disease", ["no", "yes"])
            appet = st.selectbox("Appetite", ["good", "poor"])
            pe = st.selectbox("Pedal edema", ["no", "yes"])
            ane = st.selectbox("Anemia", ["no", "yes"])

        submitted = st.form_submit_button("Predict cluster + risk", type="primary")

    if submitted:
        patient = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
            "pcv": pcv, "wc": wc, "rc": rc,
            "htn": htn, "dm": dm, "cad": cad, "appet": appet,
            "pe": pe, "ane": ane,
        }
        result = predict_patient(patient)

        st.markdown("### Prediction")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Cluster:** {result['cluster_name']}")
            st.markdown(
                f"**Risk tier:** {risk_badge(result['risk_tier'])}",
                unsafe_allow_html=True,
            )
            confidence_bar(result["confidence"])
            st.info(result["summary"])
        with c2:
            st.markdown("**Top contributing features (vs population):**")
            for fname, delta in result["top_features"]:
                arrow = "↑" if delta > 0 else "↓"
                st.markdown(f"- **{fname}**: {arrow} Δ={delta:+.2f}")
            st.markdown("**Computed clinical scores:**")
            for k in ("egfr", "multimorbidity", "anemia_severity", "cv_risk"):
                if k in result["raw_features"]:
                    st.markdown(f"- **{k}**: {result['raw_features'][k]:.2f}")
