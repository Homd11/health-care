"""Single-patient inference: dict -> cluster + risk tier + confidence."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import clean_types, BINARY_ENCODING
from src.features import (
    compute_egfr, multimorbidity_score, compute_anemia_severity, compute_cv_risk,
)

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=1)
def _load_artifacts():
    imputer = joblib.load(ARTIFACTS_ROOT / "imputer.pkl")
    scaler = joblib.load(ARTIFACTS_ROOT / "core_scaler.pkl")
    km = joblib.load(ARTIFACTS_ROOT / "kmeans.pkl")
    gmm = joblib.load(ARTIFACTS_ROOT / "gmm.pkl")
    with open(ARTIFACTS_ROOT / "core_features.json") as f:
        cfg = json.load(f)
    with open(ARTIFACTS_ROOT / "cluster_profiles.json") as f:
        profiles = json.load(f)
    return imputer, scaler, km, gmm, cfg["core_features"], profiles


def _to_clinical_frame(row: dict) -> pd.DataFrame:
    """Apply M1+M2 transforms to a single-row dict, producing the clinical core matrix."""
    df = pd.DataFrame([row])
    df = clean_types(df)
    imputer, *_ = _load_artifacts()
    df = imputer.transform(df)
    # encode binaries
    for col, mapping in BINARY_ENCODING.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(mapping).astype("Int64")
    df = compute_egfr(df)
    df = multimorbidity_score(df)
    df = compute_anemia_severity(df)
    df = compute_cv_risk(df)
    return df


def predict_patient(patient: dict) -> dict:
    """Run the full pipeline for one patient.

    Returns: {cluster_id, cluster_name, risk_tier, confidence, summary, top_features, raw_features}
    """
    imputer, scaler, km, gmm, core_features, profiles = _load_artifacts()
    df = _to_clinical_frame(patient)
    core = df[core_features].astype(float)
    core_scaled = pd.DataFrame(scaler.transform(core), columns=core_features)
    cluster_id = int(km.predict(core_scaled)[0])
    proba = gmm.predict_proba(core_scaled)[0]
    confidence = float(proba.max())

    profile = profiles[str(cluster_id)]
    raw_features = {f: float(df[f].iloc[0]) for f in core_features}

    # Top contributing features: largest |delta from population mean|, normalized
    deltas = profile["feature_deltas"]
    top = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:3]

    summary = (
        f"Cluster {cluster_id}: {profile['name']}. Risk tier: {profile['risk_tier']}. "
        f"Cluster has n={profile['size']} patients with mean eGFR "
        f"{profile['feature_means'].get('egfr', 0):.1f} mL/min/1.73m²."
    )
    return {
        "cluster_id": cluster_id,
        "cluster_name": profile["name"],
        "risk_tier": profile["risk_tier"],
        "confidence": confidence,
        "summary": summary,
        "top_features": top,
        "raw_features": raw_features,
        "feature_means": profile["feature_means"],
    }
