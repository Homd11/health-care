"""Clinical feature engineering: composite scores, threshold flags, interactions."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_egfr(df: pd.DataFrame) -> pd.DataFrame:
    """CKD-EPI 2021 sex-agnostic eGFR (mL/min/1.73m^2).

    Limitation: the CKD dataset lacks a sex column, so we use the female-coefficient
    formulation conservatively. This may underestimate eGFR for males by ~10%.
    """
    out = df.copy()
    sc = out["sc"].astype(float)
    age = out["age"].astype(float)
    kappa = 0.7
    ratio = sc / kappa
    low = np.where(ratio <= 1, ratio, 1.0)
    high = np.where(ratio > 1, ratio, 1.0)
    out["egfr"] = 142.0 * (low ** -0.241) * (high ** -1.200) * (0.9938 ** age)
    out.loc[sc.isna() | age.isna(), "egfr"] = np.nan
    return out


THRESHOLD_FLAG_DEFINITIONS: dict[str, tuple[str, str, float]] = {
    "flag_hyperglycemia":     ("bgr",  ">=", 200),
    "flag_hypertensive":      ("bp",   ">=", 140),
    "flag_anemia":            ("hemo", "<",  12),
    "flag_hyperkalemia":      ("pot",  ">",  5.0),
    "flag_hyponatremia":      ("sod",  "<",  135),
    "flag_renal_impairment":  ("sc",   ">",  1.3),
    "flag_proteinuria":       ("al",   ">=", 2),
    "flag_low_egfr":          ("egfr", "<",  60),
}


def threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary clinical threshold flags. Skips flags whose source column is absent."""
    out = df.copy()
    ops = {
        ">":  lambda s, t: s > t,
        ">=": lambda s, t: s >= t,
        "<":  lambda s, t: s < t,
        "<=": lambda s, t: s <= t,
    }
    for flag, (col, op, thr) in THRESHOLD_FLAG_DEFINITIONS.items():
        if col not in out.columns:
            continue
        out[flag] = ops[op](out[col], thr).astype(int)
    return out


MORBIDITY_COLS = ("htn", "dm", "cad", "ane", "pe")


def multimorbidity_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in MORBIDITY_COLS if c in out.columns]
    out["multimorbidity"] = out[cols].sum(axis=1).astype(int)
    return out


def _age_bin(a: float) -> str:
    if pd.isna(a):
        return "unknown"
    if a < 18:
        return "pediatric"
    if a < 65:
        return "adult"
    return "elderly"


def age_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = out["age"].apply(_age_bin)
    for grp in ("pediatric", "adult", "elderly"):
        out[f"age_{grp}"] = (out["age_group"] == grp).astype(int)
    return out


def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_x_creatinine"] = out["age"] * out["sc"]
    out["age_x_bp"] = out["age"] * out["bp"]
    if "egfr" in out.columns:
        out["age_x_egfr"] = out["age"] * out["egfr"]
    if "hemo" in out.columns:
        out["age_x_hemo"] = out["age"] * out["hemo"]
    return out


def compute_anemia_severity(df: pd.DataFrame) -> pd.DataFrame:
    """WHO anemia severity bins (0=normal, 1=mild, 2=moderate, 3=severe)."""
    out = df.copy()
    h = out["hemo"]
    sev = pd.Series(0, index=out.index, dtype=int)
    sev[(h >= 10) & (h < 12)] = 1
    sev[(h > 7)   & (h < 10)] = 2
    sev[h <= 7] = 3
    out["anemia_severity"] = sev
    return out


def compute_cv_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Additive cardiovascular risk index (0-5)."""
    out = df.copy()
    out["cv_risk"] = (
        out["htn"].astype(int)
        + out["dm"].astype(int)
        + out["cad"].astype(int)
        + (out["age"] >= 65).astype(int)
        + (out["bp"] >= 140).astype(int)
    )
    return out


def compute_electrolyte_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Count of {sod, pot} outside reference range."""
    out = df.copy()
    sod_oor = (out["sod"] < 135) | (out["sod"] > 145)
    pot_oor = (out["pot"] < 3.5) | (out["pot"] > 5.0)
    out["electrolyte_imbalance"] = sod_oor.astype(int) + pot_oor.astype(int)
    return out
