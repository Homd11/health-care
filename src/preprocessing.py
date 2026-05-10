"""Medical preprocessing: type cleanup, imputation, outlier flagging, encoding, scaling."""
from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_COLS = ("age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
                "sod", "pot", "hemo", "pcv", "wc", "rc")
BINARY_COLS = ("rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
               "appet", "pe", "ane")
TARGET_COL = "classification"


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = (
                out[col].astype(str).str.replace("\t", "", regex=False).str.strip()
            )
            out[col] = out[col].replace({"?": np.nan, "nan": np.nan, "": np.nan})
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


from typing import Any


def impute_clinical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Impute numeric labs by htn-group median; categoricals by htn-group mode.

    Rationale: missingness in clinical data is often related to disease severity,
    so imputing within htn-stratified groups preserves disease-related distribution
    shifts better than a global median.
    """
    out = df.copy()
    if "htn" in out.columns and out["htn"].isna().any():
        global_htn_mode = out["htn"].mode(dropna=True).iloc[0]
        out["htn"] = out["htn"].fillna(global_htn_mode)

    numeric_medians: dict[str, dict[str, float]] = {}
    for col in NUMERIC_COLS:
        if col not in out.columns:
            continue
        medians = out.groupby("htn")[col].median().to_dict()
        global_median = float(out[col].median())
        numeric_medians[col] = {**medians, "_global": global_median}
        out[col] = out.apply(
            lambda r, c=col: medians.get(r["htn"], global_median)
                if pd.isna(r[c]) else r[c],
            axis=1,
        )

    categorical_modes: dict[str, dict[str, str]] = {}
    for col in BINARY_COLS:
        if col not in out.columns or col == "htn":
            continue
        modes = out.groupby("htn")[col].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        ).to_dict()
        global_mode = out[col].mode(dropna=True).iloc[0]
        categorical_modes[col] = {**modes, "_global": global_mode}
        out[col] = out.apply(
            lambda r, c=col: modes.get(r["htn"], global_mode)
                if pd.isna(r[c]) else r[c],
            axis=1,
        )

    fitted = {
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
    }
    return out, fitted
