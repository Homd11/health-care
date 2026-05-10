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


BINARY_ENCODING = {
    "htn":  {"yes": 1, "no": 0},
    "dm":   {"yes": 1, "no": 0},
    "cad":  {"yes": 1, "no": 0},
    "pe":   {"yes": 1, "no": 0},
    "ane":  {"yes": 1, "no": 0},
    "pcc":  {"present": 1, "notpresent": 0},
    "ba":   {"present": 1, "notpresent": 0},
    "rbc":  {"normal": 0, "abnormal": 1},
    "pc":   {"normal": 0, "abnormal": 1},
    "appet":{"good": 0, "poor": 1},
    "classification": {"ckd": 1, "notckd": 0},
}


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in BINARY_ENCODING.items():
        if col in out.columns:
            out[col] = out[col].map(mapping).astype("Int64")
    return out


def flag_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Add `<col>_outlier_flag` columns. Does NOT remove rows.

    Extreme clinical values are typically real critical findings, not noise.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        out[f"{col}_outlier_flag"] = ((out[col] < lo) | (out[col] > hi)).astype(int)
    return out


class ClinicalImputer:
    """Stateful imputer: htn-stratified median for numerics, mode for binaries.

    Fit on training data; transform new patient rows in inference (dashboard).
    Persisted via joblib.
    """

    def __init__(self) -> None:
        self.numeric_medians: dict[str, dict[Any, float]] = {}
        self.categorical_modes: dict[str, dict[Any, str]] = {}
        self.htn_global_mode: str | None = None

    def fit(self, df: pd.DataFrame) -> "ClinicalImputer":
        out = df.copy()
        if "htn" in out.columns:
            self.htn_global_mode = out["htn"].mode(dropna=True).iloc[0]
            out["htn"] = out["htn"].fillna(self.htn_global_mode)

        for col in NUMERIC_COLS:
            if col not in out.columns:
                continue
            medians = out.groupby("htn")[col].median().to_dict()
            global_median = float(out[col].median())
            # Replace any NaN group medians with global
            medians = {k: (global_median if pd.isna(v) else float(v)) for k, v in medians.items()}
            medians["_global"] = global_median
            self.numeric_medians[col] = medians

        for col in BINARY_COLS:
            if col not in out.columns or col == "htn":
                continue
            modes = out.groupby("htn")[col].agg(
                lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
            ).to_dict()
            global_mode = out[col].mode(dropna=True).iloc[0]
            modes = {k: (global_mode if pd.isna(v) else v) for k, v in modes.items()}
            modes["_global"] = global_mode
            self.categorical_modes[col] = modes

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "htn" in out.columns and self.htn_global_mode is not None:
            out["htn"] = out["htn"].fillna(self.htn_global_mode)

        for col, medians in self.numeric_medians.items():
            if col not in out.columns:
                continue
            global_median = medians["_global"]
            fill = out["htn"].map(medians).fillna(global_median) if "htn" in out.columns else global_median
            out[col] = out[col].fillna(fill)

        for col, modes in self.categorical_modes.items():
            if col not in out.columns:
                continue
            global_mode = modes["_global"]
            fill = out["htn"].map(modes).fillna(global_mode) if "htn" in out.columns else global_mode
            out[col] = out[col].fillna(fill)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


from sklearn.preprocessing import StandardScaler


def fit_scale_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    out = df.copy()
    scaler = StandardScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out, scaler
