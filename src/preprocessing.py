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
