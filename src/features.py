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
