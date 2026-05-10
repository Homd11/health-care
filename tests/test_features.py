import numpy as np
import pandas as pd
import pytest
from src.features import compute_egfr


def test_compute_egfr_known_values():
    df = pd.DataFrame({"sc": [0.7, 1.5, 3.0], "age": [40.0, 60.0, 75.0]})
    out = compute_egfr(df)
    assert "egfr" in out.columns
    assert 105 < out["egfr"].iloc[0] < 115
    assert out["egfr"].iloc[2] < 30
    assert out["egfr"].iloc[0] > out["egfr"].iloc[1] > out["egfr"].iloc[2]


def test_compute_egfr_handles_missing():
    df = pd.DataFrame({"sc": [1.0, np.nan], "age": [40.0, 50.0]})
    out = compute_egfr(df)
    assert pd.isna(out["egfr"].iloc[1])
    assert pd.notna(out["egfr"].iloc[0])
