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


from src.features import threshold_flags, THRESHOLD_FLAG_DEFINITIONS


def test_threshold_flags_creates_all_expected_columns():
    df = pd.DataFrame({
        "bgr":  [150, 250, 100],
        "bp":   [120, 150, 130],
        "hemo": [14.0, 10.0, 13.5],
        "pot":  [4.0, 5.5, 4.5],
        "sod":  [140, 130, 138],
        "sc":   [0.9, 1.5, 1.0],
        "al":   [0, 3, 1],
        "egfr": [90.0, 50.0, 70.0],
    })
    out = threshold_flags(df)
    expected_flags = {
        "flag_hyperglycemia", "flag_hypertensive", "flag_anemia",
        "flag_hyperkalemia", "flag_hyponatremia", "flag_renal_impairment",
        "flag_proteinuria", "flag_low_egfr",
    }
    assert expected_flags.issubset(out.columns)


def test_threshold_flags_correctness():
    df = pd.DataFrame({
        "bgr":  [199, 200, 201],
        "bp":   [139, 140, 141],
        "hemo": [12.0, 11.9, 12.1],
        "pot":  [5.0, 5.1, 4.9],
        "sod":  [135, 134, 136],
        "sc":   [1.3, 1.4, 1.2],
        "al":   [1, 2, 0],
        "egfr": [60, 59, 61],
    })
    out = threshold_flags(df)
    assert out["flag_hyperglycemia"].tolist() == [0, 1, 1]
    assert out["flag_hypertensive"].tolist()  == [0, 1, 1]
    assert out["flag_anemia"].tolist()         == [0, 1, 0]
    assert out["flag_hyperkalemia"].tolist()   == [0, 1, 0]
    assert out["flag_hyponatremia"].tolist()   == [0, 1, 0]
    assert out["flag_renal_impairment"].tolist() == [0, 1, 0]
    assert out["flag_proteinuria"].tolist()    == [0, 1, 0]
    assert out["flag_low_egfr"].tolist()       == [0, 1, 0]


def test_threshold_flags_skips_missing_columns():
    df = pd.DataFrame({"bgr": [100], "bp": [120]})
    out = threshold_flags(df)
    assert "flag_hyperglycemia" in out.columns
    assert "flag_hypertensive" in out.columns
    assert "flag_anemia" not in out.columns
