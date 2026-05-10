import numpy as np
import pandas as pd
import pytest
from src.preprocessing import clean_types, NUMERIC_COLS, BINARY_COLS


def test_clean_types_strips_whitespace_and_tabs():
    df = pd.DataFrame({
        "age": ["48", "  62 ", "\t30"],
        "htn": ["yes", "yes\t", " no "],
        "classification": ["ckd", "ckd\t", "notckd"],
    })
    out = clean_types(df)
    assert out["age"].tolist() == [48.0, 62.0, 30.0]
    assert out["htn"].tolist() == ["yes", "yes", "no"]
    assert out["classification"].tolist() == ["ckd", "ckd", "notckd"]


def test_clean_types_replaces_question_mark_with_nan():
    df = pd.DataFrame({"sc": ["1.2", "?", "0.9"], "rbc": ["?", "normal", "abnormal"]})
    out = clean_types(df)
    assert pd.isna(out["sc"].iloc[1])
    assert pd.isna(out["rbc"].iloc[0])


def test_numeric_cols_are_coerced_to_float():
    df = pd.DataFrame({"sc": ["1.2", "0.9"], "bgr": ["117", "200"], "htn": ["yes", "no"]})
    out = clean_types(df)
    assert out["sc"].dtype.kind == "f"
    assert out["bgr"].dtype.kind == "f"
    assert out["htn"].dtype == object


from src.preprocessing import impute_clinical


def test_impute_numeric_uses_group_median_by_htn():
    df = pd.DataFrame({
        "htn": ["yes", "yes", "yes", "no", "no", "no"],
        "sc":  [2.0, 4.0, np.nan, 0.8, 1.0, np.nan],
        "bgr": [200.0, np.nan, 220.0, 100.0, 110.0, np.nan],
        "rbc": ["abnormal", np.nan, "abnormal", "normal", "normal", "normal"],
    })
    out, fitted = impute_clinical(df)
    assert out["sc"].iloc[2] == 3.0     # median of yes-group {2,4}
    assert out["sc"].iloc[5] == 0.9     # median of no-group {0.8,1.0}
    assert out["bgr"].iloc[1] == 210.0  # median {200,220}
    assert out["bgr"].iloc[5] == 105.0  # median {100,110}
    assert out["rbc"].iloc[1] == "abnormal"  # mode of yes-group
    assert out["rbc"].notna().all()
    assert "numeric_medians" in fitted
    assert "categorical_modes" in fitted


def test_impute_handles_missing_htn_value():
    df = pd.DataFrame({
        "htn": ["yes", np.nan, "no"],
        "sc": [2.0, 1.5, 0.8],
        "rbc": ["abnormal", "normal", "normal"],
    })
    out, _ = impute_clinical(df)
    # htn itself imputed via mode
    assert out["htn"].notna().all()
