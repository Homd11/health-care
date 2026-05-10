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
