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


from src.preprocessing import flag_outliers_iqr


def test_flag_outliers_iqr_creates_flag_columns():
    df = pd.DataFrame({"sc": [1.0, 1.1, 0.9, 1.2, 1.0, 76.0],
                       "bgr": [100, 110, 105, 115, 108, 112]})
    out = flag_outliers_iqr(df, cols=["sc", "bgr"])
    assert "sc_outlier_flag" in out.columns
    assert "bgr_outlier_flag" in out.columns
    assert out["sc_outlier_flag"].iloc[5] == 1     # 76.0 is an outlier
    assert out["sc_outlier_flag"].iloc[:5].sum() == 0
    assert out["bgr_outlier_flag"].sum() == 0      # all normal


def test_flag_outliers_does_not_remove_rows():
    df = pd.DataFrame({"sc": [1.0, 76.0]})
    out = flag_outliers_iqr(df, cols=["sc"])
    assert len(out) == 2  # row preserved


from src.preprocessing import encode_binary


def test_encode_binary_yes_no():
    df = pd.DataFrame({"htn": ["yes", "no", "yes"], "dm": ["no", "yes", "no"]})
    out = encode_binary(df)
    assert out["htn"].tolist() == [1, 0, 1]
    assert out["dm"].tolist() == [0, 1, 0]


def test_encode_binary_present_notpresent():
    df = pd.DataFrame({"pcc": ["present", "notpresent"], "ba": ["notpresent", "present"]})
    out = encode_binary(df)
    assert out["pcc"].tolist() == [1, 0]
    assert out["ba"].tolist() == [0, 1]


def test_encode_binary_normal_abnormal():
    df = pd.DataFrame({"rbc": ["normal", "abnormal"], "pc": ["abnormal", "normal"]})
    out = encode_binary(df)
    assert out["rbc"].tolist() == [0, 1]
    assert out["pc"].tolist() == [1, 0]


def test_encode_binary_appetite_and_classification():
    df = pd.DataFrame({"appet": ["good", "poor"],
                       "classification": ["ckd", "notckd"]})
    out = encode_binary(df)
    assert out["appet"].tolist() == [0, 1]
    assert out["classification"].tolist() == [1, 0]


import joblib
from src.preprocessing import fit_scale_numeric


def test_fit_scale_numeric_returns_scaled_df_and_scaler(tmp_path):
    df = pd.DataFrame({
        "age": [10.0, 20.0, 30.0, 40.0],
        "bgr": [100.0, 110.0, 120.0, 130.0],
        "htn": [1, 0, 1, 0],  # binary, should NOT be scaled
    })
    out, scaler = fit_scale_numeric(df, cols=["age", "bgr"])
    # scaled columns have ~zero mean, ~unit std
    assert abs(out["age"].mean()) < 1e-9
    assert abs(out["age"].std(ddof=0) - 1.0) < 1e-9
    # binary col untouched
    assert out["htn"].tolist() == [1, 0, 1, 0]
    # scaler can be persisted + reloaded
    p = tmp_path / "scaler.pkl"
    joblib.dump(scaler, p)
    reloaded = joblib.load(p)
    assert reloaded.mean_.shape == (2,)


from src.preprocessing import ClinicalImputer


def test_clinical_imputer_fit_transform_roundtrip(tmp_path):
    df = pd.DataFrame({
        "htn": ["yes", "yes", "no", "no"],
        "sc":  [2.0, 4.0, 0.8, 1.0],
        "bgr": [200.0, 220.0, 100.0, 110.0],
        "rbc": ["abnormal", "abnormal", "normal", "normal"],
    })
    imputer = ClinicalImputer().fit(df)

    new_row = pd.DataFrame({"htn": ["yes"], "sc": [np.nan], "bgr": [np.nan], "rbc": [np.nan]})
    out = imputer.transform(new_row)
    assert out["sc"].iloc[0] == 3.0      # yes-group median {2,4}
    assert out["bgr"].iloc[0] == 210.0   # yes-group median {200,220}
    assert out["rbc"].iloc[0] == "abnormal"

    # round-trip via joblib
    p = tmp_path / "imputer.pkl"
    joblib.dump(imputer, p)
    reloaded = joblib.load(p)
    out2 = reloaded.transform(new_row)
    assert out2["sc"].iloc[0] == 3.0


def test_clinical_imputer_handles_unseen_htn_value():
    df = pd.DataFrame({"htn": ["yes", "no"], "sc": [2.0, 1.0], "rbc": ["abnormal", "normal"]})
    imputer = ClinicalImputer().fit(df)
    new = pd.DataFrame({"htn": ["maybe"], "sc": [np.nan], "rbc": [np.nan]})
    out = imputer.transform(new)
    # falls back to global median 1.5 / global mode (any of the two)
    assert out["sc"].iloc[0] == 1.5
    assert out["rbc"].iloc[0] in ("abnormal", "normal")
