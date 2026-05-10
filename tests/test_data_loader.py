import pandas as pd
import pytest
from src.data_loader import EXPECTED_COLUMNS, validate_schema, load_ckd


def test_expected_columns_constant():
    assert len(EXPECTED_COLUMNS) == 25
    assert "classification" in EXPECTED_COLUMNS
    assert "age" in EXPECTED_COLUMNS
    assert "sc" in EXPECTED_COLUMNS  # creatinine


def test_validate_schema_passes_on_correct_df():
    df = pd.DataFrame({c: [0] for c in EXPECTED_COLUMNS})
    validate_schema(df)  # should not raise


def test_validate_schema_raises_on_missing_column():
    df = pd.DataFrame({c: [0] for c in EXPECTED_COLUMNS if c != "classification"})
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_schema(df)


def test_validate_schema_raises_on_extra_columns():
    cols = list(EXPECTED_COLUMNS) + ["unexpected_col"]
    df = pd.DataFrame({c: [0] for c in cols})
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_schema(df)
