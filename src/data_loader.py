"""Load and validate the UCI Chronic Kidney Disease dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = (
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
    "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane", "classification",
)

DEFAULT_RAW_PATH = Path("data/raw/kidney_disease.csv")


def validate_schema(df: pd.DataFrame) -> None:
    cols = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - cols
    extra = cols - expected
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")


def load_ckd(path: Path | str = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load CKD CSV from disk; auto-fetch via ucimlrepo if missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _fetch_from_ucimlrepo(path)
    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    validate_schema(df)
    return df


def _fetch_from_ucimlrepo(target: Path) -> None:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise RuntimeError(
            "ucimlrepo not installed. Run `pip install ucimlrepo` "
            f"or place the CKD CSV at {target}."
        ) from e
    ds = fetch_ucirepo(id=336)  # Chronic Kidney Disease
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [c.strip().lower() for c in df.columns]
    # UCI metadata uses abbreviated names that differ from EXPECTED_COLUMNS:
    #   wbcc -> wc (white blood cell count)
    #   rbcc -> rc (red blood cell count)
    #   class -> classification (target label)
    df = df.rename(columns={"wbcc": "wc", "rbcc": "rc", "class": "classification"})
    df.to_csv(target, index=False)
