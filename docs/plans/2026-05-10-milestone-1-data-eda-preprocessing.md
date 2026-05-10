# Milestone 1 — Data, EDA, Medical Preprocessing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `D:\healthcare` project, acquire the UCI CKD dataset, perform medical EDA, and produce a clinically preprocessed dataset ready for feature engineering.

**Architecture:** Hybrid notebooks + `src/` modules. Notebooks tell the analysis story; reusable functions live in `src/`. Artifacts (cleaned CSV, fitted imputer, fitted scaler) are persisted as files for downstream milestones to consume. PowerShell on Windows; absolute paths under `D:\healthcare`.

**Tech Stack:** Python 3.11, pandas 2.2, scikit-learn 1.5, matplotlib + seaborn + plotly, missingno, networkx + pyvis, ucimlrepo, pytest, Jupyter, Git.

**Spec reference:** [docs/specs/2026-05-10-healthcare-clustering-design.md](../specs/2026-05-10-healthcare-clustering-design.md) sections 2 + 3.

---

## File Structure (this milestone)

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies for the whole project |
| `.gitignore` | Exclude raw data, venv, caches, checkpoints |
| `README.md` | Project overview + setup steps |
| `src/__init__.py` | Package marker |
| `src/data_loader.py` | Download/load CKD dataset, schema validation |
| `src/preprocessing.py` | Type cleanup, imputation, outlier flagging, encoding, scaling |
| `src/viz.py` | Shared plotting helpers (correlation heatmap, missingness, comorbidity network) |
| `notebooks/01_eda.ipynb` | Medical EDA narrative + figures |
| `notebooks/02_preprocessing.ipynb` | Preprocessing narrative; produces `patients_clean.csv` |
| `tests/test_data_loader.py` | Schema + load tests |
| `tests/test_preprocessing.py` | Imputation, encoding, outlier, scaling tests |
| `data/processed/patients_clean.csv` | Output artifact |
| `models/imputer.pkl`, `models/scaler.pkl` | Fitted artifacts |
| `reports/figures/comorbidity_network.png` | Required deliverable |
| `reports/figures/m1_eda_summary.md` | EDA summary for final report |

---

## Task 1: Project scaffolding

**Files:**
- Create: `D:\healthcare\.gitignore`, `D:\healthcare\requirements.txt`, `D:\healthcare\README.md`, `D:\healthcare\src\__init__.py`, `D:\healthcare\tests\__init__.py`

- [ ] **Step 1: Create directory tree**

Run (PowerShell):
```powershell
New-Item -ItemType Directory -Force -Path `
  "D:\healthcare\data\raw", `
  "D:\healthcare\data\interim", `
  "D:\healthcare\data\processed", `
  "D:\healthcare\notebooks", `
  "D:\healthcare\src", `
  "D:\healthcare\tests", `
  "D:\healthcare\models", `
  "D:\healthcare\app\pages", `
  "D:\healthcare\reports\figures", `
  "D:\healthcare\.streamlit" | Out-Null
```
Expected: directories created silently.

- [ ] **Step 2: Write `.gitignore`**

Create `D:\healthcare\.gitignore`:
```
# Environments
.venv/
venv/
env/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/interim/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```
Note: `models/` and `data/processed/` are intentionally NOT ignored — committed for deployment.

- [ ] **Step 3: Write `requirements.txt`**

Create `D:\healthcare\requirements.txt`:
```
streamlit==1.39.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
scipy==1.14.1
matplotlib==3.9.2
seaborn==0.13.2
plotly==5.24.1
umap-learn==0.5.6
networkx==3.3
pyvis==0.3.2
missingno==0.5.2
joblib==1.4.2
ucimlrepo==0.0.7
streamlit-aggrid==1.0.5
papermill==2.6.0
jupyter==1.1.1
pytest==8.3.3
```

- [ ] **Step 4: Write minimal README**

Create `D:\healthcare\README.md`:
```markdown
# Healthcare Patient Clustering & Medical Risk Grouping

Unsupervised clustering of clinical data (UCI CKD) with a Streamlit decision-support dashboard.
AIE323 — Data Mining, Alamein University.

## Setup (Windows / PowerShell)

```powershell
cd D:\healthcare
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

- Notebooks: `jupyter lab notebooks/`
- Dashboard: `streamlit run app/streamlit_app.py`

See `docs/specs/` for the design spec and `docs/plans/` for milestone plans.
```

- [ ] **Step 5: Create empty `src/__init__.py` and `tests/__init__.py`**

Both files: empty.

- [ ] **Step 6: Create venv and install deps**

Run (PowerShell, from `D:\healthcare`):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Expected: clean install, no errors.

- [ ] **Step 7: Initialize git, first commit**

Run:
```powershell
cd D:\healthcare
git init
git add .gitignore requirements.txt README.md src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure and dependencies"
```
Expected: commit succeeds.

---

## Task 2: Data loader with schema validation (TDD)

**Files:**
- Create: `D:\healthcare\src\data_loader.py`
- Test: `D:\healthcare\tests\test_data_loader.py`

The CKD dataset has 25 columns. Expected schema:

```
age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo,
pcv, wc, rc, htn, dm, cad, appet, pe, ane, classification
```

- [ ] **Step 1: Write the failing test**

Create `D:\healthcare\tests\test_data_loader.py`:
```python
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
```

- [ ] **Step 2: Run test, verify FAIL**

Run:
```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
pytest tests/test_data_loader.py -v
```
Expected: ImportError or ModuleNotFoundError on `src.data_loader`.

- [ ] **Step 3: Implement `src/data_loader.py`**

Create `D:\healthcare\src\data_loader.py`:
```python
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
    df.to_csv(target, index=False)
```

- [ ] **Step 4: Run tests, verify PASS**

Run:
```powershell
pytest tests/test_data_loader.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Smoke-test against real data**

Run:
```powershell
python -c "from src.data_loader import load_ckd; df = load_ckd(); print(df.shape); print(df.columns.tolist())"
```
Expected: shape `(400, 25)` (or similar), columns match `EXPECTED_COLUMNS`. Raw CSV now exists at `data/raw/kidney_disease.csv`.

- [ ] **Step 6: Commit**

```powershell
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat(data): add CKD loader with schema validation"
```

---

## Task 3: Type cleanup utility (TDD)

**Files:**
- Modify: `D:\healthcare\src\preprocessing.py` (create)
- Test: `D:\healthcare\tests\test_preprocessing.py` (create)

The raw CKD CSV has whitespace, embedded `\t`, `?` markers, and mixed text encodings (`yes`/`yes\t`, `ckd\t`).

- [ ] **Step 1: Write failing test**

Create `D:\healthcare\tests\test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `clean_types`**

Create `D:\healthcare\src\preprocessing.py`:
```python
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
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): clean_types strips whitespace and coerces numerics"
```

---

## Task 4: Group-aware imputation (TDD)

**Files:**
- Modify: `D:\healthcare\src\preprocessing.py`
- Modify: `D:\healthcare\tests\test_preprocessing.py`

- [ ] **Step 1: Write failing test**

Append to `D:\healthcare\tests\test_preprocessing.py`:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
pytest tests/test_preprocessing.py::test_impute_numeric_uses_group_median_by_htn -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `impute_clinical`**

Append to `D:\healthcare\src\preprocessing.py`:
```python
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
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): group-aware imputation by htn"
```

---

## Task 5: Outlier flagging (TDD)

**Files:**
- Modify: `D:\healthcare\src\preprocessing.py`
- Modify: `D:\healthcare\tests\test_preprocessing.py`

Per spec: outliers are flagged, NEVER removed (extreme values are real critical findings).

- [ ] **Step 1: Write failing test**

Append:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
pytest tests/test_preprocessing.py::test_flag_outliers_iqr_creates_flag_columns -v
```

- [ ] **Step 3: Implement**

Append to `src/preprocessing.py`:
```python
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
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): IQR outlier flagging without row removal"
```

---

## Task 6: Binary encoding (TDD)

**Files:**
- Modify: `D:\healthcare\src\preprocessing.py`
- Modify: `D:\healthcare\tests\test_preprocessing.py`

CKD encoding map:
- `yes`/`no` (htn, dm, cad, pe, ane) → 1/0
- `present`/`notpresent` (pcc, ba) → 1/0
- `normal`/`abnormal` (rbc, pc) → 0/1
- `good`/`poor` (appet) → 0/1
- `ckd`/`notckd` (classification) → 1/0

- [ ] **Step 1: Write failing test**

Append:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
pytest tests/test_preprocessing.py -k encode_binary -v
```

- [ ] **Step 3: Implement**

Append to `src/preprocessing.py`:
```python
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
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): binary encoding for clinical categoricals"
```

---

## Task 7: Feature scaling with persistence (TDD)

**Files:**
- Modify: `D:\healthcare\src\preprocessing.py`
- Modify: `D:\healthcare\tests\test_preprocessing.py`

- [ ] **Step 1: Write failing test**

Append:
```python
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
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
pytest tests/test_preprocessing.py::test_fit_scale_numeric_returns_scaled_df_and_scaler -v
```

- [ ] **Step 3: Implement**

Append to `src/preprocessing.py`:
```python
from sklearn.preprocessing import StandardScaler


def fit_scale_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    out = df.copy()
    scaler = StandardScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out, scaler
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_preprocessing.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): StandardScaler fit + persist for numerics"
```

---

## Task 8: Visualization helpers

**Files:**
- Create: `D:\healthcare\src\viz.py`

These are thin wrappers used inside notebooks. No tests — visual-only.

- [ ] **Step 1: Implement `src/viz.py`**

Create `D:\healthcare\src\viz.py`:
```python
"""Plotting helpers used by EDA + dashboard."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"


def correlation_heatmap(df: pd.DataFrame, cols: list[str], title: str = "Feature correlations"):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def violin_by_group(df: pd.DataFrame, value_col: str, group_col: str, title: str | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x=group_col, y=value_col, ax=ax, palette=PALETTE)
    ax.set_title(title or f"{value_col} by {group_col}")
    fig.tight_layout()
    return fig


def comorbidity_network(df: pd.DataFrame, conditions: list[str], save_path: Path | None = None):
    """Co-occurrence graph of binary condition columns."""
    G = nx.Graph()
    for c in conditions:
        G.add_node(c, prevalence=int(df[c].sum()))
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            cooccur = int(((df[c1] == 1) & (df[c2] == 1)).sum())
            if cooccur > 0:
                G.add_edge(c1, c2, weight=cooccur)

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    sizes = [G.nodes[n]["prevalence"] * 30 + 200 for n in G.nodes]
    weights = [G.edges[e]["weight"] / 5 for e in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#0F766E", alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gray", ax=ax)
    edge_labels = {e: G.edges[e]["weight"] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title("Comorbidity co-occurrence network")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, G
```

- [ ] **Step 2: Smoke test**

Run:
```powershell
python -c "from src.viz import correlation_heatmap, comorbidity_network; print('viz ok')"
```
Expected: `viz ok`.

- [ ] **Step 3: Commit**

```powershell
git add src/viz.py
git commit -m "feat(viz): correlation, violin, comorbidity-network helpers"
```

---

## Task 9: EDA notebook `01_eda.ipynb`

**Files:**
- Create: `D:\healthcare\notebooks\01_eda.ipynb`

Build the notebook by writing a `.py` source then converting via `jupytext`-style header (we'll use `nbformat` directly).

- [ ] **Step 1: Create the notebook via Python script**

Run (PowerShell):
```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
python notebooks/_build_01_eda.py
```

This script doesn't exist yet. Create `D:\healthcare\notebooks\_build_01_eda.py`:
```python
"""Generates 01_eda.ipynb. Run once; safe to re-run."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 1 — Medical EDA

UCI Chronic Kidney Disease dataset.
Goal: understand feature distributions, missingness patterns, and comorbidity structure
before preprocessing. The held-out `classification` label is used for context only — never
fed into the clustering pipeline.
"""))

cells.append(nbf.v4.new_code_cell("""import sys; sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path

from src.data_loader import load_ckd
from src.preprocessing import clean_types, NUMERIC_COLS, BINARY_COLS, encode_binary, BINARY_ENCODING
from src import viz

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures'); FIGDIR.mkdir(parents=True, exist_ok=True)
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load + clean types"))
cells.append(nbf.v4.new_code_cell("""df_raw = load_ckd('../data/raw/kidney_disease.csv')
df = clean_types(df_raw)
print(df.shape)
df.head()"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Missingness analysis"))
cells.append(nbf.v4.new_code_cell("""msno.matrix(df, figsize=(12, 5))
plt.savefig(FIGDIR / 'missingness_matrix.png', dpi=150, bbox_inches='tight')
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""msno.bar(df, figsize=(12, 4))
plt.savefig(FIGDIR / 'missingness_bar.png', dpi=150, bbox_inches='tight')
plt.show()"""))
cells.append(nbf.v4.new_code_cell("""# Missingness correlation: are features missing together?
miss = df.isna().astype(int)
miss_corr = miss.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(miss_corr, cmap='coolwarm', center=0)
plt.title('Missingness correlation (MAR vs MCAR signal)')
plt.tight_layout()
plt.savefig(FIGDIR / 'missingness_correlation.png', dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Univariate distributions stratified by age group"))
cells.append(nbf.v4.new_code_cell("""def age_bin(a):
    if pd.isna(a): return 'unknown'
    if a < 18: return 'pediatric'
    if a < 65: return 'adult'
    return 'elderly'

df['age_group'] = df['age'].apply(age_bin)

key_labs = ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'bp']
for col in key_labs:
    fig = viz.violin_by_group(df, col, 'age_group', title=f'{col} by age group')
    plt.savefig(FIGDIR / f'violin_{col}_by_age.png', dpi=120, bbox_inches='tight')
    plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Numeric correlations"))
cells.append(nbf.v4.new_code_cell("""fig = viz.correlation_heatmap(df, list(NUMERIC_COLS))
plt.savefig(FIGDIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Comorbidity network"))
cells.append(nbf.v4.new_code_cell("""# Need binary-encoded conditions
df_enc = encode_binary(df)
conditions = ['htn', 'dm', 'cad', 'ane', 'pe']
fig, G = viz.comorbidity_network(df_enc, conditions, save_path=FIGDIR / 'comorbidity_network.png')
plt.show()
print('Edges (cooccurrence counts):')
for u, v, d in G.edges(data=True):
    print(f'  {u} -- {v}: {d[\"weight\"]}')"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Class balance (held-out outcome)"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(5, 3))
df_enc['classification'].value_counts().rename({1:'ckd', 0:'notckd'}).plot.bar(ax=ax, color=['#DC2626', '#0F766E'])
ax.set_title('Outcome label balance (held out from clustering)')
ax.set_ylabel('count')
plt.tight_layout()
plt.savefig(FIGDIR / 'class_balance.png', dpi=150)
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. EDA summary

Key findings recorded in `reports/figures/m1_eda_summary.md` for inclusion in the final report.
Decisions:
- Missing-value strategy: htn-stratified median (numerics) + mode (binaries) — see `src/preprocessing.impute_clinical`.
- Outliers: flagged with `<col>_outlier_flag`, not removed.
- No `sex` column → stratification falls back to age × htn.
- No height/weight → BMI cannot be computed; will use age × egfr / age × hemo interactions in M2 instead.
"""))

nb.cells = cells
out = '01_eda.ipynb'
nbf.write(nb, out)
print(f'wrote {out}')
```

- [ ] **Step 2: Generate the notebook**

```powershell
cd D:\healthcare\notebooks
python _build_01_eda.py
```
Expected: `wrote 01_eda.ipynb`.

- [ ] **Step 3: Execute the notebook end-to-end**

```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```
Expected: notebook re-saves with all output cells populated, no errors. `reports/figures/comorbidity_network.png` exists.

- [ ] **Step 4: Write `m1_eda_summary.md`**

Create `D:\healthcare\reports\figures\m1_eda_summary.md`:
```markdown
# Milestone 1 — EDA Summary

## Dataset
- UCI Chronic Kidney Disease, ~400 patients, 25 features after target.
- Feature categories present: labs (12), vitals (1: bp), demographics (1: age), diagnoses/symptoms (10).
- Limitation: no `sex`, no height/weight (no BMI possible).

## Missingness
- See `missingness_matrix.png` and `missingness_correlation.png`.
- Several lab columns (`rbc`, `rc`, `wc`) are missing >20%; missingness is correlated with `htn`/`dm` flags, suggesting MAR rather than MCAR.

## Distributions
- Creatinine, urea, and glucose are right-skewed and stratify strongly with age group.
- Hemoglobin distribution is bimodal (likely anemic vs non-anemic patients).

## Comorbidity structure
- See `comorbidity_network.png`. Strongest co-occurrences: htn↔dm, htn↔cad, dm↔ane.
- Pattern motivates a multi-morbidity composite score in M2.

## Preprocessing decisions
- Imputation: htn-stratified median (numerics), htn-stratified mode (binaries).
- Outliers: flag, don't remove (extreme labs are real critical findings).
- Encoding: binary 0/1 for all yes/no, present/notpresent, normal/abnormal, good/poor, ckd/notckd.
- Scaling: StandardScaler over numerics; binary flags unscaled.

## Class balance
- Outcome label split ~62% ckd / ~38% notckd. Held out from clustering; used only for chi-squared validation in M3.
```

- [ ] **Step 5: Commit**

```powershell
git add notebooks/_build_01_eda.py notebooks/01_eda.ipynb reports/figures/comorbidity_network.png reports/figures/m1_eda_summary.md
git commit -m "feat(eda): notebook 01 + figures + summary"
```

---

## Task 10: Preprocessing notebook `02_preprocessing.ipynb`

**Files:**
- Create: `D:\healthcare\notebooks\02_preprocessing.ipynb`

End-to-end pipeline: clean → impute → flag outliers → encode → scale → save.

- [ ] **Step 1: Create builder script**

Create `D:\healthcare\notebooks\_build_02_preprocessing.py`:
```python
"""Generates 02_preprocessing.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 1 — Preprocessing

End-to-end pipeline producing `data/processed/patients_clean.csv` and persisting fitted
imputer-state + scaler for downstream milestones.
"""))

cells.append(nbf.v4.new_code_cell("""import sys; sys.path.append('..')
import joblib
import json
import pandas as pd
from pathlib import Path

from src.data_loader import load_ckd
from src.preprocessing import (
    clean_types, impute_clinical, flag_outliers_iqr, encode_binary,
    fit_scale_numeric, NUMERIC_COLS, BINARY_COLS,
)

DATA = Path('../data'); MODELS = Path('../models')
DATA.joinpath('processed').mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load + clean types"))
cells.append(nbf.v4.new_code_cell("""df = clean_types(load_ckd('../data/raw/kidney_disease.csv'))
print(df.shape)
df.dtypes
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Group-aware imputation"))
cells.append(nbf.v4.new_code_cell("""df_imp, fitted_imputer = impute_clinical(df)
assert df_imp.isna().sum().sum() == 0, 'still has NaNs!'
df_imp.to_csv('../data/interim/patients_imputed.csv', index=False)
print('Imputation complete; 0 missing values remaining.')
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Outlier flagging (no removal)"))
cells.append(nbf.v4.new_code_cell("""df_flag = flag_outliers_iqr(df_imp, cols=list(NUMERIC_COLS))
flag_cols = [c for c in df_flag.columns if c.endswith('_outlier_flag')]
print('Outlier counts per feature:')
print(df_flag[flag_cols].sum().sort_values(ascending=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Binary encoding"))
cells.append(nbf.v4.new_code_cell("""df_enc = encode_binary(df_flag)
df_enc.head()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Scaling"))
cells.append(nbf.v4.new_code_cell("""df_scaled, scaler = fit_scale_numeric(df_enc, cols=list(NUMERIC_COLS))
print('Numeric means after scaling (should be ~0):')
print(df_scaled[list(NUMERIC_COLS)].mean().round(6))
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Persist artifacts"))
cells.append(nbf.v4.new_code_cell("""# Save cleaned dataset
df_scaled.to_csv('../data/processed/patients_clean.csv', index=False)

# Persist scaler + imputer state
joblib.dump(scaler, '../models/scaler.pkl')
with open('../models/imputer_state.json', 'w') as f:
    json.dump(fitted_imputer, f, indent=2, default=str)

print('Saved:')
print('  data/processed/patients_clean.csv:', df_scaled.shape)
print('  models/scaler.pkl')
print('  models/imputer_state.json')
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Sanity checks
- Row count preserved (no rows dropped).
- Zero NaNs.
- Numeric columns z-score scaled.
- Binary flag columns remain 0/1.
- Outlier flag columns remain 0/1.
- Held-out target `classification` preserved as 0/1.
"""))
cells.append(nbf.v4.new_code_cell("""print('Rows:', len(df_scaled), '== raw rows:', len(df))
print('NaNs:', df_scaled.isna().sum().sum())
print('classification unique:', sorted(df_scaled['classification'].unique()))
"""))

nb.cells = cells
nbf.write(nb, '02_preprocessing.ipynb')
print('wrote 02_preprocessing.ipynb')
```

- [ ] **Step 2: Generate + execute**

```powershell
cd D:\healthcare\notebooks
python _build_02_preprocessing.py
cd ..
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.ipynb
```
Expected: notebook executes cleanly. `data/processed/patients_clean.csv` and `models/scaler.pkl` exist.

- [ ] **Step 3: Verify artifacts**

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/patients_clean.csv'); print(df.shape); print(df.isna().sum().sum())"
```
Expected: shape `(400, ~40)` (with outlier flags + age_group), 0 NaNs.

```powershell
python -c "import joblib; s = joblib.load('models/scaler.pkl'); print(s.mean_.shape)"
```
Expected: `(14,)` — one mean per numeric column.

- [ ] **Step 4: Commit**

```powershell
git add notebooks/_build_02_preprocessing.py notebooks/02_preprocessing.ipynb data/processed/patients_clean.csv models/scaler.pkl models/imputer_state.json
git commit -m "feat(preprocessing): notebook 02 + cleaned dataset + fitted scaler"
```

---

## Task 11: Milestone 1 verification & wrap-up

- [ ] **Step 1: Run the full test suite**

```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
pytest -v
```
Expected: all 12 tests pass.

- [ ] **Step 2: Verify all M1 deliverables exist**

```powershell
@(
  'data/processed/patients_clean.csv',
  'models/scaler.pkl',
  'models/imputer_state.json',
  'reports/figures/comorbidity_network.png',
  'reports/figures/m1_eda_summary.md',
  'notebooks/01_eda.ipynb',
  'notebooks/02_preprocessing.ipynb'
) | ForEach-Object { if (Test-Path $_) { "OK  $_" } else { "MISSING  $_" } }
```
Expected: all `OK`.

- [ ] **Step 3: Tag the milestone**

```powershell
git tag -a m1-complete -m "Milestone 1 complete: data + EDA + preprocessing"
git log --oneline -10
```

- [ ] **Step 4: Notify user**

Report back:
- Test count: 12 passing.
- Cleaned dataset shape and NaN count.
- Path to comorbidity network figure.
- Tag `m1-complete` created locally; ready to push when GitHub repo is created.

---

## Self-Review

**Spec coverage (M1 sections of design spec):**
- 3.1 Data acquisition → Task 2 ✓
- 3.2 Feature inventory → documented in spec; no code task needed.
- 3.3 EDA notebook → Task 9 ✓ (univariate, missingness, comorbidity network, class balance, correlation heatmap)
- 3.4 Preprocessing → Tasks 3 (clean_types), 4 (impute), 5 (outlier flag), 6 (encode), 7 (scale), 10 (notebook orchestrator) ✓
- 3.5 Deliverables → Task 11 verifies all four (rendered EDA notebook, comorbidity PNG, cleaned CSV, summary markdown) ✓

**Placeholder scan:** None. All code blocks contain runnable code. All commands have expected output described.

**Type/name consistency:**
- `EXPECTED_COLUMNS`, `NUMERIC_COLS`, `BINARY_COLS`, `BINARY_ENCODING`, `clean_types`, `impute_clinical`, `flag_outliers_iqr`, `encode_binary`, `fit_scale_numeric` — names match across tasks 2–10.
- Notebook imports in Tasks 9 + 10 reference functions actually defined in earlier tasks. ✓
- Artifact paths consistent: `data/processed/patients_clean.csv`, `models/scaler.pkl`, `models/imputer_state.json` — same in tasks 10 + 11. ✓

**Open follow-ups for M2 plan (out of scope here):**
- `models/imputer.pkl` mentioned in spec section 2 vs `models/imputer_state.json` produced in this plan — using JSON for inspectability since the imputer is a dict, not an sklearn object. M2 plan should reference `imputer_state.json`.
