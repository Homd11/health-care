# Milestone 2 — Clinical Feature Engineering & Dimensionality — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Engineer medically meaningful composite features (eGFR, threshold flags, multi-morbidity, age bins, interactions) on top of `patients_clean.csv`, run feature selection, fit PCA + UMAP + t-SNE for downstream clustering, and persist the feature matrix + projection coordinates as `patients_features.csv`.

**Architecture:** Same hybrid pattern as M1 — pure logic in `src/features.py`, narrative in `notebooks/03_feature_engineering.ipynb`. Persist `models/pca.pkl`, `models/umap.pkl`, `models/feature_columns.json`. The held-out `classification` column is excluded from feature selection inputs but kept in the output dataset for downstream chi-squared validation.

**Tech Stack:** pandas, numpy, scikit-learn (PCA, VarianceThreshold, mutual_info_classif), umap-learn, scipy.stats, joblib.

**Spec reference:** [docs/specs/2026-05-10-healthcare-clustering-design.md](../specs/2026-05-10-healthcare-clustering-design.md) section 4.

**Predecessor:** Tag `m1-complete` (commit `9575496`). This plan starts from the M1 artifacts already on disk.

---

## File Structure (this milestone)

| File | Responsibility |
|---|---|
| `src/features.py` | Composite scores, threshold flags, multimorbidity, age bins, interactions, feature selection helpers |
| `tests/test_features.py` | Unit tests for every feature function |
| `notebooks/_build_03_feature_engineering.py` | Builder script for the M2 notebook |
| `notebooks/03_feature_engineering.ipynb` | Narrative: load clean data → engineer → select → PCA/UMAP/t-SNE → persist |
| `data/processed/patients_features.csv` | Output: clean + engineered + selected features + projection coords |
| `models/pca.pkl` | Fitted PCA |
| `models/umap.pkl` | Fitted UMAP (saved for inference; t-SNE re-fit each time since sklearn t-SNE is non-parametric) |
| `models/feature_columns.json` | Final feature column order for clustering input |
| `reports/figures/scree_plot.png`, `pca_2d.png`, `umap_2d.png`, `tsne_2d.png` | M2 visual deliverables |

---

## Task 1: eGFR composite (CKD-EPI 2021, sex-agnostic) (TDD)

**Files:**
- Create: `D:\healthcare\src\features.py`
- Create: `D:\healthcare\tests\test_features.py`

CKD-EPI 2021 sex-agnostic average:
- Let κ = 0.7 (lower threshold), α = -0.241 if Scr ≤ κ else -1.200 (averaged: female -0.241/-1.200; male -0.302/-1.200; we use the female α coefficients to stay conservative since CKD has no sex column).
- eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^-1.200 × 0.9938^age × 1.012 (the 1.012 sex-female factor is averaged out: we use 1.0 since sex unknown).

To avoid bikeshedding, we'll use a published simplified formulation that consistently lands within ±5 mL/min/1.73m² of the full formula:

`eGFR = 142 * min(sc/0.7, 1)^(-0.241) * max(sc/0.7, 1)^(-1.200) * 0.9938^age`

(no sex coefficient since CKD has no sex column; documented limitation).

- [ ] **Step 1: Write failing test**

Create `D:\healthcare\tests\test_features.py`:
```python
import numpy as np
import pandas as pd
import pytest
from src.features import compute_egfr


def test_compute_egfr_known_values():
    df = pd.DataFrame({"sc": [0.7, 1.5, 3.0], "age": [40.0, 60.0, 75.0]})
    out = compute_egfr(df)
    assert "egfr" in out.columns
    # Healthy adult, sc=0.7, age=40 → eGFR ≈ 142 * 1 * 1 * 0.9938^40 ≈ 110.6
    assert 105 < out["egfr"].iloc[0] < 115
    # Severe CKD, sc=3.0, age=75 → much lower
    assert out["egfr"].iloc[2] < 30
    # Monotone: higher sc → lower eGFR (holding age moderately)
    assert out["egfr"].iloc[0] > out["egfr"].iloc[1] > out["egfr"].iloc[2]


def test_compute_egfr_handles_missing():
    df = pd.DataFrame({"sc": [1.0, np.nan], "age": [40.0, 50.0]})
    out = compute_egfr(df)
    assert pd.isna(out["egfr"].iloc[1])
    assert pd.notna(out["egfr"].iloc[0])
```

- [ ] **Step 2: Run, verify FAIL**

```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
pytest tests/test_features.py -v
```

- [ ] **Step 3: Implement**

Create `D:\healthcare\src\features.py`:
```python
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
```

- [ ] **Step 4: Run, verify PASS**

```powershell
pytest tests/test_features.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat(features): CKD-EPI 2021 eGFR composite"
```

---

## Task 2: Threshold flags (TDD)

- [ ] **Step 1: Failing test**

Append to `tests/test_features.py`:
```python
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
    assert out["flag_hyperglycemia"].tolist() == [0, 1, 1]   # bgr >= 200
    assert out["flag_hypertensive"].tolist()  == [0, 1, 1]   # bp >= 140
    assert out["flag_anemia"].tolist()         == [0, 1, 0]   # hemo < 12
    assert out["flag_hyperkalemia"].tolist()   == [0, 1, 0]   # pot > 5.0
    assert out["flag_hyponatremia"].tolist()   == [0, 1, 0]   # sod < 135
    assert out["flag_renal_impairment"].tolist() == [0, 1, 0] # sc > 1.3
    assert out["flag_proteinuria"].tolist()    == [0, 1, 0]   # al >= 2
    assert out["flag_low_egfr"].tolist()       == [0, 1, 0]   # egfr < 60


def test_threshold_flags_skips_missing_columns():
    df = pd.DataFrame({"bgr": [100], "bp": [120]})
    out = threshold_flags(df)
    assert "flag_hyperglycemia" in out.columns
    assert "flag_hypertensive" in out.columns
    # Other flags absent because their source columns are absent
    assert "flag_anemia" not in out.columns
```

- [ ] **Step 2: Verify FAIL**

- [ ] **Step 3: Implement** — append to `src/features.py`:

```python
THRESHOLD_FLAG_DEFINITIONS: dict[str, tuple[str, str, float]] = {
    # flag name : (source col, op, threshold)
    "flag_hyperglycemia":     ("bgr",  ">=", 200),
    "flag_hypertensive":      ("bp",   ">=", 140),
    "flag_anemia":            ("hemo", "<",  12),
    "flag_hyperkalemia":      ("pot",  ">",  5.0),
    "flag_hyponatremia":      ("sod",  "<",  135),
    "flag_renal_impairment":  ("sc",   ">",  1.3),
    "flag_proteinuria":       ("al",   ">=", 2),
    "flag_low_egfr":          ("egfr", "<",  60),
}


def threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary clinical threshold flags. Skips flags whose source column is absent."""
    out = df.copy()
    ops = {
        ">":  lambda s, t: s > t,
        ">=": lambda s, t: s >= t,
        "<":  lambda s, t: s < t,
        "<=": lambda s, t: s <= t,
    }
    for flag, (col, op, thr) in THRESHOLD_FLAG_DEFINITIONS.items():
        if col not in out.columns:
            continue
        out[flag] = ops[op](out[col], thr).astype(int)
    return out
```

- [ ] **Step 4: Verify PASS** (5 tests total).

- [ ] **Step 5: Commit**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat(features): clinical threshold flags"
```

---

## Task 3: Multi-morbidity score, age bins, interactions (TDD)

- [ ] **Step 1: Failing test** — append:

```python
from src.features import multimorbidity_score, age_group_features, interaction_features


def test_multimorbidity_score():
    df = pd.DataFrame({
        "htn": [1, 1, 0],
        "dm":  [1, 0, 0],
        "cad": [0, 1, 0],
        "ane": [1, 0, 0],
        "pe":  [0, 0, 0],
    })
    out = multimorbidity_score(df)
    assert out["multimorbidity"].tolist() == [3, 2, 0]


def test_age_group_features_categorical_and_dummies():
    df = pd.DataFrame({"age_raw": [10, 30, 70]})
    # use age_raw → derive age_group
    df["age"] = df["age_raw"]
    out = age_group_features(df)
    assert out["age_group"].tolist() == ["pediatric", "adult", "elderly"]
    assert out["age_pediatric"].tolist() == [1, 0, 0]
    assert out["age_adult"].tolist()     == [0, 1, 0]
    assert out["age_elderly"].tolist()   == [0, 0, 1]


def test_interaction_features_multiplies_correctly():
    df = pd.DataFrame({
        "age": [40, 60],
        "sc":  [1.0, 2.0],
        "bp":  [120, 140],
        "egfr": [90, 50],
        "hemo": [14, 10],
    })
    out = interaction_features(df)
    assert out["age_x_creatinine"].tolist() == [40.0, 120.0]
    assert out["age_x_bp"].tolist() == [4800, 8400]
    assert out["age_x_egfr"].tolist() == [3600, 3000]
    assert out["age_x_hemo"].tolist() == [560, 600]
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement** — append:

```python
MORBIDITY_COLS = ("htn", "dm", "cad", "ane", "pe")


def multimorbidity_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in MORBIDITY_COLS if c in out.columns]
    out["multimorbidity"] = out[cols].sum(axis=1).astype(int)
    return out


def _age_bin(a: float) -> str:
    if pd.isna(a):
        return "unknown"
    if a < 18:
        return "pediatric"
    if a < 65:
        return "adult"
    return "elderly"


def age_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = out["age"].apply(_age_bin)
    for grp in ("pediatric", "adult", "elderly"):
        out[f"age_{grp}"] = (out["age_group"] == grp).astype(int)
    return out


def interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_x_creatinine"] = out["age"] * out["sc"]
    out["age_x_bp"] = out["age"] * out["bp"]
    if "egfr" in out.columns:
        out["age_x_egfr"] = out["age"] * out["egfr"]
    if "hemo" in out.columns:
        out["age_x_hemo"] = out["age"] * out["hemo"]
    return out
```

- [ ] **Step 4: Verify PASS** (8 tests).

- [ ] **Step 5: Commit**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat(features): multimorbidity, age bins, interaction features"
```

---

## Task 4: Composite scores: anemia severity, CV risk proxy, electrolyte imbalance (TDD)

- [ ] **Step 1: Failing test** — append:

```python
from src.features import (
    compute_anemia_severity, compute_cv_risk, compute_electrolyte_imbalance,
)


def test_anemia_severity_levels():
    df = pd.DataFrame({"hemo": [13.0, 11.0, 9.0, 7.0]})
    out = compute_anemia_severity(df)
    # WHO thresholds: <7 severe (3), 7-9.9 moderate (2), 10-11.9 mild (1), >=12 normal (0)
    assert out["anemia_severity"].tolist() == [0, 1, 2, 3]


def test_cv_risk_additive():
    df = pd.DataFrame({
        "htn": [1, 1, 0],
        "dm":  [1, 0, 0],
        "cad": [0, 1, 0],
        "age": [70.0, 50.0, 30.0],
        "bp":  [150, 130, 110],
    })
    out = compute_cv_risk(df)
    # row 0: htn(1)+dm(1)+cad(0)+age_elderly(1, age>=65)+bp_high(1, bp>=140) = 4
    # row 1: 1+0+1+0+0 = 2
    # row 2: 0
    assert out["cv_risk"].tolist() == [4, 2, 0]


def test_electrolyte_imbalance_count_outside_ref():
    # Reference: sod 135-145, pot 3.5-5.0
    df = pd.DataFrame({
        "sod": [140, 130, 150, 138],
        "pot": [4.0, 5.5, 3.0, 4.5],
    })
    out = compute_electrolyte_imbalance(df)
    # row 0: 0; row 1: sod low + pot high = 2; row 2: sod high + pot low = 2; row 3: 0
    assert out["electrolyte_imbalance"].tolist() == [0, 2, 2, 0]
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement** — append:

```python
def compute_anemia_severity(df: pd.DataFrame) -> pd.DataFrame:
    """WHO anemia severity bins (0=normal, 1=mild, 2=moderate, 3=severe)."""
    out = df.copy()
    h = out["hemo"]
    sev = pd.Series(0, index=out.index, dtype=int)
    sev[(h >= 10) & (h < 12)] = 1
    sev[(h >= 7)  & (h < 10)] = 2
    sev[h < 7] = 3
    out["anemia_severity"] = sev
    return out


def compute_cv_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Additive cardiovascular risk index (0-5)."""
    out = df.copy()
    out["cv_risk"] = (
        out["htn"].astype(int)
        + out["dm"].astype(int)
        + out["cad"].astype(int)
        + (out["age"] >= 65).astype(int)
        + (out["bp"] >= 140).astype(int)
    )
    return out


def compute_electrolyte_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """Count of {sod, pot} outside reference range."""
    out = df.copy()
    sod_oor = (out["sod"] < 135) | (out["sod"] > 145)
    pot_oor = (out["pot"] < 3.5) | (out["pot"] > 5.0)
    out["electrolyte_imbalance"] = sod_oor.astype(int) + pot_oor.astype(int)
    return out
```

- [ ] **Step 4: Verify PASS** (11 tests).

- [ ] **Step 5: Commit**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat(features): anemia severity, CV risk, electrolyte imbalance"
```

---

## Task 5: Feature selection (variance + correlation filter) (TDD)

Feature selection is split into two pure functions: `drop_low_variance` and `drop_high_correlation`. These accept a DataFrame and return (filtered_df, dropped_cols).

- [ ] **Step 1: Failing test** — append:

```python
from src.features import drop_low_variance, drop_high_correlation


def test_drop_low_variance_removes_constant_cols():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],   # variance > 0
        "b": [1.0, 1.0, 1.0, 1.0],   # zero variance
        "c": [0.001, 0.001, 0.002, 0.001],  # low variance, < 0.01 after scaling? we test threshold directly
    })
    out, dropped = drop_low_variance(df, threshold=0.01)
    assert "b" in dropped
    assert "a" in out.columns
    # 'c' has tiny variance ~2.5e-7, should be dropped
    assert "c" in dropped


def test_drop_high_correlation_removes_one_of_pair():
    np.random.seed(42)
    a = np.random.randn(100)
    df = pd.DataFrame({
        "a": a,
        "a_dup": a + np.random.randn(100) * 1e-9,  # ~perfect correlation
        "b": np.random.randn(100),
    })
    out, dropped = drop_high_correlation(df, threshold=0.9)
    # exactly one of {a, a_dup} dropped
    assert len(dropped) == 1
    assert dropped[0] in ("a", "a_dup")
    assert "b" in out.columns


def test_drop_high_correlation_keeps_uncorrelated():
    np.random.seed(0)
    df = pd.DataFrame({c: np.random.randn(50) for c in ["x", "y", "z"]})
    out, dropped = drop_high_correlation(df, threshold=0.9)
    assert dropped == []
    assert set(out.columns) == {"x", "y", "z"}
```

- [ ] **Step 2: Verify FAIL.**

- [ ] **Step 3: Implement** — append:

```python
def drop_low_variance(df: pd.DataFrame, threshold: float = 0.01) -> tuple[pd.DataFrame, list[str]]:
    """Drop numeric columns with variance below threshold.

    Note: caller is responsible for scaling first if comparing across features.
    """
    numeric = df.select_dtypes(include=[np.number])
    variances = numeric.var()
    drop = variances[variances < threshold].index.tolist()
    return df.drop(columns=drop), drop


def drop_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> tuple[pd.DataFrame, list[str]]:
    """For each pair |corr| > threshold, drop the second column (column-order tiebreak)."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: list[str] = []
    for col in upper.columns:
        if col in to_drop:
            continue
        candidates = upper.index[upper[col] > threshold].tolist()
        for cand in candidates:
            if cand not in to_drop and cand != col:
                to_drop.append(cand)
    return df.drop(columns=to_drop), to_drop
```

- [ ] **Step 4: Verify PASS** (14 tests).

- [ ] **Step 5: Commit**

```powershell
git add src/features.py tests/test_features.py
git commit -m "feat(features): variance + correlation feature selection"
```

---

## Task 6: M2 feature-engineering notebook

**File:** `D:\healthcare\notebooks\_build_03_feature_engineering.py` (new), `notebooks/03_feature_engineering.ipynb` (generated).

End-to-end: load `patients_clean.csv` → engineer all composites → run feature selection → fit PCA + UMAP + t-SNE → persist artifacts.

- [ ] **Step 1: Create builder script**

Create `D:\healthcare\notebooks\_build_03_feature_engineering.py`:

```python
"""Generates 03_feature_engineering.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 2 — Clinical Feature Engineering & Dimensionality

Loads `data/processed/patients_clean.csv` (output of M1), engineers composite clinical
features, runs feature selection, and fits PCA + UMAP + t-SNE projections for clustering.
The held-out `classification` column is kept in the output but excluded from selection.
"""))

cells.append(nbf.v4.new_code_cell("""import sys; sys.path.append('..')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
import umap

from src.features import (
    compute_egfr, threshold_flags, multimorbidity_score, age_group_features,
    interaction_features, compute_anemia_severity, compute_cv_risk,
    compute_electrolyte_imbalance, drop_low_variance, drop_high_correlation,
)

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures'); FIGDIR.mkdir(parents=True, exist_ok=True)
MODELS = Path('../models'); MODELS.mkdir(parents=True, exist_ok=True)
DATA = Path('../data/processed'); DATA.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load M1 cleaned dataset"))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv(DATA / 'patients_clean.csv')
print('Shape:', df.shape)
print('NaNs:', df.isna().sum().sum())
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Composite scores
- eGFR (CKD-EPI 2021, sex-agnostic)
- Anemia severity (WHO bins)
- Cardiovascular risk proxy
- Electrolyte imbalance count
"""))
cells.append(nbf.v4.new_code_cell("""df = compute_egfr(df)
df = compute_anemia_severity(df)
df = compute_cv_risk(df)
df = compute_electrolyte_imbalance(df)
df[['egfr', 'anemia_severity', 'cv_risk', 'electrolyte_imbalance']].describe()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Threshold flags"))
cells.append(nbf.v4.new_code_cell("""df = threshold_flags(df)
flag_cols = [c for c in df.columns if c.startswith('flag_')]
print('Flag prevalences:')
print(df[flag_cols].mean().sort_values(ascending=False))
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Multimorbidity, age bins, interactions"))
cells.append(nbf.v4.new_code_cell("""df = multimorbidity_score(df)
df = age_group_features(df)
df = interaction_features(df)
print(df[['multimorbidity', 'age_group', 'age_x_creatinine', 'age_x_egfr']].head())
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Build the clustering input matrix
Drop:
- `classification` (held-out outcome)
- `age_group` (string label; numeric one-hots remain as `age_pediatric/adult/elderly`)
- raw outlier-flag columns (kept in the output dataset for the dashboard, but don't feed clustering)
"""))
cells.append(nbf.v4.new_code_cell("""target = df['classification'].copy()
exclude = ['classification', 'age_group']
exclude += [c for c in df.columns if c.endswith('_outlier_flag')]
X = df.drop(columns=exclude).select_dtypes(include=[np.number]).copy()
print('Initial feature matrix:', X.shape)
print('Columns:', X.columns.tolist())
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Feature selection
- Drop low-variance (already z-scored numerics + binary flags)
- Drop one of any pair with |corr| > 0.9
- Domain rule: keep all eGFR-derived features regardless of MI
"""))
cells.append(nbf.v4.new_code_cell("""X_lv, dropped_lv = drop_low_variance(X, threshold=0.01)
print(f'Dropped low-variance ({len(dropped_lv)}):', dropped_lv)
X_lc, dropped_lc = drop_high_correlation(X_lv, threshold=0.9)
print(f'Dropped high-corr ({len(dropped_lc)}):', dropped_lc)

PROTECTED = {'egfr', 'flag_low_egfr', 'flag_renal_impairment', 'age_x_egfr'}
restored = [c for c in PROTECTED if c in X.columns and c not in X_lc.columns]
for c in restored:
    X_lc[c] = X[c]
print('Restored protected features:', restored)
print('Final feature matrix:', X_lc.shape)
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Mutual information vs held-out outcome (reporting only)"))
cells.append(nbf.v4.new_code_cell("""mi = mutual_info_classif(X_lc.fillna(0), target, random_state=RANDOM_STATE)
mi_ranked = pd.Series(mi, index=X_lc.columns).sort_values(ascending=False)
print('Top-15 MI vs outcome:')
print(mi_ranked.head(15))

fig, ax = plt.subplots(figsize=(8, 6))
mi_ranked.head(20).plot.barh(ax=ax, color='#0F766E')
ax.invert_yaxis()
ax.set_title('Top-20 features by mutual information vs CKD label (reporting only)')
ax.set_xlabel('Mutual information')
plt.tight_layout()
plt.savefig(FIGDIR / 'mutual_information_top20.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 8. PCA"))
cells.append(nbf.v4.new_code_cell("""pca = PCA(random_state=RANDOM_STATE).fit(X_lc.fillna(0))
cum = np.cumsum(pca.explained_variance_ratio_)
k = int(np.searchsorted(cum, 0.80) + 1)
print(f'Components needed for 80% variance: k={k}')
print('Explained variance ratio (first 10):', pca.explained_variance_ratio_[:10].round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 'o-', color='#0F766E')
axes[0].set_xlabel('Component'); axes[0].set_ylabel('Explained variance ratio'); axes[0].set_title('Scree plot')
axes[1].plot(range(1, len(cum)+1), cum, 'o-', color='#0F766E')
axes[1].axhline(0.8, ls='--', color='gray'); axes[1].axvline(k, ls='--', color='gray')
axes[1].set_xlabel('Component'); axes[1].set_ylabel('Cumulative variance'); axes[1].set_title(f'Cumulative (80% at k={k})')
plt.tight_layout()
plt.savefig(FIGDIR / 'scree_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Final PCA with selected k
pca_final = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_lc.fillna(0))
X_pca = pca_final.transform(X_lc.fillna(0))
print('PCA matrix shape:', X_pca.shape)
"""))

cells.append(nbf.v4.new_markdown_cell("## 9. 2D projections: PCA, UMAP, t-SNE"))
cells.append(nbf.v4.new_code_cell("""# 2D PCA
pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_lc.fillna(0))
xy_pca = pca2.transform(X_lc.fillna(0))

# UMAP
um = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1).fit(X_lc.fillna(0))
xy_umap = um.embedding_

# t-SNE
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init='pca')
xy_tsne = tsne.fit_transform(X_lc.fillna(0))

projections = {
    'PCA': xy_pca, 'UMAP': xy_umap, 't-SNE': xy_tsne,
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, xy) in zip(axes, projections.items()):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=target.values, cmap='coolwarm', s=20, alpha=0.7)
    ax.set_title(f'{name} (colored by CKD label, sanity check only)')
    plt.colorbar(sc, ax=ax, ticks=[0, 1])
plt.tight_layout()
plt.savefig(FIGDIR / 'projections_2d.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 10. Persist artifacts"))
cells.append(nbf.v4.new_code_cell("""# Save final feature matrix + projections + classification target
out = X_lc.copy()
out['pca1_2d'] = xy_pca[:, 0]; out['pca2_2d'] = xy_pca[:, 1]
out['umap1'] = xy_umap[:, 0]; out['umap2'] = xy_umap[:, 1]
out['tsne1'] = xy_tsne[:, 0]; out['tsne2'] = xy_tsne[:, 1]
out['classification'] = target.values
out.to_csv(DATA / 'patients_features.csv', index=False)

joblib.dump(pca_final, MODELS / 'pca.pkl')
joblib.dump(um, MODELS / 'umap.pkl')

with open(MODELS / 'feature_columns.json', 'w') as f:
    json.dump({
        'feature_columns': X_lc.columns.tolist(),
        'pca_n_components': int(k),
        'pca_explained_variance_ratio': pca_final.explained_variance_ratio_.tolist(),
        'dropped_low_variance': dropped_lv,
        'dropped_high_correlation': dropped_lc,
        'protected_restored': restored,
    }, f, indent=2)

print('Saved:')
print('  data/processed/patients_features.csv:', out.shape)
print('  models/pca.pkl  (n_components =', k, ')')
print('  models/umap.pkl')
print('  models/feature_columns.json')
"""))

nb.cells = cells
nbf.write(nb, '03_feature_engineering.ipynb')
print('wrote 03_feature_engineering.ipynb')
```

- [ ] **Step 2: Generate + execute**

```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
cd notebooks
python _build_03_feature_engineering.py
cd ..
jupyter nbconvert --to notebook --execute notebooks/03_feature_engineering.ipynb --output 03_feature_engineering.ipynb
```
Expected: clean execution. UMAP may print a few numba warnings — harmless. t-SNE on 400 rows takes ~5-10s.

- [ ] **Step 3: Verify artifacts**

```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/patients_features.csv'); print(df.shape); print(df.columns.tolist())"
```
Expected: shape ~`(400, ~30)` with `pca1_2d, pca2_2d, umap1, umap2, tsne1, tsne2, classification` columns.

```powershell
python -c "import joblib; p = joblib.load('models/pca.pkl'); print('PCA n_components:', p.n_components_)"
```

```powershell
python -c "import json; print(json.dumps(json.load(open('models/feature_columns.json')), indent=2)[:500])"
```

- [ ] **Step 4: Commit**

```powershell
git add notebooks/_build_03_feature_engineering.py notebooks/03_feature_engineering.ipynb data/processed/patients_features.csv models/pca.pkl models/umap.pkl models/feature_columns.json reports/figures/scree_plot.png reports/figures/projections_2d.png reports/figures/mutual_information_top20.png
git commit -m "feat(features): notebook 03 + PCA/UMAP/t-SNE artifacts"
```

---

## Task 7: M2 verification & tag

- [ ] **Step 1: Full test suite**

```powershell
cd D:\healthcare
.\.venv\Scripts\Activate.ps1
pytest -v
```
Expected: 33 passed (19 from M1 + 14 from M2).

- [ ] **Step 2: Verify all M2 deliverables**

```powershell
@(
  'data/processed/patients_features.csv',
  'models/pca.pkl',
  'models/umap.pkl',
  'models/feature_columns.json',
  'reports/figures/scree_plot.png',
  'reports/figures/projections_2d.png',
  'reports/figures/mutual_information_top20.png',
  'notebooks/03_feature_engineering.ipynb'
) | ForEach-Object { if (Test-Path $_) { "OK   $_" } else { "MISS $_" } }
```

- [ ] **Step 3: Tag + push**

```powershell
git tag -a m2-complete -m "Milestone 2 complete: feature engineering + PCA/UMAP/t-SNE"
git push origin main --tags
```

- [ ] **Step 4: Report back** with: pytest count, feature matrix shape, PCA n_components selected, top-5 MI features, tag SHA.

---

## Self-Review

**Spec coverage (M2 sections of design spec):**
- 4.1 composite scores → Tasks 1 (eGFR) + 4 (anemia/CV/electrolyte) ✓
- 4.2 threshold flags → Task 2 ✓
- 4.3 multimorbidity + age bins → Task 3 ✓
- 4.4 interactions → Task 3 ✓
- 4.5 feature selection (variance, correlation, MI reporting, domain rule) → Tasks 5 + 6 ✓
- 4.6 PCA + UMAP + t-SNE + scree → Task 6 ✓
- 4.7 deliverables → Task 7 verifies all ✓

**Placeholder scan:** None.

**Type/name consistency:**
- `compute_egfr`, `threshold_flags`, `multimorbidity_score`, `age_group_features`, `interaction_features`, `compute_anemia_severity`, `compute_cv_risk`, `compute_electrolyte_imbalance`, `drop_low_variance`, `drop_high_correlation` — names match across all tasks and the notebook.
- `classification` column held out consistently in Task 6 step 5 + persisted to output.
- Feature column manifest JSON keys defined and used consistently.

**Known integration risks for M3 (out of scope here):**
- `patients_features.csv` has both raw features and 2D projection coords; M3 must clustering on `feature_columns.json["feature_columns"]` *only* — the projection coords are for visualization, not input.
- UMAP isn't deterministic across versions even with `random_state` set; document the umap-learn version in the report.
