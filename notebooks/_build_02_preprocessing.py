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

# Persist scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Persist imputer (callable; used by dashboard for new-patient prediction)
from src.preprocessing import ClinicalImputer
imputer = ClinicalImputer().fit(df)  # fit on the post-clean_types frame
joblib.dump(imputer, '../models/imputer.pkl')

# Also persist raw fitted state for inspectability
with open('../models/imputer_state.json', 'w') as f:
    json.dump(fitted_imputer, f, indent=2, default=str)

print('Saved:')
print('  data/processed/patients_clean.csv:', df_scaled.shape)
print('  models/scaler.pkl')
print('  models/imputer.pkl')
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
