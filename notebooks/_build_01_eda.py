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
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
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
