"""Generates 06_outcome_validation.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 3 — Outcome Validation

Strict-unsupervised honesty check: do discovered clusters align with the held-out CKD label?
Chi-squared independence test + cluster purity score.
"""))

cells.append(nbf.v4.new_code_cell("""import sys; sys.path.append('..')
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures')
DATA = Path('../data/processed')
MODELS = Path('../models')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load clustered dataset"))
cells.append(nbf.v4.new_code_cell("""df = pd.read_csv(DATA / 'patients_clustered.csv')
print(df[['cluster_id', 'cluster_name', 'risk_tier', 'classification']].head())
print('Total:', len(df))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Contingency table + chi-squared"))
cells.append(nbf.v4.new_code_cell("""ct = pd.crosstab(df['cluster_id'], df['classification'])
print(ct)
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f'Chi-squared = {chi2:.2f}, dof = {dof}, p = {p:.4g}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Purity score"))
cells.append(nbf.v4.new_code_cell("""purity = ct.max(axis=1).sum() / ct.values.sum()
print(f'Cluster purity vs CKD label: {purity:.3f}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Cluster x CKD heatmap"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(7, 5))
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
sns.heatmap(ct_pct, annot=True, fmt='.0f', cmap='coolwarm', ax=ax, cbar_kws={'label': '% within cluster'})
ax.set_xticklabels(['notckd (0)', 'ckd (1)'])
ax.set_title(f'Cluster vs CKD label (chi2={chi2:.1f}, p={p:.2g}, purity={purity:.2f})')
plt.tight_layout()
plt.savefig(FIGDIR / 'cluster_vs_ckd.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Save validation summary"))
cells.append(nbf.v4.new_code_cell("""validation = {
    'chi2': float(chi2),
    'dof': int(dof),
    'p_value': float(p),
    'purity': float(purity),
    'contingency': ct.to_dict(),
}
with open(MODELS / 'outcome_validation.json', 'w') as f:
    json.dump(validation, f, indent=2, default=str)
print('Saved outcome_validation.json')
"""))

nb.cells = cells
nbf.write(nb, '06_outcome_validation.ipynb')
print('wrote 06_outcome_validation.ipynb')
