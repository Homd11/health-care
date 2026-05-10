"""Generates 05_cluster_profiles.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 3 — Cluster Profiles, Names, Risk Tiers

Build clinical profiles per cluster, compute ANOVA significance, assign clinically meaningful
names, and categorize Low/Medium/High risk. Persist `models/cluster_profiles.json` and
`data/processed/patients_clustered.csv`.

Uses the clinical core matrix (egfr, sc, hemo, bgr, multimorbidity, anemia_severity, cv_risk,
age, bp) — same as notebook 04.
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
from scipy import stats

from src.data_loader import load_ckd
from src.preprocessing import clean_types, ClinicalImputer, encode_binary
from src.features import (
    compute_egfr, multimorbidity_score, compute_anemia_severity, compute_cv_risk,
)
from src.profiles import compute_cluster_profiles, name_clusters, assign_risk_tiers, persist_profiles

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures')
MODELS = Path('../models')
DATA = Path('../data/processed')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Rebuild the clinical core matrix + load models"))
cells.append(nbf.v4.new_code_cell("""raw = clean_types(load_ckd('../data/raw/kidney_disease.csv'))
imputer = ClinicalImputer().fit(raw)
raw = imputer.transform(raw)
raw = encode_binary(raw)
raw = compute_egfr(raw)
raw = multimorbidity_score(raw)
raw = compute_anemia_severity(raw)
raw = compute_cv_risk(raw)

with open(MODELS / 'core_features.json') as f:
    cfg = json.load(f)
CORE_FEATURES = cfg['core_features']

core_raw = raw[CORE_FEATURES].copy()
core_scaler = joblib.load(MODELS / 'core_scaler.pkl')
X = pd.DataFrame(core_scaler.transform(core_raw), columns=CORE_FEATURES)

km = joblib.load(MODELS / 'kmeans.pkl')
gmm = joblib.load(MODELS / 'gmm.pkl')
labels = km.predict(X)
proba = gmm.predict_proba(X)
confidence = proba.max(axis=1)
print('Cluster sizes:', dict(zip(*np.unique(labels, return_counts=True))))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Profiles + delta vs population (using raw clinical values)"))
cells.append(nbf.v4.new_code_cell("""# Use unscaled clinical values for interpretable means
profiles = compute_cluster_profiles(core_raw, labels)
print('Per-cluster sizes + key clinical means:')
for cid, p in profiles.items():
    fm = p['feature_means']
    print(f"  cluster {cid} (n={p['size']}): egfr={fm['egfr']:.1f}, "
          f"sc={fm['sc']:.2f}, hemo={fm['hemo']:.1f}, multimorb={fm['multimorbidity']:.2f}, "
          f"anemia_sev={fm['anemia_severity']:.2f}, cv_risk={fm['cv_risk']:.2f}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. ANOVA across clusters"))
cells.append(nbf.v4.new_code_cell("""anova_results = {}
for col in core_raw.columns:
    valid_groups = [core_raw.loc[labels == cid, col].values
                    for cid in sorted(set(labels))
                    if (labels == cid).sum() >= 2]
    if len(valid_groups) >= 2:
        f, pv = stats.f_oneway(*valid_groups)
        anova_results[col] = {'F': float(f), 'p': float(pv)}
anova_df = pd.DataFrame(anova_results).T.sort_values('p')
alpha = 0.05 / len(anova_df) if len(anova_df) else 0.05
print(f'Bonferroni-corrected alpha: {alpha:.4f}')
print('ANOVA across clusters (sorted by p):')
print(anova_df)
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Names + risk tiers"))
cells.append(nbf.v4.new_code_cell("""named = name_clusters(profiles)
tiered = assign_risk_tiers(named)

for cid in tiered:
    mask = labels == cid
    tiered[cid]['gmm_proba_mean'] = float(confidence[mask].mean())

print('Final cluster summary:')
for cid, p in tiered.items():
    print(f"  {cid}: {p['name']:35s} | risk={p['risk_tier']:6s} | "
          f"n={p['size']:3d} | conf={p['gmm_proba_mean']:.3f}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Radar chart per cluster"))
cells.append(nbf.v4.new_code_cell("""features_for_radar = ['egfr', 'sc', 'hemo', 'bgr', 'multimorbidity', 'anemia_severity', 'cv_risk']
mat = np.array([[tiered[cid]['feature_means'].get(f, 0.0) for f in features_for_radar]
                for cid in sorted(tiered)])
mn, mx = mat.min(0), mat.max(0)
norm = np.where(mx > mn, (mat - mn) / (mx - mn + 1e-9), 0.5)

theta = np.linspace(0, 2 * np.pi, len(features_for_radar), endpoint=False)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
risk_colors = {'Low': '#0F766E', 'Medium': '#F59E0B', 'High': '#DC2626'}
for i, cid in enumerate(sorted(tiered)):
    vals = np.concatenate([norm[i], [norm[i, 0]]])
    angles = np.concatenate([theta, [theta[0]]])
    color = risk_colors.get(tiered[cid]['risk_tier'], '#6B7280')
    ax.plot(angles, vals, label=f"{cid}: {tiered[cid]['name']} ({tiered[cid]['risk_tier']})", color=color, linewidth=2)
    ax.fill(angles, vals, alpha=0.15, color=color)
ax.set_xticks(theta); ax.set_xticklabels(features_for_radar)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_title('Cluster radar (min-max normalized clinical features)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(FIGDIR / 'cluster_radar.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Persist artifacts"))
cells.append(nbf.v4.new_code_cell("""persist_profiles(tiered, MODELS / 'cluster_profiles.json')

# Build patients_clustered.csv: M2 features + 2D projections + cluster info
df_features = pd.read_csv(DATA / 'patients_features.csv')
out = df_features.copy()
out['cluster_id'] = labels
out['gmm_confidence'] = confidence
out['cluster_name'] = pd.Series(labels).map(lambda c: tiered[c]['name'])
out['risk_tier'] = pd.Series(labels).map(lambda c: tiered[c]['risk_tier'])
# Append raw clinical values used for profiling (so dashboard can show them)
for col in CORE_FEATURES:
    out[f'raw_{col}'] = core_raw[col].values

out.to_csv(DATA / 'patients_clustered.csv', index=False)
print('Saved cluster_profiles.json + patients_clustered.csv:', out.shape)
print('Risk tier distribution:')
print(out['risk_tier'].value_counts())
"""))

nb.cells = cells
nbf.write(nb, '05_cluster_profiles.ipynb')
print('wrote 05_cluster_profiles.ipynb')
