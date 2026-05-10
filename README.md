# Healthcare Patient Clustering & Medical Risk Grouping

Unsupervised clustering of clinical data (UCI CKD) with a Streamlit decision-support dashboard.
AIE323 — Data Mining, Alamein University.

## 🚀 Live Demo

**https://health-care21.streamlit.app/**

Five interactive pages:
1. **Population Overview** — KPIs, cluster sizes, risk-tier donut
2. **Cluster Explorer** — PCA / UMAP / t-SNE projection + per-cluster radar
3. **Patient Risk Lookup** — score by index or fill a new-patient form
4. **Feature Distributions** — violin/box plots per cluster + ANOVA
5. **Batch Risk Scoring** — upload a CSV, download scored results

> ⚠ Research / decision-support only. Not for clinical diagnosis.

## Discovered Patient Subgroups

| Cluster | Name | Risk Tier | Patients | Mean GMM Confidence |
|---|---|---|---|---|
| 0 | Moderate Renal Risk | Medium | 120 | 0.996 |
| 1 | Stable / Low-Risk | Low | 208 | 0.990 |
| 2 | Severe Renal Impairment | High | 72 | 0.992 |

**Validation vs held-out CKD label:** χ² = 221.5, p = 7.8×10⁻⁴⁹, purity = 0.855 (strict-unsupervised).

## Setup (Windows / PowerShell)

```powershell
cd D:\healthcare
py -3.11 -m venv .venv     # or `python -m venv .venv` if `python` is on PATH
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # full dev env (notebooks + EDA)
# or `pip install -r requirements.txt` for the slim dashboard-only env
```

## Run

- Notebooks: `jupyter lab notebooks/`
- Dashboard: `streamlit run app/streamlit_app.py`

See `docs/specs/` for the design spec and `docs/plans/` for milestone plans.
