# Healthcare Patient Clustering & Medical Risk Grouping — Design Spec

- **Date:** 2026-05-10
- **Project root:** `D:\healthcare`
- **Course:** AIE323 — Data Mining (Alamein University)
- **Scope:** All five project milestones (Data → EDA → Feature Engineering → Clustering → Dashboard → Final Report)
- **Execution model:** One design document; staged implementation (M1 → M2 → M3 → M4 → M5) with verification between stages.

## 1. Goals & Constraints

Build an unsupervised clustering pipeline over real clinical data that discovers clinically meaningful patient subgroups, validates them statistically, and exposes the results through a deployed multi-page Streamlit clinical decision-support dashboard. Reproduce all outputs from raw data with a single command.

**Hard requirements (from rubric):**
- Coverage of at least three feature categories (labs, vitals, demographics, diagnoses).
- Medically appropriate missing-value handling and outlier treatment.
- At least four clustering algorithms compared (KMeans, Agglomerative, DBSCAN, GMM).
- Standard clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz).
- Clinically named clusters with risk tiers (Low / Medium / High).
- Five-page Streamlit/Dash dashboard with mandatory disclaimer and confidence scores.
- Cloud deployment.
- Final report with ethics + recommendations.

**Locked-in choices:**
- **Dataset:** UCI Chronic Kidney Disease (CKD), ~400 rows, 25 columns. Hits all four feature categories and includes a held-out outcome label (`classification`).
- **Approach to outcome label:** strict unsupervised — drop before clustering, use only for post-hoc validation (chi-squared, purity).
- **Framework:** Streamlit (with custom theme + Plotly + AgGrid for a clinical look).
- **Project structure:** Hybrid — notebooks for analysis narrative, `src/` modules for reusable logic, `app/` for the dashboard.
- **Architecture:** Approach 1 — notebooks save artifacts (pickles + JSON + CSV); dashboard loads them via `@st.cache_resource`. A single `src/inference.py::predict_patient()` glues the pipeline for new-patient prediction.
- **Deployment:** Streamlit Community Cloud, deployed from a GitHub repo (user has an account).
- **Final report:** Markdown source → PDF via pandoc.
- **Reproducibility:** `random_state=42` throughout; pipeline rebuildable end-to-end.

## 2. Project Layout

```
D:\healthcare\
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml
├── data/
│   ├── raw/kidney_disease.csv               # gitignored
│   ├── interim/patients_imputed.csv
│   └── processed/
│       ├── patients_clean.csv               # M1 output
│       ├── patients_features.csv            # M2 output
│       └── patients_clustered.csv           # M3 output
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_cluster_profiles.ipynb
│   └── 06_outcome_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── profiles.py
│   ├── inference.py
│   └── viz.py
├── models/                          # committed (small pickles) so dashboard works without re-running pipeline
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── kmeans.pkl
│   ├── gmm.pkl
│   ├── cluster_profiles.json
│   └── feature_columns.json
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_Population_Overview.py
│   │   ├── 2_Cluster_Explorer.py
│   │   ├── 3_Patient_Risk_Lookup.py
│   │   ├── 4_Feature_Distributions.py
│   │   └── 5_Batch_Risk_Scoring.py
│   ├── components.py
│   └── style.css
├── reports/
│   ├── final_report.md
│   ├── final_report.pdf
│   └── figures/
├── docs/specs/
│   └── 2026-05-10-healthcare-clustering-design.md
└── run_pipeline.py            # papermill-based runner: 01 → 06
```

## 3. Milestone 1 — Data, EDA, Medical Preprocessing

### 3.1 Data acquisition (`src/data_loader.py`)
- `load_ckd(path="data/raw/kidney_disease.csv") -> pd.DataFrame`
- Auto-fetch via `ucimlrepo` if file missing; fall back to a clear error pointing to the UCI URL.
- Schema validation: expected columns + dtypes; raises `ValueError` on mismatch.

### 3.2 Feature inventory
| Category | Columns |
|---|---|
| Labs | `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`, `al`, `su`, `sg` |
| Vitals | `bp` |
| Demographics | `age` |
| Diagnoses / symptoms | `htn`, `dm`, `cad`, `appet`, `pe`, `ane`, `rbc`, `pc`, `pcc`, `ba` |
| Outcome (held out) | `classification` |

**Known gap:** CKD has no `sex` column. EDA stratification falls back to `age_group` × `htn`. BMI is also unavailable (no height/weight); compensated in M2 by interaction features that don't depend on BMI.

### 3.3 EDA notebook `01_eda.ipynb`
- Univariate: histograms + violin plots of all 12 lab/vital features, stratified by age bin.
- Missingness: `missingno` matrix + bar; correlation of missingness across columns to assess MCAR vs MAR.
- Comorbidity network: NetworkX graph of co-occurrence among `{htn, dm, cad, ane, pe}`, exported as static PNG (`reports/figures/comorbidity_network.png`) and interactive HTML via `pyvis`.
- Class balance bar chart for held-out `classification` (context only).
- Numeric correlation heatmap.

### 3.4 Preprocessing notebook `02_preprocessing.ipynb` + `src/preprocessing.py`
- **Type cleanup:** strip whitespace, lowercase strings, coerce numerics; the raw CSV contains `"\t?"` and stray tabs.
- **Missing values:**
  - Numeric labs → median by `htn` group (preserves disease-related distribution shifts).
  - Categorical → mode.
  - Persist fitted imputer to `models/imputer.pkl`.
- **Outliers:** IQR-based detection; **flag, do not remove**. Add `<feature>_outlier_flag` columns. Rationale documented in the notebook: extreme lab values are real critical findings.
- **Encoding:** binary categoricals (`yes/no`, `present/notpresent`, `good/poor`, `normal/abnormal`) → 0/1.
- **Scaling:** `StandardScaler` on continuous features; persist to `models/scaler.pkl`.
- Outputs: `data/interim/patients_imputed.csv`, `data/processed/patients_clean.csv`.

### 3.5 Deliverables
- Rendered `01_eda.ipynb` (Medical EDA Report).
- `reports/figures/comorbidity_network.png` + `.html`.
- `data/processed/patients_clean.csv`.
- `reports/figures/m1_eda_summary.md` for inclusion in the final report.

## 4. Milestone 2 — Clinical Feature Engineering & Dimensionality

### 4.1 Composite scores (`src/features.py`)
- **eGFR** — CKD-EPI 2021 formula on creatinine + age. Sex-agnostic average coefficient (no sex column); limitation documented. Output column `egfr` (mL/min/1.73m²).
- **Anemia severity** — derived from `hemo` per WHO thresholds: `0=normal, 1=mild, 2=moderate, 3=severe`.
- **Cardiovascular risk proxy** — additive index 0–5: `htn + dm + cad + age_elderly + bp_high`.
- **Electrolyte imbalance score** — count of {sod, pot} outside reference range.

### 4.2 Threshold flags (binary)
`flag_hyperglycemia` (bgr ≥ 200), `flag_hypertensive` (bp ≥ 140), `flag_anemia` (hemo < 12), `flag_hyperkalemia` (pot > 5.0), `flag_hyponatremia` (sod < 135), `flag_renal_impairment` (sc > 1.3), `flag_proteinuria` (al ≥ 2), `flag_low_egfr` (egfr < 60).

### 4.3 Multi-morbidity & age binning
- `multimorbidity` = sum of `{htn, dm, cad, ane, pe}` (0–5).
- `age_group` ∈ `{pediatric (<18), adult (18–64), elderly (≥65)}`, one-hot encoded.

### 4.4 Interaction features
- `age_x_creatinine`, `age_x_bp`, `age_x_egfr`, `age_x_hemo` (BMI interactions skipped — no BMI available).

### 4.5 Feature selection
1. Variance threshold: drop features with var < 0.01 after scaling.
2. Correlation filter: for any pair with `|r| > 0.9`, drop one (e.g., `pcv` vs `hemo`); drops are logged.
3. Mutual information vs held-out outcome — **reporting only**, not for selection (preserves unsupervised purity).
4. Domain rule: keep all eGFR-derived and threshold-flag features regardless of MI.
- Persist final column order to `models/feature_columns.json`.

### 4.6 Dimensionality reduction
- **PCA** on standardized features. Scree + cumulative explained variance plots. Choose smallest *k* with cumulative ≥ 80%. Persist `models/pca.pkl`.
- **UMAP** (2D) and **t-SNE** (2D) for visualization only. Coordinates appended to dataset for the dashboard's Cluster Explorer.
- 2D PCA scatter colored by held-out `classification` as sanity check.

### 4.7 Deliverables
- Rendered `03_feature_engineering.ipynb` (Feature Engineering Report).
- Scree + 2D projection plots in `reports/figures/`.
- `data/processed/patients_features.csv`.
- `models/pca.pkl`, `models/feature_columns.json`.

## 5. Milestone 3 — Clustering & Clinical Validation

### 5.1 Algorithms (`src/clustering.py`)
All fit on PCA-reduced features. `random_state=42`.
- **K-Means**: elbow (inertia, k=2..10) + silhouette analysis → optimal *k*. `n_init=20`.
- **Agglomerative Hierarchical**: Ward / Complete / Average linkage; dendrograms saved; cluster count guided by dendrogram cut + silhouette.
- **DBSCAN**: k-distance plot for `eps`; `min_samples = 2 * n_features`. `label = -1` reported as anomalous patients.
- **GMM**: BIC across n_components=2..8; `covariance_type='full'`. Soft probabilities = dashboard confidence.

### 5.2 Evaluation (`src/evaluation.py`)
- Per-algorithm: silhouette, Davies-Bouldin, Calinski-Harabasz.
- **Bootstrap stability:** 50 resamples at 80% size; report mean + std Adjusted Rand Index between original and resample labelings.
- Comparison table → markdown for the report.
- **Final selection rule:** highest silhouette with stable ARI > 0.7. Expected: K-Means. Persist as `models/kmeans.pkl`. GMM persisted alongside (`models/gmm.pkl`) for confidence scores.

### 5.3 Clinical interpretation (`05_cluster_profiles.ipynb` + `src/profiles.py`)
- Mean/median per cluster for every clinical feature; deltas vs population.
- ANOVA (continuous) + chi-squared (binary flags) per feature; Bonferroni-corrected significance.
- **Cluster naming heuristic** (rule-based on feature deltas):
  - low eGFR + high creatinine + anemia → "Severe Renal Impairment"
  - high glucose + dm flag elevated → "Metabolic / Diabetic Risk"
  - mostly normal vs population → "Stable / Low-Risk"
  - other patterns named by dominant deltas
- **Risk tier assignment** (Low / Medium / High) — composite of:
  - mean eGFR percentile (lower → higher risk)
  - mean multi-morbidity score
  - mean anemia severity
  - thresholds documented in `src/profiles.py`.
- Persist `models/cluster_profiles.json`:
  ```json
  {
    "0": {"name": "...", "risk_tier": "High", "size": 87,
          "feature_means": {...}, "feature_deltas": {...},
          "gmm_proba_mean": 0.92}
  }
  ```

### 5.4 Outcome validation (`06_outcome_validation.ipynb`)
- Chi-squared: cluster vs held-out `classification`.
- Cluster purity score.
- Confusion-style heatmap of cluster × ckd/notckd.

### 5.5 Deliverables
- Clustering Comparison Report (markdown export of `04_clustering.ipynb`).
- Clinical Cluster Profiles (`models/cluster_profiles.json` + section in final report).
- All model pickles.
- Dendrograms, elbow, silhouette, BIC plots in `reports/figures/`.

## 6. Milestone 4 — Streamlit Clinical Dashboard

### 6.1 Theme & shared components
- `.streamlit/config.toml`: clinical palette (primary `#0F766E` deep teal, slate text on white, soft red `#DC2626` for high-risk).
- `app/style.css`: KPI card styling, risk badges (green/amber/red), banner.
- `app/components.py`:
  - `render_disclaimer()` — banner shown on every page: *"This tool is for research and decision-support only. It does not replace clinical judgment or diagnosis."*
  - `kpi_card(label, value, delta=None)`.
  - `risk_badge(tier)`.
- Sidebar: project title, dataset note, link to GitHub repo.

### 6.2 Pages

**Page 1 — Population Overview**
- KPI cards: Total Patients, # Clusters, % High-Risk, % CKD-positive (held-out validation context).
- Bar chart: cluster sizes colored by risk tier.
- Donut chart: risk-tier distribution.
- Data-quality card: imputation rate, # outliers flagged.

**Page 2 — Cluster Explorer**
- Projection toggle: PCA / UMAP / t-SNE.
- Plotly scatter colored by cluster; hover shows patient ID, risk tier, GMM confidence.
- Cluster selector → side panel: name, risk tier, size, **radar chart** of normalized feature means with population-baseline overlay; feature table with deltas + significance stars.

**Page 3 — Patient Risk Lookup**
- Mode A — *Existing patient*: select index → show cluster, risk, top-3 contributing features (distance-to-centroid decomposition).
- Mode B — *New patient*: form with all input features (sensible defaults, units shown). Submit calls `src/inference.py::predict_patient(dict) -> {cluster_id, cluster_name, risk_tier, confidence, summary, top_features}`. Pipeline: impute → engineer → scale → PCA → KMeans label + GMM probability.
- Confidence bar always rendered next to prediction.
- Plain-language summary generated from cluster profile JSON.

**Page 4 — Feature Distributions**
- Feature dropdown → side-by-side violin + box plots per cluster, with population overlay and ANOVA p-value annotation.

**Page 5 — Batch Risk Scoring**
- CSV upload (downloadable template). Schema validation with explicit per-column errors.
- Result table: `cluster_id, cluster_name, risk_tier, gmm_confidence, top_3_features` per row.
- Aggregate summary above table.
- Download results CSV button.

### 6.3 Caching
- `@st.cache_resource` for models, scaler, PCA, profiles JSON.
- `@st.cache_data` for the clustered patient dataframe.

### 6.4 Clinical safety
- Disclaimer on every page (mandatory).
- Confidence shown for every prediction (Pages 3 + 5).

## 7. Milestone 5 — Final Report & Documentation

### 7.1 `reports/final_report.md` outline
1. Executive summary
2. Dataset & domain background
3. Methodology (M1 → M4)
4. Clinical cluster interpretation (subsection per cluster + radar)
5. Model evaluation & comparison
6. Outcome validation (chi-squared vs ckd label)
7. Ethical considerations (HIPAA / GDPR, algorithmic bias, unsupervised-label limitations)
8. Recommendations (resource planning, screening, personalized care)
9. Limitations & future work
10. Appendix: feature dictionary, hyperparameters, dashboard screenshots

### 7.2 Generation
- `pandoc reports/final_report.md -o reports/final_report.pdf --pdf-engine=xelatex --toc`.
- Output committed.

### 7.3 README
- Project overview + screenshot.
- Setup: `python -m venv .venv`, activate, `pip install -r requirements.txt`.
- Pipeline: `python run_pipeline.py` (papermill executes notebooks 01→06).
- Local dashboard: `streamlit run app/streamlit_app.py`.
- Live deployment URL.
- Demo walkthrough notes.

## 8. Tech Stack & Reproducibility

**Python 3.11**. `requirements.txt`:
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
```

- Single random seed (42) propagated through KMeans, DBSCAN-pre-shuffles, GMM, train splits, bootstrap sampling.
- `run_pipeline.py` rebuilds every artifact end-to-end via papermill.

## 9. Deployment

- `git init`; `.gitignore` excludes `data/raw/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.
- **Models committed** (`models/*.pkl`) — small enough for plain Git. Same for `data/processed/*.csv` so the dashboard works without re-running the pipeline.
- User creates empty GitHub repo; we add remote, push.
- Streamlit Community Cloud → connect repo → main file `app/streamlit_app.py` → deploy. Live URL added to README.

## 10. Risks & Open Questions

- **CKD missing `sex` and BMI** — documented as a limitation in M1 EDA + Final Report; eGFR uses sex-agnostic coefficient; BMI replaced with `age × egfr` and `age × hemo` interactions.
- **Small dataset (~400 rows)** — bootstrap stability check guards against unstable clusters.
- **DBSCAN may produce only one cluster + noise on this dataset** — acceptable; reported honestly with the noise (anomalous patient) count.
- **Cluster naming is heuristic** — documented as such in the final report; supported by ANOVA p-values.

## 11. Out of Scope

- The bonus Tweet Sentiment Analysis project — separate brainstorm + spec after this one ships.
- Time-series / longitudinal analysis — CKD is cross-sectional.
- Federated learning, differential privacy — mentioned in Ethics section as future work only.
- Mobile-responsive dashboard — desktop only is acceptable per rubric.
