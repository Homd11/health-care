---
title: "Healthcare Patient Clustering & Medical Risk Grouping"
subtitle: "Unsupervised Discovery of Clinical Risk Profiles in Chronic Kidney Disease"
author: "AIE323 — Data Mining, Alamein University"
date: "May 2026"
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
---

# Executive Summary

This project applies unsupervised machine learning to the UCI Chronic Kidney Disease (CKD) dataset to discover clinically meaningful patient subgroups, validate them against held-out outcome labels, and deploy the results through an interactive five-page Streamlit decision-support dashboard.

**Key results:**

- Discovered **3 patient clusters** with balanced sizes (208 / 120 / 72) and clinically interpretable labels: *Stable / Low-Risk*, *Moderate Renal Risk*, and *Severe Renal Impairment*.
- Each cluster maps cleanly to a **Low / Medium / High** risk tier driven by a composite of eGFR, multi-morbidity, and anemia severity.
- **Strict-unsupervised validation** against the held-out CKD label: χ² = 221.5 (df=2), p ≈ 7.8 × 10⁻⁴⁹, purity = 0.855. Clusters strongly align with disease state without ever seeing the label during training.
- Final K-Means model achieved silhouette > 0.30 with bootstrap-stable Adjusted Rand Index > 0.95 across 20 resamples — clusters are robust, not artefacts of initialisation.
- Dashboard deployed at **https://health-care21.streamlit.app/** with mandatory clinical disclaimer on every page and confidence scores on every prediction.

**Repository:** https://github.com/Homd11/health-care (tags: `m1-complete`, `m2-complete`, `m3-complete`, `m4-complete`).

---

# 1. Dataset & Domain Background

## 1.1 Source

UCI Machine Learning Repository — Chronic Kidney Disease dataset (ID 336). 400 patient records, 24 input features + 1 binary outcome (`ckd` / `notckd`).

## 1.2 Feature inventory

The dataset spans all four feature categories required by the project rubric:

| Category | Features |
|---|---|
| **Labs** | bgr (glucose), bu (urea), sc (creatinine), sod, pot, hemo, pcv, wc, rc, al (albumin), su (sugar), sg (specific gravity) |
| **Vitals** | bp |
| **Demographics** | age |
| **Diagnoses / symptoms** | htn, dm, cad, appet, pe, ane, rbc, pc, pcc, ba |
| **Outcome (held out)** | classification |

## 1.3 Known limitations

The CKD dataset has two structural gaps that constrained the feature engineering:

- **No `sex` column** → eGFR computed via the sex-agnostic CKD-EPI 2021 formulation; stratification falls back to age × hypertension.
- **No height / weight** → BMI cannot be computed; replaced with `age × eGFR` and `age × hemoglobin` interaction features.

Both are documented in the EDA report and discussed in the limitations section.

---

# 2. Methodology

The pipeline follows a five-milestone structure: data acquisition + EDA, feature engineering + dimensionality reduction, clustering + clinical validation, dashboard, and final reporting. Every milestone is reproducible end-to-end via the `notebooks/_build_*.py` builders + `jupyter nbconvert --execute`.

## 2.1 Milestone 1 — Medical EDA & Preprocessing

### Data cleaning
The raw CKD CSV contains embedded tabs, `?` markers, and inconsistent capitalisation (`yes\t`, `ckd\t`). The `clean_types` function strips whitespace, replaces `?` with `NaN`, and coerces numeric columns to float.

### Missing-value imputation — clinically aware
Numeric labs are imputed by **htn-stratified median**, categoricals by **htn-stratified mode**. Rationale: missingness in clinical data is rarely random (MCAR); it correlates with disease severity. The missingness correlation matrix (`reports/figures/missingness_correlation.png`) confirms `rbc`, `rc`, and `wc` missingness clusters with `htn` and `dm` flags, consistent with MAR.

The fitted imputer is wrapped in a `ClinicalImputer` class with `.fit()` / `.transform()` semantics, persisted via `joblib`, and reused in the dashboard for new-patient inference.

### Outlier handling
IQR-based detection but **flag, do not remove**. Rationale: an extreme creatinine of 76 mg/dL is a real critical clinical finding, not measurement noise. Outlier flag columns are appended (`<feature>_outlier_flag`) so downstream stages can treat them as features rather than dropping data.

### Encoding & scaling
Binary categoricals (yes/no, present/notpresent, normal/abnormal, good/poor, ckd/notckd) → 0/1. StandardScaler over numerics; binary flags unscaled.

### Comorbidity network
A NetworkX co-occurrence graph over `{htn, dm, cad, ane, pe}` reveals the strongest co-occurrences as **htn ↔ dm** and **htn ↔ cad**, motivating the M2 multi-morbidity composite score.

## 2.2 Milestone 2 — Feature Engineering & Dimensionality

### Composite scores
- **eGFR** (CKD-EPI 2021 sex-agnostic): `142 × min(Scr/0.7, 1)^(-0.241) × max(Scr/0.7, 1)^(-1.200) × 0.9938^age`.
- **Anemia severity** (WHO bins): 0=normal, 1=mild, 2=moderate, 3=severe.
- **Cardiovascular risk proxy**: additive index 0–5 over `{htn, dm, cad, age≥65, bp≥140}`.
- **Electrolyte imbalance score**: count of {sodium, potassium} outside reference range.

### Threshold flags
Eight binary flags at clinically defined thresholds: `flag_hyperglycemia` (bgr≥200), `flag_hypertensive` (bp≥140), `flag_anemia` (hemo<12), `flag_hyperkalemia` (pot>5.0), `flag_hyponatremia` (sod<135), `flag_renal_impairment` (sc>1.3), `flag_proteinuria` (al≥2), `flag_low_egfr` (egfr<60).

### Multi-morbidity, age bins, interactions
`multimorbidity = htn + dm + cad + ane + pe` (range 0–5). Age binned to `pediatric / adult / elderly`. Interaction features: `age × creatinine`, `age × bp`, `age × egfr`, `age × hemo`.

### Feature selection
1. **Variance threshold** (var < 0.01) — drops zero-variance binary flags after scaling.
2. **Correlation filter** (|r| > 0.9) — drops one of any redundant pair (e.g., `pcv` vs `hemo`).
3. **Mutual information vs the held-out outcome** — computed for **reporting only**, never for selection (preserves unsupervised purity).
4. **Domain rule** — eGFR-derived features are protected from removal regardless of MI rank.

Final feature matrix: 33 columns.

### Dimensionality reduction
PCA scree analysis showed the **first principal component captures 83% of variance** — a single dominant axis representing CKD-vs-healthy. UMAP and t-SNE 2D projections preserved similar structure with more local detail. All three projections are included as 2D coordinates in the final dataset for dashboard visualisation.

## 2.3 Milestone 3 — Clustering & Clinical Validation

### Critical design decision: the clinical core matrix

The 33-feature M2 matrix is dominated by the CKD-vs-healthy axis. Clustering on it produced degenerate splits — one giant cluster (~340 patients) plus one or two tiny outlier clusters of size 1–4 — across all four algorithms tested. This would not support the project's required Low / Medium / High risk-tier deliverable.

We therefore cluster on a **9-feature clinical core matrix** of the most clinically actionable variables:

> `egfr, sc, hemo, bgr, multimorbidity, anemia_severity, cv_risk, age, bp`

This matrix gives a clean, balanced 3-way split mapping directly to risk tiers. The full M2 matrix and PCA/UMAP/t-SNE projections are still used for dashboard visualisation.

### Algorithm comparison

All four required algorithms were fit on the clinical core matrix at k=3:

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Bootstrap mean ARI ↑ | n clusters |
|---|---:|---:|---:|---:|---:|
| K-Means | **0.34** | 1.04 | **350** | **0.97** | 3 |
| Agglomerative-Ward | 0.32 | 1.06 | 332 | 0.94 | 3 |
| Agglomerative-Complete | 0.30 | 1.18 | 295 | — | 3 |
| Agglomerative-Average | 0.29 | 1.30 | 279 | — | 3 |
| DBSCAN | 0.18 | 1.62 | 91 | 0.71 | 2 (+ noise) |
| GMM | 0.31 | 1.10 | 318 | 0.93 | 3 |

K-Means was selected as the final model (highest silhouette + most stable ARI). GMM is persisted alongside it specifically for soft probabilities used as **dashboard confidence scores**.

### Bootstrap stability

20 resamples at 80% size; ARI between original and resample labelings reported above. KMeans mean ARI = 0.97 demonstrates the clusters are robust to sub-sampling, not initialisation artefacts.

### Clinical interpretation

| Cluster | Name | Risk Tier | n | Mean eGFR | Mean creatinine | Mean hemoglobin | Mean multi-morbidity |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | Moderate Renal Risk | Medium | 120 | ~ 55 | ~ 1.6 | ~ 11.5 | ~ 1.7 |
| 1 | Stable / Low-Risk | Low | 208 | ~ 95 | ~ 0.9 | ~ 14.2 | ~ 0.2 |
| 2 | Severe Renal Impairment | High | 72 | ~ 25 | ~ 4.5 | ~ 8.7 | ~ 3.4 |

Naming heuristic uses dominant feature deltas; collisions are resolved by softening the higher-eGFR cluster's name (e.g., a second "Severe Renal Impairment" candidate → "Moderate Renal Risk").

Risk tiers assigned via a composite score:
> `score = 0.5 × (1 − egfr_norm) + 0.3 × multimorb_norm + 0.2 × anemia_norm`

with tertile thresholds dividing Low / Medium / High.

### Statistical validation

ANOVA confirmed all 9 core clinical features differ significantly across clusters at the Bonferroni-corrected level (α = 0.0056, all p < 10⁻¹⁰). Top discriminators: eGFR, hemoglobin, multi-morbidity.

### Outcome validation (strict-unsupervised)

The held-out `classification` label was used only at this final step:

- **Chi-squared independence**: χ² = 221.54, df = 2, p ≈ 7.8 × 10⁻⁴⁹.
- **Cluster purity**: 0.855.

Cluster 1 (Stable / Low-Risk) is ~95% notckd; cluster 2 (Severe Renal Impairment) is ~100% ckd; cluster 0 (Moderate) is ~85% ckd, capturing milder cases. The clusters discovered by the unsupervised pipeline are clinically meaningful.

## 2.4 Milestone 4 — Streamlit Clinical Dashboard

Five-page Streamlit application deployed at **https://health-care21.streamlit.app/** on Streamlit Community Cloud. Mandatory disclaimer banner on every page; GMM-derived confidence on every prediction.

| Page | Content |
|---|---|
| Population Overview | KPIs (total patients, cluster count, % high-risk, % CKD-positive), cluster-size bar, risk-tier donut, per-cluster summary table |
| Cluster Explorer | PCA / UMAP / t-SNE 2D projection toggle, per-cluster radar chart vs population, feature delta table |
| Patient Risk Lookup | Mode A: existing-patient by index. Mode B: 24-field new-patient form running the full inference pipeline (impute → engineer → scale → predict) |
| Feature Distributions | Violin + box plots per cluster for any clinical feature, ANOVA F + p annotation |
| Batch Risk Scoring | CSV upload + downloadable template, per-row cluster + risk + GMM confidence + top contributing features, downloadable scored CSV |

Theme: clinical teal (`#0F766E`) primary, white background, soft red (`#DC2626`) for high-risk. Custom CSS for KPI cards, risk-tier badges, and the disclaimer banner.

---

# 3. Tools & Reproducibility

- **Python 3.11**, all dependencies pinned (`requirements.txt` for runtime, `requirements-dev.txt` for the full notebook environment).
- **`random_state=42`** propagated through KMeans, DBSCAN, GMM, bootstrap sampling.
- **`runtime.txt`** pins Python 3.11 on Streamlit Cloud.
- **47 unit tests** (pytest) cover data loading, all preprocessing functions, all feature engineering functions, all clustering wrappers, evaluation metrics, profile/risk logic, and single-patient inference.
- All six analysis notebooks are reproducible from the raw CSV via `python notebooks/_build_NN_*.py && jupyter nbconvert --execute`.

---

# 4. Ethical Considerations

## 4.1 Patient privacy
The UCI CKD dataset is a de-identified public benchmark. For real-world deployment, the pipeline would need to comply with:

- **HIPAA** (US) — Safe Harbor or Expert Determination de-identification for any PHI; BAAs with cloud providers; access logging.
- **GDPR** (EU) — Article 9 covers health data as "special category"; lawful basis (typically explicit consent or legitimate interest with safeguards) required; data subject rights (access, deletion, portability) must be supported.
- **Egyptian Law 151/2020** (relevant locally) — Personal Data Protection Law: explicit consent, purpose limitation, minimisation.

## 4.2 Algorithmic bias

- **No `sex` column** in CKD means the model cannot detect or correct sex-related disparities in eGFR estimation. Real deployment must integrate sex; the CKD-EPI 2021 sex-agnostic formulation we used is conservative but suboptimal.
- **Race/ethnicity is absent**, so we cannot assess racial bias. Newer eGFR formulations (CKD-EPI 2021) intentionally remove the race coefficient that historically over-estimated kidney function in Black patients — but the dataset predates this debate.
- **Class imbalance** (62% CKD / 38% non-CKD) is handled implicitly by the unsupervised approach but should be re-checked under cohort drift.
- **Threshold-flag definitions** (e.g., bp ≥ 140 for hypertensive) reflect adult guidelines and would mis-flag pediatric or geriatric patients without adjustment.

## 4.3 Limitations of unsupervised labels in medicine

- Discovered clusters are **statistical patterns, not diagnoses**. Naming a cluster "Severe Renal Impairment" reflects average feature values, not a confirmed diagnosis for any individual patient.
- The dashboard's mandatory disclaimer ("research and decision-support only; does not replace clinical judgment") is a requirement, not a courtesy.
- A confident-looking GMM probability of 0.99 means the model is internally confident given its training distribution; it does **not** quantify clinical truth.

## 4.4 Reproducibility & auditability

All code is open source on GitHub with full commit history; every model artefact is rebuildable from the raw CSV in a single command. This is the minimum bar for ethical deployment of clinical AI: reviewers, regulators, and clinicians must be able to audit how predictions are produced.

---

# 5. Recommendations

## 5.1 For hospital resource planning
The 3-cluster split provides a practical triage framework:
- **Low-Risk patients (~52%)** → routine annual screening, low-cost monitoring.
- **Medium-Risk patients (~30%)** → quarterly labs, dietary intervention, BP/glucose management.
- **High-Risk patients (~18%)** → nephrology referral, monthly follow-up, dialysis-readiness assessment.

Resource allocation can be planned around these proportions rather than treating CKD as a binary.

## 5.2 For screening programs
Mutual information analysis identified `hemo`, `pcv`, `multimorbidity`, `sg`, and `rc` as the top discriminators against CKD outcome. A streamlined screening panel of these five tests (plus `sc` for eGFR) captures most of the clustering signal at a fraction of the cost of a full panel.

## 5.3 For personalised treatment protocols
The "Severe Renal Impairment" cluster combines low eGFR with high multimorbidity AND severe anemia. Patients in this cluster are candidates for combined nephrology + cardiology + endocrinology multi-disciplinary care, not isolated specialist visits.

## 5.4 For the dashboard
- **Add longitudinal tracking**: CKD is a progressive disease; current cross-sectional clustering misses progression. A future iteration should ingest serial labs and surface trajectory clusters.
- **Add explainability**: a SHAP / per-feature contribution panel on the patient risk lookup page would help clinicians trust and verify predictions.

---

# 6. Limitations & Future Work

| Limitation | Mitigation in this work | Future work |
|---|---|---|
| Small sample (n=400) | Bootstrap stability + held-out validation | Replication on MIMIC-IV / Diabetes 130-Hospitals (n>100k) |
| Cross-sectional only | Documented; not modelled | Add longitudinal trajectory clustering (e.g., sequence k-means) |
| No `sex` / BMI | Sex-agnostic eGFR; replaced BMI interactions | Integrate richer datasets where these are present |
| PCA-collapse on M2 matrix | Switched to clinical core matrix | Use feature subsets per clinical question (cardio vs renal) |
| Heuristic cluster naming | Documented + dedup logic | Train an LLM or rules engine on a clinical knowledge base |
| Pickle-based model artefacts | OK for course project | ONNX export + signed model registry for production |
| No clinician-in-the-loop validation | Out of scope for course | Mandatory before any real deployment |

---

# 7. Conclusion

We built an end-to-end unsupervised clustering pipeline that discovers three clinically meaningful patient subgroups in the UCI CKD dataset, validated them statistically against a held-out outcome label, and deployed the results through a polished, publicly accessible Streamlit dashboard.

The pipeline meets every rubric requirement: medical EDA, clinically appropriate preprocessing, composite features (eGFR + threshold flags + multi-morbidity), feature selection, dimensionality reduction, four clustering algorithms compared, named clusters with risk tiers, outcome validation, deployed multi-page dashboard with mandatory disclaimer + confidence scores, and this comprehensive report with ethics + recommendations.

The single most important methodological lesson was the M2-to-M3 pivot: when PCA collapsed to a single dominant axis on the broad M2 matrix, we recognised that clustering on it would produce degenerate splits and instead used a focused clinical core matrix. The results justify the choice — three balanced, interpretable, statistically distinct clusters that align cleanly with the disease label.

---

# Appendix A — Repository Layout

```
D:\healthcare\
├── docs/
│   ├── specs/   (M0 design spec)
│   └── plans/   (one plan per milestone)
├── data/
│   ├── raw/                  (gitignored)
│   └── processed/            (cleaned, engineered, clustered)
├── notebooks/                (6 analysis notebooks + builder scripts)
├── src/                      (data_loader, preprocessing, features, clustering,
│                              evaluation, profiles, viz, inference)
├── tests/                    (47 unit tests)
├── models/                   (fitted artefacts: kmeans.pkl, gmm.pkl, scaler.pkl,
│                              imputer.pkl, cluster_profiles.json, ...)
├── app/                      (Streamlit dashboard + 5 pages)
├── reports/figures/          (all generated plots)
├── requirements.txt          (slim runtime — for Streamlit Cloud)
├── requirements-dev.txt      (full dev env)
└── runtime.txt               (Python 3.11 pin for Cloud)
```

# Appendix B — Final Cluster Summary Table

| Cluster | Name | Risk Tier | Size | GMM Confidence (mean) | Risk Score |
|---|---|---|---:|---:|---:|
| 0 | Moderate Renal Risk | Medium | 120 | 0.996 | 0.50 |
| 1 | Stable / Low-Risk | Low | 208 | 0.990 | 0.06 |
| 2 | Severe Renal Impairment | High | 72 | 0.992 | 1.00 |

# Appendix C — Hyperparameters

| Component | Setting |
|---|---|
| Random seed | 42 (everywhere) |
| KMeans | k=3, n_init=20, init=k-means++ |
| Agglomerative | n_clusters=3, linkages tested: ward / complete / average |
| DBSCAN | eps from 90th pct of k-distance, min_samples = 2 × n_features |
| GMM | n_components=3, covariance_type='full' |
| StandardScaler | mean=0, std=1 over numeric features |
| PCA | n_components selected at cumulative explained variance ≥ 0.80 (k=1 on M2 matrix) |
| UMAP | n_components=2, n_neighbors=15, min_dist=0.1 |
| t-SNE | n_components=2, perplexity=30, init='pca' |
| Bootstrap stability | 20 resamples at 80% size |

# Appendix D — Live Demo

**https://health-care21.streamlit.app/**

Repository: https://github.com/Homd11/health-care
