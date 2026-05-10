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
