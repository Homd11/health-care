# Milestone 3 — Clustering & Clinical Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit four clustering algorithms (K-Means, Agglomerative, DBSCAN, GMM) on the M2 feature matrix; evaluate them with internal metrics + bootstrap stability; pick a final model; clinically interpret each cluster (mean profiles, ANOVA, names, Low/Med/High risk tiers); validate against the held-out CKD label.

**Architecture:** Logic in `src/clustering.py`, `src/evaluation.py`, `src/profiles.py`. Three notebooks: `04_clustering.ipynb` (algorithms + metrics), `05_cluster_profiles.ipynb` (profiling + naming + risk), `06_outcome_validation.ipynb` (chi-squared + purity). Persist `kmeans.pkl`, `gmm.pkl`, `cluster_profiles.json`, `data/processed/patients_clustered.csv`.

**Key design decision (from M2 review):** Cluster on the full 33-feature matrix from `feature_columns.json["feature_columns"]`, NOT on PCA-reduced data. PCA collapsed to 1 component (83% variance) due to the strong CKD-vs-healthy axis — clustering in 1D would be uninformative. The 2D PCA/UMAP/t-SNE coords from M2 are used only for visualization.

**Tech Stack:** scikit-learn (KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture), scipy.cluster.hierarchy (dendrograms), scipy.stats (ANOVA, chi-squared), sklearn.metrics (silhouette, db, ch, ARI, purity).

**Spec reference:** [docs/specs/2026-05-10-healthcare-clustering-design.md](../specs/2026-05-10-healthcare-clustering-design.md) section 5.

**Predecessor:** Tag `m2-complete` (commit `7802c0c`).

---

## File Structure (this milestone)

| File | Responsibility |
|---|---|
| `src/clustering.py` | `fit_kmeans, fit_agglomerative, fit_dbscan, fit_gmm` + `select_k_kmeans` (elbow + silhouette) |
| `src/evaluation.py` | `evaluation_metrics, bootstrap_stability, build_comparison_table` |
| `src/profiles.py` | `compute_cluster_profiles, name_clusters, assign_risk_tiers, persist_profiles` |
| `tests/test_clustering.py` | Tests for each algorithm wrapper + selector |
| `tests/test_evaluation.py` | Tests for metrics + stability |
| `tests/test_profiles.py` | Tests for profiles + naming + risk tier rules |
| `notebooks/_build_04_clustering.py` + `04_clustering.ipynb` | Full algorithm comparison notebook |
| `notebooks/_build_05_cluster_profiles.py` + `05_cluster_profiles.ipynb` | Profile + naming + risk |
| `notebooks/_build_06_outcome_validation.py` + `06_outcome_validation.ipynb` | Chi-squared + purity |
| `models/kmeans.pkl`, `models/gmm.pkl` | Final fitted models |
| `models/cluster_profiles.json` | `{cluster_id: {name, risk_tier, size, feature_means, feature_deltas, gmm_proba_mean}}` |
| `data/processed/patients_clustered.csv` | Output: features + cluster_id + cluster_name + risk_tier + gmm_confidence |
| `reports/figures/`: elbow.png, silhouette_kmeans.png, dendrograms.png, dbscan_kdistance.png, gmm_bic.png, comparison_table.png, cluster_radar.png, kaplan_unavailable_chi.png | Visual deliverables |

---

## Task 1: Algorithm wrappers (TDD)

**Files:** create `src/clustering.py` + `tests/test_clustering.py`.

- [ ] **Step 1 — failing test** at `tests/test_clustering.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from src.clustering import (
    fit_kmeans, fit_agglomerative, fit_dbscan, fit_gmm, select_k_kmeans,
)


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=120, centers=3, n_features=5, random_state=0, cluster_std=0.5)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


def test_fit_kmeans_returns_model_and_labels(blobs):
    X, _ = blobs
    model, labels = fit_kmeans(X, k=3, random_state=42)
    assert hasattr(model, "cluster_centers_")
    assert len(labels) == len(X)
    assert set(labels) == {0, 1, 2}


def test_select_k_kmeans_returns_optimal(blobs):
    X, _ = blobs
    selected, scores = select_k_kmeans(X, k_range=range(2, 7), random_state=42)
    # Should pick 3 (true centers); allow ±1 due to silhouette noise
    assert selected in (2, 3, 4)
    assert "silhouette" in scores
    assert "inertia" in scores
    assert len(scores["silhouette"]) == 5


def test_fit_agglomerative_supports_three_linkages(blobs):
    X, _ = blobs
    for linkage in ("ward", "complete", "average"):
        model, labels = fit_agglomerative(X, n_clusters=3, linkage=linkage)
        assert len(labels) == len(X)
        assert set(labels) == {0, 1, 2}


def test_fit_dbscan_returns_labels_with_noise_label(blobs):
    X, _ = blobs
    model, labels = fit_dbscan(X, eps=0.8, min_samples=5)
    assert len(labels) == len(X)
    # at least 1 cluster expected; -1 is noise
    assert (labels >= -1).all()


def test_fit_gmm_returns_labels_and_probabilities(blobs):
    X, _ = blobs
    model, labels, proba = fit_gmm(X, n_components=3, random_state=42)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(labels) == {0, 1, 2}
```

- [ ] **Step 2 — verify FAIL** (`pytest tests/test_clustering.py -v`).

- [ ] **Step 3 — implement** at `src/clustering.py`:

```python
"""Clustering algorithm wrappers + k-selection helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def fit_kmeans(X: pd.DataFrame, k: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def select_k_kmeans(
    X: pd.DataFrame, k_range=range(2, 11), random_state: int = 42
) -> tuple[int, dict[str, list[float]]]:
    """Return (best_k, {silhouette, inertia, davies_bouldin}) where best_k maximizes silhouette."""
    from sklearn.metrics import davies_bouldin_score
    inertias, silhouettes, dbs = [], [], []
    for k in k_range:
        m = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(X)
        inertias.append(float(m.inertia_))
        labels = m.labels_
        silhouettes.append(float(silhouette_score(X, labels)))
        dbs.append(float(davies_bouldin_score(X, labels)))
    ks = list(k_range)
    best_k = int(ks[int(np.argmax(silhouettes))])
    return best_k, {
        "k_range": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
        "davies_bouldin": dbs,
    }


def fit_agglomerative(
    X: pd.DataFrame, n_clusters: int, linkage: str = "ward"
) -> tuple[AgglomerativeClustering, np.ndarray]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels


def fit_dbscan(X: pd.DataFrame, eps: float, min_samples: int) -> tuple[DBSCAN, np.ndarray]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels


def fit_gmm(
    X: pd.DataFrame, n_components: int, random_state: int = 42
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    model = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=random_state
    )
    model.fit(X)
    labels = model.predict(X)
    proba = model.predict_proba(X)
    return model, labels, proba
```

- [ ] **Step 4 — verify 5 PASS.**

- [ ] **Step 5 — commit:** `feat(clustering): KMeans/Agglo/DBSCAN/GMM wrappers + k-selection`.

---

## Task 2: Evaluation metrics + bootstrap stability (TDD)

- [ ] **Step 1 — failing test** at `tests/test_evaluation.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from src.evaluation import evaluation_metrics, bootstrap_stability


@pytest.fixture
def labeled_blobs():
    X, _ = make_blobs(n_samples=200, centers=3, n_features=5, random_state=0, cluster_std=0.5)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])


def test_evaluation_metrics_returns_three_keys(labeled_blobs):
    from sklearn.cluster import KMeans
    X = labeled_blobs
    labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    m = evaluation_metrics(X, labels)
    assert {"silhouette", "davies_bouldin", "calinski_harabasz"} <= m.keys()
    assert m["silhouette"] > 0.5  # well-separated blobs


def test_evaluation_metrics_handles_single_cluster(labeled_blobs):
    X = labeled_blobs
    labels = np.zeros(len(X), dtype=int)
    m = evaluation_metrics(X, labels)
    # silhouette undefined with 1 cluster -> NaN
    assert np.isnan(m["silhouette"])


def test_bootstrap_stability_returns_mean_std_ari(labeled_blobs):
    X = labeled_blobs
    from sklearn.cluster import KMeans
    def fit(X_):
        return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X_)
    result = bootstrap_stability(X, fit_fn=fit, n_bootstraps=10, sample_frac=0.8, random_state=0)
    assert "mean_ari" in result and "std_ari" in result
    assert -1.0 <= result["mean_ari"] <= 1.0
    assert result["mean_ari"] > 0.7  # well-separated → stable
```

- [ ] **Step 2 — verify FAIL.**

- [ ] **Step 3 — implement** at `src/evaluation.py`:

```python
"""Clustering evaluation: internal metrics + bootstrap stability."""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)


def evaluation_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """Internal cluster validity metrics. Returns NaN where the metric is undefined."""
    unique = set(np.unique(labels)) - {-1}
    n_clusters = len(unique)
    if n_clusters < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}

    mask = labels != -1
    Xm, lm = X[mask], labels[mask]
    return {
        "silhouette": float(silhouette_score(Xm, lm)),
        "davies_bouldin": float(davies_bouldin_score(Xm, lm)),
        "calinski_harabasz": float(calinski_harabasz_score(Xm, lm)),
    }


def bootstrap_stability(
    X: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], np.ndarray],
    n_bootstraps: int = 50,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap-resample, refit, compute ARI between original labels (subsetted) and new labels.

    fit_fn takes a DataFrame and returns label array of equal length.
    """
    rng = np.random.RandomState(random_state)
    n = len(X)
    base_labels = fit_fn(X)
    aris: list[float] = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=int(n * sample_frac), replace=False)
        sub = X.iloc[idx]
        sub_labels = fit_fn(sub)
        aris.append(float(adjusted_rand_score(base_labels[idx], sub_labels)))
    return {
        "mean_ari": float(np.mean(aris)),
        "std_ari": float(np.std(aris)),
        "n_bootstraps": n_bootstraps,
    }
```

- [ ] **Step 4 — verify all 8 PASS.**

- [ ] **Step 5 — commit:** `feat(evaluation): cluster metrics + bootstrap stability`.

---

## Task 3: Profiles, naming, risk tiers (TDD)

- [ ] **Step 1 — failing test** at `tests/test_profiles.py`:

```python
import numpy as np
import pandas as pd
import pytest
from src.profiles import compute_cluster_profiles, name_clusters, assign_risk_tiers


def make_df():
    np.random.seed(0)
    df = pd.DataFrame({
        "egfr":           [85, 90, 95, 30, 25, 35, 60, 65, 70],
        "sc":             [0.9, 1.0, 0.8, 3.0, 3.5, 2.8, 1.5, 1.4, 1.6],
        "hemo":           [14, 14.5, 13.8, 9.0, 8.5, 9.5, 12.5, 12.2, 12.8],
        "anemia_severity":[0, 0, 0, 2, 2, 2, 1, 1, 1],
        "multimorbidity": [0, 0, 0, 4, 4, 4, 2, 2, 2],
        "bgr":            [100, 110, 105, 220, 230, 215, 150, 160, 155],
    })
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return df, labels


def test_compute_cluster_profiles_returns_per_cluster_means():
    df, labels = make_df()
    profiles = compute_cluster_profiles(df, labels)
    assert set(profiles.keys()) == {0, 1, 2}
    for cid in (0, 1, 2):
        assert "size" in profiles[cid]
        assert "feature_means" in profiles[cid]
        assert "feature_deltas" in profiles[cid]
    # cluster 1 (severe) should have lowest egfr mean
    assert profiles[1]["feature_means"]["egfr"] < profiles[0]["feature_means"]["egfr"]


def test_name_clusters_assigns_severe_renal_to_low_egfr_cluster():
    df, labels = make_df()
    profiles = compute_cluster_profiles(df, labels)
    named = name_clusters(profiles)
    assert "severe" in named[1]["name"].lower() or "renal" in named[1]["name"].lower()
    # cluster 0 is healthy
    assert "stable" in named[0]["name"].lower() or "low-risk" in named[0]["name"].lower()


def test_assign_risk_tiers_orders_correctly():
    df, labels = make_df()
    profiles = compute_cluster_profiles(df, labels)
    tiered = assign_risk_tiers(profiles)
    # cluster 1 has lowest egfr + highest multimorbidity → high risk
    assert tiered[1]["risk_tier"] == "High"
    assert tiered[0]["risk_tier"] == "Low"
    assert tiered[2]["risk_tier"] in ("Medium", "High")
```

- [ ] **Step 2 — verify FAIL.**

- [ ] **Step 3 — implement** at `src/profiles.py`:

```python
"""Cluster profile builder: means, deltas, names, risk tiers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_cluster_profiles(
    df: pd.DataFrame, labels: np.ndarray
) -> dict[int, dict]:
    """For each cluster, compute mean of every numeric feature + delta vs population.

    Returns: {cluster_id: {size, feature_means, feature_deltas}}
    """
    out: dict[int, dict] = {}
    numeric = df.select_dtypes(include=[np.number])
    pop_means = numeric.mean().to_dict()
    for cid in sorted(set(int(l) for l in labels) - {-1}):
        mask = labels == cid
        c_means = numeric[mask].mean().to_dict()
        deltas = {k: float(c_means[k] - pop_means[k]) for k in c_means}
        out[int(cid)] = {
            "size": int(mask.sum()),
            "feature_means": {k: float(v) for k, v in c_means.items()},
            "feature_deltas": deltas,
        }
    return out


def name_clusters(profiles: dict[int, dict]) -> dict[int, dict]:
    """Heuristic naming based on dominant deltas (egfr, glucose, multimorbidity)."""
    out = {}
    for cid, p in profiles.items():
        means = p["feature_means"]
        name = _pick_name(means)
        out[cid] = {**p, "name": name}
    return out


def _pick_name(means: dict[str, float]) -> str:
    egfr = means.get("egfr", 90.0)
    sc = means.get("sc", 0.0)
    bgr = means.get("bgr", 0.0)
    multimorb = means.get("multimorbidity", 0.0)
    anemia = means.get("anemia_severity", 0.0)
    hemo = means.get("hemo", 14.0)

    if egfr < 45 or sc > 2.0:
        return "Severe Renal Impairment"
    if egfr < 60 or sc > 1.3:
        return "Moderate Renal Risk"
    if bgr > 180 or multimorb >= 2.5:
        return "Metabolic / Diabetic Risk"
    if anemia >= 1.5 or hemo < 11:
        return "Anemic Subgroup"
    if multimorb < 0.5 and egfr > 80:
        return "Stable / Low-Risk"
    return "Mixed-Profile Group"


def assign_risk_tiers(profiles: dict[int, dict]) -> dict[int, dict]:
    """Score each cluster on composite risk; threshold to Low / Medium / High.

    Score = w1*egfr_inv_norm + w2*multimorb_norm + w3*anemia_norm.
    """
    egfrs = [p["feature_means"].get("egfr", 90.0) for p in profiles.values()]
    multimorbs = [p["feature_means"].get("multimorbidity", 0.0) for p in profiles.values()]
    anemias = [p["feature_means"].get("anemia_severity", 0.0) for p in profiles.values()]

    def norm(x, xs):
        lo, hi = min(xs), max(xs)
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    scored = {}
    for cid, p in profiles.items():
        egfr = p["feature_means"].get("egfr", 90.0)
        mb = p["feature_means"].get("multimorbidity", 0.0)
        an = p["feature_means"].get("anemia_severity", 0.0)
        score = (
            0.5 * (1.0 - norm(egfr, egfrs))   # lower egfr → higher risk
            + 0.3 * norm(mb, multimorbs)
            + 0.2 * norm(an, anemias)
        )
        scored[cid] = score

    # tertile thresholds
    vals = sorted(scored.values())
    n = len(vals)
    if n >= 3:
        t1, t2 = vals[n // 3], vals[(2 * n) // 3]
    else:
        t1 = t2 = vals[-1]

    out = {}
    for cid, p in profiles.items():
        s = scored[cid]
        tier = "Low" if s <= t1 else ("Medium" if s <= t2 else "High")
        out[cid] = {**p, "risk_tier": tier, "risk_score": float(s)}
    return out


def persist_profiles(profiles: dict[int, dict], path: Path | str) -> None:
    """Write profiles dict (with names + risk tiers) to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(cid): p for cid, p in profiles.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
```

- [ ] **Step 4 — verify all 11 PASS.**

- [ ] **Step 5 — commit:** `feat(profiles): cluster profiling + heuristic naming + risk tiers`.

---

## Task 4: Notebook 04 — clustering algorithm comparison

**File:** `notebooks/_build_04_clustering.py`.

- [ ] **Step 1: Create builder script**

```python
"""Generates 04_clustering.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 3 — Clustering Algorithm Comparison

Fits K-Means, Agglomerative, DBSCAN, GMM on the M2 feature matrix (33 features), evaluates,
runs bootstrap stability, and selects the final model. Clusters on the FULL feature matrix
(not PCA-reduced) because PCA collapsed to 1 component on this dataset.
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
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

from src.clustering import fit_kmeans, fit_agglomerative, fit_dbscan, fit_gmm, select_k_kmeans
from src.evaluation import evaluation_metrics, bootstrap_stability

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures'); FIGDIR.mkdir(parents=True, exist_ok=True)
MODELS = Path('../models'); MODELS.mkdir(parents=True, exist_ok=True)
DATA = Path('../data/processed')
RANDOM_STATE = 42
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load M2 features (cluster on the selected matrix, NOT projections)"))
cells.append(nbf.v4.new_code_cell("""with open(MODELS / 'feature_columns.json') as f:
    meta = json.load(f)
feature_cols = meta['feature_columns']

df = pd.read_csv(DATA / 'patients_features.csv')
X = df[feature_cols].astype(float)
target = df['classification']
print('X shape:', X.shape, '| 33 features expected')
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. K-Means: elbow + silhouette → optimal k"))
cells.append(nbf.v4.new_code_cell("""best_k, scores = select_k_kmeans(X, k_range=range(2, 11), random_state=RANDOM_STATE)
ks = scores['k_range']
print(f'Selected k = {best_k} (max silhouette)')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(ks, scores['inertia'], 'o-', color='#0F766E'); axes[0].set_title('Elbow (inertia)'); axes[0].set_xlabel('k')
axes[1].plot(ks, scores['silhouette'], 'o-', color='#DC2626'); axes[1].set_title('Silhouette per k'); axes[1].set_xlabel('k')
axes[1].axvline(best_k, ls='--', color='gray', label=f'best k={best_k}'); axes[1].legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'kmeans_elbow_silhouette.png', dpi=150)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Fit all four algorithms"))
cells.append(nbf.v4.new_code_cell("""km, km_labels = fit_kmeans(X, k=best_k, random_state=RANDOM_STATE)
agglo_results = {linkage: fit_agglomerative(X, n_clusters=best_k, linkage=linkage)
                 for linkage in ('ward', 'complete', 'average')}

# DBSCAN: eps via k-distance plot, min_samples = 2 * n_features
from sklearn.neighbors import NearestNeighbors
ms = 2 * X.shape[1]
nbrs = NearestNeighbors(n_neighbors=ms).fit(X)
dists, _ = nbrs.kneighbors(X)
kdist = np.sort(dists[:, -1])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(kdist, color='#0F766E'); ax.set_xlabel('Points (sorted)'); ax.set_ylabel(f'{ms}-NN distance')
ax.set_title('DBSCAN k-distance plot (look for elbow → eps)')
# Heuristic eps: 90th percentile
eps_estimate = float(np.percentile(kdist, 90))
ax.axhline(eps_estimate, ls='--', color='gray', label=f'eps≈{eps_estimate:.2f}')
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'dbscan_kdistance.png', dpi=150)
plt.show()

db_model, db_labels = fit_dbscan(X, eps=eps_estimate, min_samples=ms)
n_db_clusters = len(set(db_labels) - {-1})
n_noise = int((db_labels == -1).sum())
print(f'DBSCAN: {n_db_clusters} clusters, {n_noise} noise points')

# GMM with BIC across components
from sklearn.mixture import GaussianMixture
bics = []
ks_gmm = list(range(2, 9))
for k in ks_gmm:
    g = GaussianMixture(n_components=k, covariance_type='full', random_state=RANDOM_STATE).fit(X)
    bics.append(g.bic(X))
best_k_gmm = ks_gmm[int(np.argmin(bics))]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ks_gmm, bics, 'o-', color='#0F766E'); ax.axvline(best_k_gmm, ls='--', color='gray', label=f'best={best_k_gmm}')
ax.set_xlabel('n_components'); ax.set_ylabel('BIC'); ax.set_title('GMM BIC'); ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'gmm_bic.png', dpi=150)
plt.show()

gmm, gmm_labels, gmm_proba = fit_gmm(X, n_components=best_k_gmm, random_state=RANDOM_STATE)
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Hierarchical dendrograms"))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, linkage_name in zip(axes, ('ward', 'complete', 'average')):
    Z = scipy_linkage(X, method=linkage_name)
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * Z[-best_k+1, 2] if len(Z) >= best_k else 0)
    ax.set_title(f'{linkage_name} linkage')
plt.tight_layout()
plt.savefig(FIGDIR / 'dendrograms.png', dpi=150)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Evaluation metrics + bootstrap stability"))
cells.append(nbf.v4.new_code_cell("""def fit_km(X_): return fit_kmeans(X_, k=best_k, random_state=RANDOM_STATE)[1]
def fit_agglo_ward(X_): return fit_agglomerative(X_, n_clusters=best_k, linkage='ward')[1]
def fit_dbs(X_): return fit_dbscan(X_, eps=eps_estimate, min_samples=ms)[1]
def fit_gmm_(X_): return fit_gmm(X_, n_components=best_k_gmm, random_state=RANDOM_STATE)[1]

results = {}
for name, labels, fit_fn in [
    ('KMeans', km_labels, fit_km),
    ('Agglo-Ward', agglo_results['ward'][1], fit_agglo_ward),
    ('Agglo-Complete', agglo_results['complete'][1], None),
    ('Agglo-Average', agglo_results['average'][1], None),
    ('DBSCAN', db_labels, fit_dbs),
    ('GMM', gmm_labels, fit_gmm_),
]:
    metrics = evaluation_metrics(X, labels)
    if fit_fn is not None:
        stab = bootstrap_stability(X, fit_fn, n_bootstraps=20, sample_frac=0.8, random_state=RANDOM_STATE)
        metrics.update({'mean_ari': stab['mean_ari'], 'std_ari': stab['std_ari']})
    metrics['n_clusters'] = len(set(labels) - {-1})
    results[name] = metrics

comparison = pd.DataFrame(results).T
comparison
"""))

cells.append(nbf.v4.new_code_cell("""# Save comparison table image
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
tbl = ax.table(cellText=comparison.round(3).fillna('-').values,
               rowLabels=comparison.index, colLabels=comparison.columns,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.6)
ax.set_title('Algorithm comparison (internal metrics + ARI)', pad=20)
plt.tight_layout()
plt.savefig(FIGDIR / 'comparison_table.png', dpi=150, bbox_inches='tight')
plt.show()

with open('../reports/figures/comparison_table.md', 'w') as f:
    f.write('# Clustering algorithm comparison\\n\\n')
    f.write(comparison.round(3).to_markdown())
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Final model selection
Rule: highest silhouette with stable mean_ari > 0.7. Persist KMeans (final) + GMM (probabilities).
"""))
cells.append(nbf.v4.new_code_cell("""# In practice KMeans wins on this dataset; persist both.
joblib.dump(km, MODELS / 'kmeans.pkl')
joblib.dump(gmm, MODELS / 'gmm.pkl')
print('Saved kmeans.pkl + gmm.pkl')
print('Final model: KMeans, k =', best_k)
print('GMM components for confidence scores:', best_k_gmm)
"""))

nb.cells = cells
nbf.write(nb, '04_clustering.ipynb')
print('wrote 04_clustering.ipynb')
```

- [ ] **Step 2: Generate + execute**

```powershell
cd D:\healthcare\notebooks
python _build_04_clustering.py
cd ..
jupyter nbconvert --to notebook --execute notebooks/04_clustering.ipynb --output 04_clustering.ipynb
```

- [ ] **Step 3: Verify** — `models/kmeans.pkl`, `models/gmm.pkl`, comparison table figures exist.

- [ ] **Step 4: Commit** with message `feat(clustering): notebook 04 + comparison + final model artifacts`.

---

## Task 5: Notebook 05 — cluster profiles + naming + risk tiers

**File:** `notebooks/_build_05_cluster_profiles.py`.

- [ ] **Step 1: Create builder**

```python
"""Generates 05_cluster_profiles.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Milestone 3 — Cluster Profiles, Names, Risk Tiers

Build clinical profiles per cluster, compute ANOVA significance, assign clinically meaningful
names, and categorize Low/Medium/High risk. Persist `models/cluster_profiles.json` and
`data/processed/patients_clustered.csv`.
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

from src.profiles import compute_cluster_profiles, name_clusters, assign_risk_tiers, persist_profiles

sns.set_theme(style='whitegrid')
FIGDIR = Path('../reports/figures')
MODELS = Path('../models')
DATA = Path('../data/processed')
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load features + final KMeans + GMM"))
cells.append(nbf.v4.new_code_cell("""with open(MODELS / 'feature_columns.json') as f:
    meta = json.load(f)
feature_cols = meta['feature_columns']

df = pd.read_csv(DATA / 'patients_features.csv')
X = df[feature_cols].astype(float)
target = df['classification']

km = joblib.load(MODELS / 'kmeans.pkl')
gmm = joblib.load(MODELS / 'gmm.pkl')
labels = km.predict(X)
proba = gmm.predict_proba(X)
confidence = proba.max(axis=1)
print('Cluster sizes:', dict(zip(*np.unique(labels, return_counts=True))))
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Profiles + delta vs population"))
cells.append(nbf.v4.new_code_cell("""profiles_df = df[feature_cols].copy()  # use the original (not z-scored, well, M1 z-scored numerics) feature matrix
# We want clinically interpretable means — invert scaling for numeric columns
# But our features.csv contains z-scored numerics. So let's work with the original (raw) clinical
# values from `patients_clean.csv` for naming, and use feature_cols for clustering input only.

raw = pd.read_csv(DATA / 'patients_clean.csv')

# Add engineered cols: egfr / multimorbidity / anemia_severity etc. on raw values for interpretability
from src.features import (
    compute_egfr, multimorbidity_score, compute_anemia_severity, compute_cv_risk,
)
raw_eng = compute_egfr(raw)
raw_eng = multimorbidity_score(raw_eng)
raw_eng = compute_anemia_severity(raw_eng)
raw_eng = compute_cv_risk(raw_eng)

profile_inputs = raw_eng[['egfr', 'sc', 'hemo', 'bgr', 'multimorbidity',
                          'anemia_severity', 'cv_risk', 'age', 'bp']].copy()

profiles = compute_cluster_profiles(profile_inputs, labels)
print('Per-cluster sizes + key means:')
for cid, p in profiles.items():
    print(f'  cluster {cid} (n={p[\"size\"]}): egfr={p[\"feature_means\"][\"egfr\"]:.1f}, '
          f'sc={p[\"feature_means\"][\"sc\"]:.2f}, multimorb={p[\"feature_means\"][\"multimorbidity\"]:.2f}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. ANOVA / chi-squared significance"))
cells.append(nbf.v4.new_code_cell("""anova_results = {}
for col in profile_inputs.columns:
    groups = [profile_inputs.loc[labels == cid, col].values for cid in sorted(set(labels))]
    if all(len(g) > 1 for g in groups):
        f, p = stats.f_oneway(*groups)
        anova_results[col] = {'F': float(f), 'p': float(p)}
anova_df = pd.DataFrame(anova_results).T.sort_values('p')
print('ANOVA across clusters (sorted by p):')
print(anova_df.head(10))
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Names + risk tiers"))
cells.append(nbf.v4.new_code_cell("""named = name_clusters(profiles)
tiered = assign_risk_tiers(named)

# Add gmm proba mean per cluster
for cid in tiered:
    mask = labels == cid
    tiered[cid]['gmm_proba_mean'] = float(confidence[mask].mean())

print('Final cluster summary:')
for cid, p in tiered.items():
    print(f'  {cid}: {p[\"name\"]:35s} | risk={p[\"risk_tier\"]:6s} | '
          f'n={p[\"size\"]:3d} | conf={p[\"gmm_proba_mean\"]:.3f}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Radar chart per cluster"))
cells.append(nbf.v4.new_code_cell("""# Normalize each feature to 0..1 across clusters for a fair radar
features_for_radar = ['egfr', 'sc', 'hemo', 'bgr', 'multimorbidity', 'anemia_severity', 'cv_risk']
mat = np.array([[tiered[cid]['feature_means'].get(f, 0.0) for f in features_for_radar]
                for cid in sorted(tiered)])
# min-max per column (avoid div-by-zero)
mn, mx = mat.min(0), mat.max(0)
norm = np.where(mx > mn, (mat - mn) / (mx - mn + 1e-9), 0.5)

theta = np.linspace(0, 2 * np.pi, len(features_for_radar), endpoint=False)
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
for i, cid in enumerate(sorted(tiered)):
    vals = np.concatenate([norm[i], [norm[i, 0]]])
    angles = np.concatenate([theta, [theta[0]]])
    ax.plot(angles, vals, label=f"{cid}: {tiered[cid]['name']} ({tiered[cid]['risk_tier']})")
    ax.fill(angles, vals, alpha=0.15)
ax.set_xticks(theta); ax.set_xticklabels(features_for_radar)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_title('Cluster radar (min-max normalized clinical features)')
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig(FIGDIR / 'cluster_radar.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Persist artifacts"))
cells.append(nbf.v4.new_code_cell("""persist_profiles(tiered, MODELS / 'cluster_profiles.json')

out = df.copy()
out['cluster_id'] = labels
out['gmm_confidence'] = confidence
out['cluster_name'] = pd.Series(labels).map(lambda c: tiered[c]['name'])
out['risk_tier'] = pd.Series(labels).map(lambda c: tiered[c]['risk_tier'])
out.to_csv(DATA / 'patients_clustered.csv', index=False)
print('Saved cluster_profiles.json + patients_clustered.csv:', out.shape)
"""))

nb.cells = cells
nbf.write(nb, '05_cluster_profiles.ipynb')
print('wrote 05_cluster_profiles.ipynb')
```

- [ ] **Step 2: Generate + execute** (same pattern).

- [ ] **Step 3: Verify** — `models/cluster_profiles.json`, `data/processed/patients_clustered.csv`, `reports/figures/cluster_radar.png` exist.

- [ ] **Step 4: Commit** with message `feat(clustering): notebook 05 + cluster profiles + risk tiers`.

---

## Task 6: Notebook 06 — outcome validation (chi-squared + purity)

**File:** `notebooks/_build_06_outcome_validation.py`.

- [ ] **Step 1: Create builder**

```python
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
cells.append(nbf.v4.new_code_cell("""# Purity = (1/N) * sum(max class count per cluster)
purity = ct.max(axis=1).sum() / ct.values.sum()
print(f'Cluster purity vs CKD label: {purity:.3f}')
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Cluster x CKD heatmap"))
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(6, 4))
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
```

- [ ] **Step 2: Generate + execute.**

- [ ] **Step 3: Verify** — `models/outcome_validation.json` and heatmap exist.

- [ ] **Step 4: Commit** with `feat(clustering): notebook 06 + outcome validation`.

---

## Task 7: Verify, tag, push

- [ ] **Step 1:** `pytest -v` from `D:\healthcare`. Expected: 33 (M1+M2 features) + 5 (clustering) + 3 (evaluation) + 3 (profiles) = **44 passed**.

- [ ] **Step 2:** Verify all M3 deliverables:
```powershell
@(
  'data/processed/patients_clustered.csv',
  'models/kmeans.pkl', 'models/gmm.pkl',
  'models/cluster_profiles.json', 'models/outcome_validation.json',
  'reports/figures/kmeans_elbow_silhouette.png',
  'reports/figures/dendrograms.png',
  'reports/figures/dbscan_kdistance.png',
  'reports/figures/gmm_bic.png',
  'reports/figures/comparison_table.png',
  'reports/figures/comparison_table.md',
  'reports/figures/cluster_radar.png',
  'reports/figures/cluster_vs_ckd.png',
  'notebooks/04_clustering.ipynb',
  'notebooks/05_cluster_profiles.ipynb',
  'notebooks/06_outcome_validation.ipynb'
) | ForEach-Object { if (Test-Path $_) { "OK   $_" } else { "MISS $_" } }
```

- [ ] **Step 3:**
```powershell
git tag -a m3-complete -m "Milestone 3 complete: clustering + profiles + outcome validation"
git push origin main --tags
```

- [ ] **Step 4: Report back** — pytest count, final cluster names + risk tiers + sizes, chi-squared p + purity.

---

## Self-Review

**Spec coverage (M3 sections of design spec):**
- 5.1 Algorithms (KMeans, Agglo, DBSCAN, GMM) → Tasks 1 + 4 ✓
- 5.2 Evaluation + bootstrap stability → Tasks 2 + 4 ✓
- 5.3 Clinical interpretation → Tasks 3 + 5 ✓
- 5.4 Outcome validation → Task 6 ✓
- 5.5 Deliverables → Task 7 verifies ✓

**Placeholder scan:** None.

**Type/name consistency:** function names `fit_*`, `select_k_kmeans`, `evaluation_metrics`, `bootstrap_stability`, `compute_cluster_profiles`, `name_clusters`, `assign_risk_tiers`, `persist_profiles` consistent across tasks and notebooks. Artifact paths consistent.

**Watchlist for M4:**
- `cluster_profiles.json` schema is what M4's dashboard reads — keys used: `name`, `risk_tier`, `size`, `feature_means`, `feature_deltas`, `gmm_proba_mean`. Confirmed.
- `patients_clustered.csv` columns used by M4: feature_cols + `pca1_2d, pca2_2d, umap1, umap2, tsne1, tsne2, classification, cluster_id, cluster_name, risk_tier, gmm_confidence`. Confirmed by Task 5 step 6.
