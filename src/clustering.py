"""Clustering algorithm wrappers + k-selection helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score


def fit_kmeans(X: pd.DataFrame, k: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def select_k_kmeans(
    X: pd.DataFrame, k_range=range(2, 11), random_state: int = 42
) -> tuple[int, dict[str, list[float]]]:
    """Return (best_k, {silhouette, inertia, davies_bouldin}) where best_k maximizes silhouette."""
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
