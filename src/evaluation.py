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
