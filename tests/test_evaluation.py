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
    assert m["silhouette"] > 0.5


def test_evaluation_metrics_handles_single_cluster(labeled_blobs):
    X = labeled_blobs
    labels = np.zeros(len(X), dtype=int)
    m = evaluation_metrics(X, labels)
    assert np.isnan(m["silhouette"])


def test_bootstrap_stability_returns_mean_std_ari(labeled_blobs):
    X = labeled_blobs
    from sklearn.cluster import KMeans
    def fit(X_):
        return KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X_)
    result = bootstrap_stability(X, fit_fn=fit, n_bootstraps=10, sample_frac=0.8, random_state=0)
    assert "mean_ari" in result and "std_ari" in result
    assert -1.0 <= result["mean_ari"] <= 1.0
    assert result["mean_ari"] > 0.7
