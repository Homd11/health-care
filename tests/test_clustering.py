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
    assert (labels >= -1).all()


def test_fit_gmm_returns_labels_and_probabilities(blobs):
    X, _ = blobs
    model, labels, proba = fit_gmm(X, n_components=3, random_state=42)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(labels) == {0, 1, 2}
