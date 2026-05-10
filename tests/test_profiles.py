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
    assert profiles[1]["feature_means"]["egfr"] < profiles[0]["feature_means"]["egfr"]


def test_name_clusters_assigns_severe_renal_to_low_egfr_cluster():
    df, labels = make_df()
    profiles = compute_cluster_profiles(df, labels)
    named = name_clusters(profiles)
    assert "severe" in named[1]["name"].lower() or "renal" in named[1]["name"].lower()
    assert "stable" in named[0]["name"].lower() or "low-risk" in named[0]["name"].lower()


def test_assign_risk_tiers_orders_correctly():
    df, labels = make_df()
    profiles = compute_cluster_profiles(df, labels)
    tiered = assign_risk_tiers(profiles)
    assert tiered[1]["risk_tier"] == "High"
    assert tiered[0]["risk_tier"] == "Low"
    assert tiered[2]["risk_tier"] in ("Medium", "High")
