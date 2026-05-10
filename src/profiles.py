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
    """Heuristic naming based on dominant deltas (egfr, glucose, multimorbidity).

    If two clusters get the same heuristic name, differentiate them by eGFR severity:
    the lower-eGFR cluster keeps the original name; the higher-eGFR cluster gets
    its name softened to one severity tier lower (e.g., Severe → Moderate).
    """
    SOFTEN = {
        "Severe Renal Impairment": "Moderate Renal Risk",
        "Moderate Renal Risk": "Mild Renal Risk",
        "Anemic Subgroup": "Mild Anemia Subgroup",
        "Metabolic / Diabetic Risk": "Mild Metabolic Risk",
    }
    out = {}
    for cid, p in profiles.items():
        out[cid] = {**p, "name": _pick_name(p["feature_means"])}

    # Dedupe collisions by softening the higher-eGFR (less severe) cluster's name
    name_to_cids: dict[str, list[int]] = {}
    for cid, p in out.items():
        name_to_cids.setdefault(p["name"], []).append(cid)
    for name, cids in name_to_cids.items():
        if len(cids) <= 1 or name not in SOFTEN:
            continue
        # Sort by eGFR ascending; the cluster(s) with higher eGFR get softened names
        cids_sorted = sorted(cids, key=lambda c: out[c]["feature_means"].get("egfr", 90.0))
        for cid in cids_sorted[1:]:
            out[cid]["name"] = SOFTEN[name]
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
            0.5 * (1.0 - norm(egfr, egfrs))
            + 0.3 * norm(mb, multimorbs)
            + 0.2 * norm(an, anemias)
        )
        scored[cid] = score

    vals = sorted(scored.values())
    n = len(vals)
    if n >= 3:
        t1, t2 = vals[n // 3 - 1], vals[(2 * n) // 3 - 1]
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
