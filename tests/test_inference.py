import json
import pandas as pd
import pytest
from src.inference import predict_patient


@pytest.fixture
def healthy_patient_input():
    return {
        "age": 35, "bp": 120, "sg": 1.020, "al": 0, "su": 0,
        "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
        "bgr": 100, "bu": 30, "sc": 0.9, "sod": 140, "pot": 4.2, "hemo": 14.5,
        "pcv": 45, "wc": 7000, "rc": 5.0,
        "htn": "no", "dm": "no", "cad": "no", "appet": "good",
        "pe": "no", "ane": "no",
    }


@pytest.fixture
def severe_ckd_patient_input():
    return {
        "age": 70, "bp": 160, "sg": 1.010, "al": 4, "su": 2,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "present",
        "bgr": 250, "bu": 120, "sc": 5.5, "sod": 132, "pot": 5.8, "hemo": 8.0,
        "pcv": 25, "wc": 11000, "rc": 3.0,
        "htn": "yes", "dm": "yes", "cad": "yes", "appet": "poor",
        "pe": "yes", "ane": "yes",
    }


def test_predict_patient_returns_required_fields(healthy_patient_input):
    result = predict_patient(healthy_patient_input)
    assert "cluster_id" in result
    assert "cluster_name" in result
    assert "risk_tier" in result
    assert "confidence" in result
    assert "summary" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_healthy_patient_routed_to_low_risk(healthy_patient_input):
    result = predict_patient(healthy_patient_input)
    assert result["risk_tier"] == "Low"
    assert "stable" in result["cluster_name"].lower() or "low-risk" in result["cluster_name"].lower()


def test_severe_patient_routed_to_high_risk(severe_ckd_patient_input):
    result = predict_patient(severe_ckd_patient_input)
    assert result["risk_tier"] == "High"
    assert "severe" in result["cluster_name"].lower() or "renal" in result["cluster_name"].lower()
