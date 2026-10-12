"""HTTP contract tests use only synthetic, trusted local releases."""
from __future__ import annotations

import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from ml.serving_contract import MODEL_SPECS, SCHEMA_VERSION, SOURCE_IDS, runtime_versions


class FixturePredictor:
    """Serializable predictor enforcing the real serving DataFrame contract."""

    def __init__(self, model_id, *, probability=.8, estimate=7.25, warning=False,
                 expected_nulls=(), probabilities=None):
        self.feature_names_in_ = np.asarray(MODEL_SPECS[model_id][2], dtype=object)
        self.classes_ = np.array([0, 1])
        self.probability = probability
        self.estimate = estimate
        self.warning = warning
        self.expected_nulls = expected_nulls
        self.probabilities = probabilities

    def _check(self, frame):
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == list(self.feature_names_in_)
        assert len(frame) == 1
        for name in self.expected_nulls:
            assert np.isnan(frame[name].iloc[0])
        if self.warning:
            warnings.warn("private runtime warning /secret/training/location", RuntimeWarning)

    def predict(self, frame):
        self._check(frame)
        return [self.estimate]

    def predict_proba(self, frame):
        self._check(frame)
        if self.probabilities is not None:
            return self.probabilities
        return [[1 - self.probability, self.probability]]


def metadata_for(model_id, release_id="synthetic-test-release"):
    task, mode, features = MODEL_SPECS[model_id]
    return {
        "schema_version": SCHEMA_VERSION, "model_id": model_id, "task": task,
        "mode": mode, "feature_order": features, "release_id": release_id,
        "policy": "physical_sanitized", "fit_scope": "primary_training_only",
        "status": "demo_only", "calibrated": False,
        "source_experiment_id": SOURCE_IDS[model_id],
        "target_cohort": "binary_targets" if task == "classification" else "physical_valid_targets",
        "dataset_sha256": "0" * 64, "source_candidate_sha256": "1" * 64,
        "source_results_sha256": "2" * 64,
        "threshold": .7 if task == "classification" else None,
        "probability_semantics": "P(Potability=1)" if task == "classification" else None,
        "reference_ranges": {name: {"min": 1., "q01": 2., "median": 5., "q99": 8., "max": 10.}
                             for name in features},
        "evaluation": {}, "limitations": ["synthetic fixture only"],
        "training_versions": runtime_versions(),
    }


def write_release(directory, ids=None, *, predictors=None, metadata=None,
                  release_id="synthetic-test-release"):
    manifest = {"schema_version": SCHEMA_VERSION, "release_id": release_id,
                "intended_use": "competition_demo_only", "models": {}}
    for model_id in (MODEL_SPECS if ids is None else ids):
        meta = (metadata or {}).get(model_id, metadata_for(model_id, release_id))
        predictor = (predictors or {}).get(model_id, FixturePredictor(model_id))
        path = directory / f"{model_id}.joblib"
        joblib.dump({"pipeline": predictor, "metadata": meta}, path)
        manifest["models"][model_id] = {"file": path.name,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "metadata": meta}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def payload(model_id="classification_full"):
    # Reverse incoming key order so predictors check actual serving order.
    return {name: 5. for name in reversed(MODEL_SPECS[model_id][2])}


def codes(result):
    return {warning["code"] for warning in result["warnings"]}


@pytest.fixture
def client(tmp_path):
    write_release(tmp_path)
    return TestClient(create_app(artifacts_dir=tmp_path))


def test_missing_release_health_and_prediction_recover(tmp_path):
    client = TestClient(create_app(artifacts_dir=tmp_path))
    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert not health["ready"]
    response = client.post("/predict/quick", json=payload("classification_quick"))
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "missing_manifest"
    assert str(tmp_path) not in response.text
    write_release(tmp_path)
    assert client.get("/health").json()["ready"]
    assert client.post("/predict/quick", json=payload("classification_quick")).status_code == 200


@pytest.mark.parametrize("route", ["/predict", "/predict/full", "/predict/quick",
                                    "/classify/full", "/classify/quick", "/ph/full", "/ph/quick"])
def test_all_routes_and_named_frame_order(client, route):
    model_id = ("regression" if route.startswith("/ph") else "classification") + (
        "_quick" if route.endswith("quick") else "_full")
    response = client.post(route, json=payload(model_id))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["release_id"] == "synthetic-test-release"
    assert result["contract_version"] == SCHEMA_VERSION
    assert result["api_version"] == "2.0.0"
    if route.startswith("/predict"):
        assert result["classification"]["probability_safe"] == .8
        assert result["regression"]["estimate"] == 7.25
    elif route.startswith("/ph"):
        assert result["reliability"] == "limited"
        assert {"weak_regression", "not_lab_measurement"} <= codes(result)
    else:
        assert {"uncalibrated_probability", "not_certification"} <= codes(result)


@pytest.mark.parametrize("value", ["5", True, False, [], {}, -1, 1001])
def test_bad_values_are_safe_422(client, value):
    data = payload("classification_quick")
    data["Turbidity"] = value
    response = client.post("/classify/quick", json=data)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    for detail in response.json()["details"]:
        assert set(detail) == {"loc", "type", "msg"}


@pytest.mark.parametrize("body", [
    '{"Turbidity":NaN,"Hardness":5,"TDS":5}',
    '{"Turbidity":Infinity,"Hardness":5,"TDS":5}',
    '{"Turbidity":-Infinity,"Hardness":5,"TDS":5}',
    '{"Turbidity":1e9999,"Hardness":5,"TDS":5}',
    '{"Turbidity":', 'not json', 'null', '[]', '"string"', b'\xff',
])
def test_nonfinite_and_malformed_json_are_safe_422(client, body):
    response = client.post("/predict/quick", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.json()["error"] == "invalid_request"
    assert "ctx" not in response.json()["details"][0]
    assert "input" not in response.json()["details"][0]


@pytest.mark.parametrize("route,model_id", [("/classify/full", "classification_full"),
                                            ("/ph/full", "regression_full"),
                                            ("/predict/quick", "classification_quick")])
def test_exact_required_keys_all_null_and_extras(client, route, model_id):
    data = payload(model_id)
    missing = dict(data)
    missing.pop(next(iter(missing)))
    assert client.post(route, json=missing).status_code == 422
    assert client.post(route, json={name: None for name in data}).status_code == 422
    assert client.post(route, json={**data, "Potability": 1}).status_code == 422
    assert client.post(route, json={**data, "hardness": 5}).status_code == 422


@pytest.mark.parametrize("value", [-.01, 14.01])
def test_ph_physical_range(client, value):
    assert client.post("/classify/full", json={**payload(), "pH": value}).status_code == 422


def test_explicit_null_becomes_nan_and_not_regression_fill(tmp_path):
    predictor = FixturePredictor("classification_full", expected_nulls=["pH"])
    write_release(tmp_path, predictors={"classification_full": predictor})
    client = TestClient(create_app(artifacts_dir=tmp_path))
    response = client.post("/predict/full", json={**payload(), "pH": None})
    assert response.status_code == 200
    classification = response.json()["classification"]
    assert classification["assessment"] == "insufficient_evidence"
    assert "missing_measurements" in codes(classification)
    assert response.json()["regression"]["estimate"] == 7.25


def test_ph_endpoint_excludes_measured_ph_and_target(client):
    data = payload("regression_full")
    assert len(data) == 17 and "pH" not in data and "Potability" not in data
    assert client.post("/ph/full", json=data).status_code == 200
    assert client.post("/ph/full", json={**data, "pH": 7}).status_code == 422


def test_combined_inference_cannot_estimate_ph_from_only_ph(client):
    data = {name: None for name in payload()}
    data["pH"] = 7.
    assert client.post("/predict/full", json=data).status_code == 422
    classification = client.post("/classify/full", json=data)
    assert classification.status_code == 200
    assert classification.json()["assessment"] == "insufficient_evidence"


@pytest.mark.parametrize("task,route", [("classification", "/classify"), ("regression", "/ph")])
def test_independent_task_needs_only_own_model(tmp_path, task, route):
    model_id = f"{task}_full"
    write_release(tmp_path, ids=[model_id])
    client = TestClient(create_app(artifacts_dir=tmp_path))
    assert client.post(f"{route}/full", json=payload(model_id)).status_code == 200
    assert client.post("/predict/full", json=payload()).status_code == 503


def test_probability_is_class_one_not_label_confidence_and_uses_recorded_threshold(tmp_path):
    model_id = "classification_full"
    meta = metadata_for(model_id)
    meta["threshold"] = .9
    # predict returns a completely invalid classification label; classification
    # must use predict_proba + stored threshold rather than predict.
    write_release(tmp_path, metadata={model_id: meta}, predictors={model_id: FixturePredictor(model_id, probability=.6)})
    client = TestClient(create_app(artifacts_dir=tmp_path))
    result = client.post("/classify/full", json=payload()).json()
    assert result["class_id"] == 0 and result["model_label"] == "unsafe"
    assert result["probability_safe"] == .6
    assert result["decision_threshold"] == .9
    assert result["assessment"] == "predicted_unsafe"


def test_quick_always_abstains(client):
    result = client.post("/classify/quick", json=payload("classification_quick")).json()
    assert result["class_id"] == 1
    assert result["assessment"] == "insufficient_evidence"
    assert "limited_inputs" in codes(result)


@pytest.mark.parametrize("value,assessment,code", [
    (None, "insufficient_evidence", "missing_measurements"),
    (11, "insufficient_evidence", "outside_training_range"),
    (.5, "insufficient_evidence", "outside_training_range"),
    (9, "predicted_safe", "unusual_measurements"),
    (1.5, "predicted_safe", "unusual_measurements"),
])
def test_full_input_quality(client, value, assessment, code):
    result = client.post("/classify/full", json={**payload(), "Hardness": value}).json()
    assert result["assessment"] == assessment
    assert code in codes(result)
    assert next(w for w in result["warnings"] if w["code"] == code)["features"] == ["Hardness"]


def test_near_threshold_is_heuristic_abstention(tmp_path):
    model_id = "classification_full"
    write_release(tmp_path, predictors={model_id: FixturePredictor(model_id, probability=.72)})
    result = TestClient(create_app(artifacts_dir=tmp_path)).post("/classify/full", json=payload()).json()
    assert result["assessment"] == "insufficient_evidence"
    assert "near_threshold" in codes(result)
    assert "heuristic" in next(w["message"] for w in result["warnings"] if w["code"] == "near_threshold")


def test_unknown_training_range_abstains(tmp_path):
    model_id = "classification_full"
    meta = metadata_for(model_id)
    meta["reference_ranges"]["TDS"] = {key: None for key in ("min", "q01", "median", "q99", "max")}
    write_release(tmp_path, metadata={model_id: meta})
    result = TestClient(create_app(artifacts_dir=tmp_path)).post("/classify/full", json=payload()).json()
    assert result["assessment"] == "insufficient_evidence"
    assert "unknown_training_range" in codes(result)


def test_regression_missing_inputs_warn_and_become_nan(tmp_path):
    model_id = "regression_full"
    write_release(tmp_path, predictors={model_id: FixturePredictor(model_id, expected_nulls=["TDS"])})
    result = TestClient(create_app(artifacts_dir=tmp_path)).post("/ph/full", json={**payload(model_id), "TDS": None}).json()
    assert result["estimate"] == 7.25
    assert result["reliability"] == "limited"
    assert "missing_measurements" in codes(result)


@pytest.mark.parametrize("probabilities", [
    [[float("nan"), .8]], [[.2, float("inf")]], [[-.1, 1.1]], [[.1, .8]],
    [.2, .8], [[.2, .3, .5]], [[".2", ".8"]], [[False, True]],
])
def test_invalid_probabilities_fail_closed(tmp_path, probabilities):
    model_id = "classification_full"
    write_release(tmp_path, predictors={model_id: FixturePredictor(model_id, probabilities=probabilities)})
    response = TestClient(create_app(artifacts_dir=tmp_path)).post("/classify/full", json=payload())
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "invalid_model_output"


@pytest.mark.parametrize("estimate", [float("nan"), float("inf"), -.1, 14.1, "7", True, [7]])
def test_invalid_ph_prediction_not_clipped(tmp_path, estimate):
    model_id = "regression_full"
    write_release(tmp_path, predictors={model_id: FixturePredictor(model_id, estimate=estimate)})
    response = TestClient(create_app(artifacts_dir=tmp_path)).post("/ph/full", json=payload(model_id))
    assert response.status_code == 503
    assert response.json()["detail"]["reason"] == "invalid_model_output"


def test_runtime_warning_public_but_does_not_leak_details(tmp_path):
    predictors = {model_id: FixturePredictor(model_id, warning=True) for model_id in MODEL_SPECS}
    write_release(tmp_path, predictors=predictors)
    response = TestClient(create_app(artifacts_dir=tmp_path)).post("/predict/full", json=payload())
    assert response.status_code == 200
    assert "model_runtime_warning" in codes(response.json()["classification"])
    assert "model_runtime_warning" in codes(response.json()["regression"])
    assert "secret" not in response.text


def test_models_and_health_are_public_metadata_only(client, tmp_path):
    health = client.get("/health")
    assert health.json()["ready"]
    assert health.json()["profiles"]["full"] == {"available": True, "classification": True, "regression": True}
    response = client.get("/models")
    result = response.json()
    assert set(result["models"]) == set(MODEL_SPECS)
    assert result["units"] == "unknown"
    assert result["bounds_policy"] == "engineering_checks_not_BIS"
    for entry in result["models"].values():
        assert entry["available"]
        assert all(f["units"] == "unknown" and f["required"] and f["nullable"] for f in entry["features"])
    for text in (response.text, health.text):
        assert str(tmp_path) not in text
        assert "source_experiment_id" not in text
        assert "training_versions" not in text
        assert ".joblib" not in text
    assert client.get("/openapi.json").json()["info"]["version"] == "2.0.0"


def test_models_do_not_echo_arbitrary_metadata(tmp_path):
    meta = metadata_for("classification_full")
    meta["reference_ranges"]["TDS"]["private_path"] = "/secret/training/path"
    meta["evaluation"]["private_path"] = "/secret/training/path"
    meta["limitations"] = ["/secret/training/path"]
    write_release(tmp_path, metadata={"classification_full": meta})
    response = TestClient(create_app(artifacts_dir=tmp_path)).get("/models")
    assert response.status_code == 200
    assert "secret" not in response.text
