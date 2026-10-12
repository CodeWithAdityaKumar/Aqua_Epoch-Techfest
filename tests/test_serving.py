"""Release integrity, compatibility, caching, and real sklearn pipeline tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.app import create_app
from backend.artifacts import ArtifactStore, ArtifactUnavailable
from ml.quality import PhysicalSanitizer
from ml.serving_contract import MODEL_SPECS
from test_backend import FixturePredictor, metadata_for, payload, write_release


MODEL_ID = "classification_full"


def save_manifest(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def assert_reason(store, reason):
    with pytest.raises(ArtifactUnavailable) as exc:
        store.load(MODEL_ID)
    assert exc.value.reason == reason


@pytest.mark.parametrize("mutate,reason", [
    (lambda m: m.update(schema_version="phase1"), "incompatible_manifest"),
    (lambda m: m.update(intended_use="certified"), "incompatible_manifest"),
    (lambda m: m.update(release_id=""), "incompatible_manifest"),
    (lambda m: m["models"][MODEL_ID]["metadata"].update(schema_version="old"), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"].update(release_id="other"), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"].update(feature_order=["TDS"]), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"]["training_versions"].update(sklearn="0.0.0"), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"].update(threshold=.4), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"].update(probability_semantics="confidence"), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID]["metadata"]["reference_ranges"].update(TDS={}), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID].update(sha256="broken"), "incompatible_model_metadata"),
    (lambda m: m["models"][MODEL_ID].update(sha256="0" * 64), "artifact_hash_mismatch"),
    (lambda m: m["models"][MODEL_ID].update(file="../classification_full.joblib"), "invalid_artifact_path"),
    (lambda m: m["models"][MODEL_ID].update(file="/tmp/classification_full.joblib"), "invalid_artifact_path"),
    (lambda m: m["models"][MODEL_ID].update(file="classification_full.pkl"), "invalid_artifact_path"),
])
def test_validation_and_hash_before_deserialization(tmp_path, monkeypatch, mutate, reason):
    manifest = write_release(tmp_path)
    mutate(manifest)
    save_manifest(tmp_path, manifest)
    calls = []
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: calls.append(args))
    assert_reason(ArtifactStore(tmp_path), reason)
    assert calls == []


@pytest.mark.parametrize("raw", ['{', '{}', '[]', '{"models":NaN}', '{"release_id":"a","release_id":"b"}'])
def test_malformed_manifest(tmp_path, raw):
    (tmp_path / "manifest.json").write_text(raw)
    with pytest.raises(ArtifactUnavailable):
        ArtifactStore(tmp_path).load(MODEL_ID)


def test_symlink_escape_for_model_and_manifest(tmp_path, monkeypatch):
    root = tmp_path / "release"
    root.mkdir()
    write_release(root)
    outside = tmp_path / "outside.joblib"
    artifact = root / f"{MODEL_ID}.joblib"
    artifact.rename(outside)
    artifact.symlink_to(outside)
    calls = []
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: calls.append(args))
    assert_reason(ArtifactStore(root), "invalid_artifact_path")
    manifest = root / "manifest.json"
    manifest.rename(tmp_path / "outside.json")
    manifest.symlink_to(tmp_path / "outside.json")
    assert_reason(ArtifactStore(root), "invalid_artifact_path")
    assert calls == []


def test_missing_artifact_recovers_and_legacy_files_never_fallback(tmp_path):
    manifest = write_release(tmp_path)
    artifact = tmp_path / f"{MODEL_ID}.joblib"
    data = artifact.read_bytes()
    artifact.unlink()
    (tmp_path / "classifier.pkl").write_bytes(data)
    store = ArtifactStore(tmp_path)
    assert_reason(store, "missing_artifact")
    artifact.write_bytes(data)
    assert store.load(MODEL_ID).metadata == manifest["models"][MODEL_ID]["metadata"]


def test_cache_changes_invalidate_and_errors_recover(tmp_path, monkeypatch):
    write_release(tmp_path)
    original_load = joblib.load
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counted)
    store = ArtifactStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        models = list(executor.map(lambda _: store.load(MODEL_ID), range(8)))
    assert all(model is models[0] for model in models)
    assert len(calls) == 1
    write_release(tmp_path, release_id="replacement")
    replacement = store.load(MODEL_ID)
    assert replacement.metadata["release_id"] == "replacement"
    assert replacement is not models[0]
    artifact = tmp_path / f"{MODEL_ID}.joblib"
    content = artifact.read_bytes()
    artifact.write_bytes(b"corrupt")
    assert_reason(store, "artifact_hash_mismatch")
    artifact.write_bytes(content)
    assert store.load(MODEL_ID).metadata["release_id"] == "replacement"
    assert len(calls) == 3


@pytest.mark.parametrize("fault", ["features", "classes", "classes_order", "classes_bool", "proba", "predict", "metadata", "bare", "corrupt"])
def test_embedded_schema_and_capabilities(tmp_path, fault):
    manifest = write_release(tmp_path)
    predictor = FixturePredictor(MODEL_ID)
    meta = metadata_for(MODEL_ID)
    if fault == "features":
        predictor.feature_names_in_ = predictor.feature_names_in_[::-1]
    if fault == "classes":
        predictor.classes_ = ["0", "1"]
    if fault == "classes_order":
        predictor.classes_ = [1, 0]
    if fault == "classes_bool":
        predictor.classes_ = [False, True]
    if fault == "proba":
        predictor.predict_proba = None
    if fault == "predict":
        predictor.predict = None
    if fault == "metadata":
        meta["release_id"] = "embedded-other-release"
    artifact = tmp_path / f"{MODEL_ID}.joblib"
    joblib.dump(predictor if fault == "bare" else {"pipeline": predictor, "metadata": meta}, artifact)
    if fault == "corrupt":
        artifact.write_bytes(b"not a serialized model")
    manifest["models"][MODEL_ID]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    save_manifest(tmp_path, manifest)
    assert_reason(ArtifactStore(tmp_path), "incompatible_artifact")


def test_real_sklearn_sanitizer_imputer_pipeline(tmp_path):
    predictors = {}
    for model_id, (task, _, features) in MODEL_SPECS.items():
        frame = pd.DataFrame([[3.] * len(features), [5.] * len(features), [7.] * len(features)], columns=features)
        estimator = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
        model = Pipeline([("sanitize", PhysicalSanitizer()), ("impute", SimpleImputer()), ("model", estimator)])
        model.fit(frame, [0, 1, 1] if task == "classification" else [6.5, 7., 7.5])
        predictors[model_id] = model
    write_release(tmp_path, predictors=predictors)
    client = TestClient(create_app(artifacts_dir=tmp_path))
    response = client.post("/predict/full", json={**payload(), "TDS": None})
    assert response.status_code == 200, response.text
    assert response.json()["regression"]["estimate"] == 7.
    assert response.json()["classification"]["probability_safe"] == pytest.approx(2 / 3)
    assert response.json()["classification"]["assessment"] == "insufficient_evidence"
