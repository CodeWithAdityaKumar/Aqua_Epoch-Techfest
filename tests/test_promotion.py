"""Release promotion preserves fitted estimators and fails closed on evidence drift."""
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.evaluate import build_pipeline
from ml.promote import FIT_SCOPE, main, promote
from ml.quality import FULL_FEATURES, PhysicalSanitizer
from ml.serving_contract import MODEL_SPECS, SCHEMA_VERSION, SOURCE_IDS, runtime_versions, validate_metadata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(fixture):
    (fixture["source"] / "results.json").write_text(json.dumps(fixture["report"]))


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "reports" / "phase2_validated"
    (source / "candidates").mkdir(parents=True)
    frame = pd.DataFrame({feature: np.arange(12, dtype=float) + i + 1
                          for i, feature in enumerate(FULL_FEATURES)})
    frame["pH"] = np.linspace(6, 8, 12)
    frame["Potability"] = np.arange(12) % 2
    frame.loc[0, "TDS"] = -500
    frame.loc[:7, "Iron"] = np.nan
    frame.loc[8:, "TDS"] = 1e9  # Holdout values must not enter training bounds.
    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False)
    train, test = list(range(8)), list(range(8, 12))
    common = {"phase": 2, "experimental": True, "approved": False,
              "dataset_sha256": sha(dataset), "dataset": str(dataset), "versions": runtime_versions(),
              "seed": 2026, "validation": "Internal holdout, NOT unseen external data."}
    report = {"status": "completed", "metadata": common, "cohorts": {}, "experiments": []}
    pipelines = {}
    for model_id, (task, mode, features) in MODEL_SPECS.items():
        cohort = "binary_targets" if task == "classification" else "physical_valid_targets"
        target = "Potability" if task == "classification" else "pH"
        pipeline = build_pipeline(task, "linear", "physical_sanitized").fit(frame.loc[train, features], frame.loc[train, target])
        pipelines[model_id] = pipeline
        key = "f1" if task == "classification" else "rmse"
        threshold = {"value": .75, "training_row_ids": train, "provenance": "training OOF"} if task == "classification" else None
        meta = {**common, "experiment_id": SOURCE_IDS[model_id], "task": task, "cohort": cohort,
                "mode": mode, "policy": "physical_sanitized", "feature_order": features,
                "selected_model": "linear", "selection_metric": f"training_cv_{key}",
                "model_parameters": pipeline.named_steps["model"].get_params(), "threshold": threshold,
                "probability_semantics": "P(Potability=1), NOT confidence in the predicted class" if task == "classification" else None,
                "fit_scope": FIT_SCOPE, "training_row_ids": train, "holdout_row_ids": test, "target_imputed": False}
        joblib.dump({"pipeline": pipeline, "metadata": meta}, source / "candidates" / f"{SOURCE_IDS[model_id]}.joblib")
        report["cohorts"][f"{task}/{cohort}"] = {"split": {"train_row_ids": train, "test_row_ids": test}}
        result = {"id": SOURCE_IDS[model_id], "task": task, "cohort": cohort, "mode": mode,
                  "policy": "physical_sanitized", "train_count": len(train), "holdout_count": len(test),
                  "selected_configuration": meta, "threshold": threshold,
                  "cv_candidates": {"dummy": {"mean_std": {key: {"mean": .3 if task == "classification" else 3., "std": .1}}},
                                    "linear": {"mean_std": {key: {"mean": .7 if task == "classification" else 1., "std": .1}}}},
                  "selected_internal_holdout": {key: .6}, "dummy_internal_holdout": {key: .2},
                  "candidate_bundle": "/untrusted/path/ignored.joblib"}
        report["experiments"].append(result)
        if task == "regression":
            raw = copy.deepcopy(result)
            raw["id"] = f"regression_raw_targets_{mode}_physical_sanitized"
            raw["cohort"] = "raw_targets"
            report["experiments"].append(raw)
    # Unrelated raw notes and Phase 1 files must stay byte-identical and private.
    (source / "private-notes.txt").write_text("DO NOT COPY PRIVATE NOTES")
    phase1 = tmp_path / "artifacts" / "phase1"
    phase1.mkdir(parents=True)
    (phase1 / "old.joblib").write_bytes(b"existing phase 1 artifact")
    fixture = {"source": source, "dataset": dataset, "output": tmp_path / "artifacts" / "phase3",
               "frame": frame, "report": report, "pipelines": pipelines, "phase1": phase1}
    write_report(fixture)
    return fixture


def run(fixture):
    return promote(fixture["source"], fixture["dataset"], fixture["output"])


def test_cli_release_preserves_predictions_hashes_sources_and_training_bounds(evidence, monkeypatch):
    snapshot = {path: path.read_bytes() for root in (evidence["source"], evidence["phase1"])
                for path in root.rglob("*") if path.is_file()}
    snapshot[evidence["dataset"]] = evidence["dataset"].read_bytes()

    def no_fit(*args, **kwargs):
        pytest.fail("Promotion must never refit an estimator")

    monkeypatch.setattr(Pipeline, "fit", no_fit)
    manifest = main(["--source", str(evidence["source"]), "--dataset", str(evidence["dataset"]),
                     "--output", str(evidence["output"])])
    output = evidence["output"]
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["intended_use"] == "competition_demo_only"
    assert set(manifest["models"]) == set(MODEL_SPECS)
    assert {p.name for p in output.iterdir()} == {
        *(f"{model_id}.joblib" for model_id in MODEL_SPECS), "manifest.json", "promotion.json", "requirements-serving.txt"}
    for model_id, entry in manifest["models"].items():
        task, _, features = MODEL_SPECS[model_id]
        assert entry["file"] == f"{model_id}.joblib"
        assert entry["sha256"] == sha(output / entry["file"])
        bundle = joblib.load(output / entry["file"])
        meta = bundle["metadata"]
        assert meta == entry["metadata"]
        validate_metadata(meta, model_id)
        assert meta["release_id"] == manifest["release_id"]
        assert meta["source_candidate_sha256"] == sha(evidence["source"] / "candidates" / f"{SOURCE_IDS[model_id]}.joblib")
        assert meta["source_results_sha256"] == sha(evidence["source"] / "results.json")
        assert meta["dataset_sha256"] == sha(evidence["dataset"])
        assert meta["training_versions"] == evidence["report"]["metadata"]["versions"]
        assert meta["status"] == "demo_only"
        X = evidence["frame"][features]
        original = evidence["pipelines"][model_id]
        np.testing.assert_array_equal(bundle["pipeline"].predict(X), original.predict(X))
        if task == "classification":
            np.testing.assert_array_equal(bundle["pipeline"].predict_proba(X), original.predict_proba(X))
            assert meta["threshold"] == .75
            assert meta["probability_semantics"] == "P(Potability=1)"
        else:
            assert meta["threshold"] is None
            assert "raw_regression_cohort" in meta["evaluation"]
        training = PhysicalSanitizer().fit_transform(evidence["frame"].loc[:7, features])
        assert meta["reference_ranges"]["TDS"]["max"] == training["TDS"].max() < 1e9
        assert meta["reference_ranges"]["TDS"]["min"] >= 0
        for feature in features:
            values = training[feature].dropna()
            bounds = meta["reference_ranges"][feature]
            if values.empty:
                assert all(value is None for value in bounds.values())
            else:
                np.testing.assert_allclose(list(bounds.values()), values.quantile([0, .01, .5, .99, 1]))
    for path, payload in snapshot.items():
        assert path.read_bytes() == payload
    requirements = (output / "requirements-serving.txt").read_text()
    versions = runtime_versions()
    assert f"# Python {versions['python']}" in requirements
    for key, package in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"), ("joblib", "joblib")):
        assert f"{package}=={versions[key]}\n" in requirements
    promotion = json.loads((output / "promotion.json").read_text())
    assert "not holdout policy winners" in promotion["operational_policy"]
    assert "trusted local pickles" in promotion["trusted_pickle_notice"]
    assert "PRIVATE NOTES" not in json.dumps(promotion)


def test_release_cannot_be_overwritten(evidence):
    run(evidence)
    before = {p: p.read_bytes() for p in evidence["output"].iterdir()}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(evidence)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_empty_destination_supported(evidence):
    evidence["output"].mkdir()
    run(evidence)
    assert (evidence["output"] / "manifest.json").is_file()


@pytest.mark.parametrize("fault,match", [
    ("status", "completed"), ("versions", "Runtime mismatch"), ("dataset", "Dataset SHA"),
    ("overlap", "overlap"), ("invalid_ids", "Invalid training row IDs"),
    ("features", "feature_order"), ("policy", "policy"), ("fit_scope", "fit_scope"),
    ("winner", "CV winner"), ("threshold", "safety-oriented"),
    ("config_versions", "Runtime mismatch"), ("cohort_split", "cohort split"),
])
def test_invalid_json_evidence_fails_before_pickle_load(evidence, monkeypatch, fault, match):
    report = evidence["report"]
    result = report["experiments"][0]
    config = result["selected_configuration"]
    if fault == "status":
        report["status"] = "running"
    elif fault == "versions":
        report["metadata"]["versions"]["numpy"] = "0.0.0"
    elif fault == "dataset":
        evidence["dataset"].write_text(evidence["dataset"].read_text() + "\n")
    elif fault == "overlap":
        config["holdout_row_ids"] = [0, 9, 10, 11]
    elif fault == "invalid_ids":
        config["training_row_ids"] = [999]
    elif fault == "features":
        config["feature_order"] = list(reversed(config["feature_order"]))
    elif fault == "policy":
        config["policy"] = "raw"
    elif fault == "fit_scope":
        config["fit_scope"] = "all rows"
    elif fault == "winner":
        config["selected_model"] = "dummy"
    elif fault == "threshold":
        config["threshold"]["value"] = .4
    elif fault == "config_versions":
        config["versions"] = {**runtime_versions(), "pandas": "0.0.0"}
    elif fault == "cohort_split":
        report["cohorts"]["classification/binary_targets"]["split"]["train_row_ids"] = [0, 1]
    write_report(evidence)
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("Must validate JSON before unpickling"))
    with pytest.raises(ValueError, match=match):
        run(evidence)
    assert not evidence["output"].exists()


def test_embedded_metadata_drift_fails_and_cleans_staging(evidence):
    # Last candidate fails, after three successful staged bundles.
    path = evidence["source"] / "candidates" / f"{SOURCE_IDS['regression_quick']}.joblib"
    bundle = joblib.load(path)
    bundle["metadata"]["feature_order"] = ["TDS", "Hardness", "Turbidity"]
    joblib.dump(bundle, path)
    before = {p: p.read_bytes() for p in evidence["source"].rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Candidate metadata disagrees"):
        run(evidence)
    assert not evidence["output"].exists()
    assert not list(evidence["output"].parent.glob(".phase3-staging-*"))
    assert all(p.read_bytes() == payload for p, payload in before.items())


def test_candidate_symlink_escape_is_rejected(evidence, tmp_path):
    path = evidence["source"] / "candidates" / f"{SOURCE_IDS['classification_full']}.joblib"
    outside = tmp_path / "outside.joblib"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink escapes"):
        run(evidence)
    assert not evidence["output"].exists()


def test_source_destination_overlap_rejected(evidence):
    with pytest.raises(ValueError, match="separate from source"):
        promote(evidence["source"], evidence["dataset"], evidence["source"] / "release")


def test_failed_publication_does_not_leave_partial_release(evidence, monkeypatch):
    def fail(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr("ml.promote.os.rename", fail)
    with pytest.raises(OSError, match="publication failure"):
        run(evidence)
    assert not evidence["output"].exists()
    assert not list(evidence["output"].parent.glob(".phase3-staging-*"))
