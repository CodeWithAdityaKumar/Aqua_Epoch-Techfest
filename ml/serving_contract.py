"""Versioned shared contract for trusted, local demo model releases.

No pickle is loaded by this module. Bounds describe sanitized training inputs,
not regulatory limits. Runtime compatibility is intentionally fail-closed.
"""
from __future__ import annotations

import math
import platform
from importlib.metadata import version

from ml.quality import FULL_FEATURES, QUICK_FEATURES, REGRESSION_FEATURES

SCHEMA_VERSION = "aqua-epoch-serving-v1"
MODEL_SPECS = {
    "classification_full": ("classification", "full", list(FULL_FEATURES)),
    "classification_quick": ("classification", "quick", list(QUICK_FEATURES)),
    "regression_full": ("regression", "full", list(REGRESSION_FEATURES)),
    "regression_quick": ("regression", "quick", list(QUICK_FEATURES)),
}
SOURCE_IDS = {
    model_id: f"{task}_{'binary_targets' if task == 'classification' else 'physical_valid_targets'}_{mode}_physical_sanitized"
    for model_id, (task, mode, _) in MODEL_SPECS.items()
}
RUNTIME_PACKAGES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "joblib": "joblib"}


def runtime_versions():
    return {"python": platform.python_version(), **{k: version(v) for k, v in RUNTIME_PACKAGES.items()}}


def check_runtime(expected):
    actual = runtime_versions()
    if not isinstance(expected, dict) or any(k not in expected for k in actual):
        raise ValueError("Incomplete training runtime versions")
    for name in RUNTIME_PACKAGES:
        if expected[name] != actual[name]:
            raise ValueError(f"Runtime mismatch for {name}; recreate the recorded environment or retrain and promote")
    if str(expected["python"]).split(".")[:2] != actual["python"].split(".")[:2]:
        raise ValueError("Python major/minor version mismatch")


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA-256 in release metadata")


def validate_metadata(meta, model_id, *, check_versions=True):
    if model_id not in MODEL_SPECS or not isinstance(meta, dict):
        raise ValueError("Unsupported model metadata")
    task, mode, features = MODEL_SPECS[model_id]
    required = {
        "schema_version": SCHEMA_VERSION, "model_id": model_id,
        "task": task, "mode": mode, "feature_order": features,
        "policy": "physical_sanitized", "fit_scope": "primary_training_only",
        "status": "demo_only", "calibrated": False,
        "source_experiment_id": SOURCE_IDS[model_id],
        "target_cohort": "binary_targets" if task == "classification" else "physical_valid_targets",
    }
    for field, expected in required.items():
        if field not in meta or meta[field] != expected:
            raise ValueError(f"Invalid serving metadata field: {field}")
    if meta["calibrated"] is not False:
        raise ValueError("Calibrated must explicitly be false")
    for field in ("dataset_sha256", "source_candidate_sha256", "source_results_sha256"):
        check_hash(meta.get(field))
    if not isinstance(meta.get("release_id"), str) or not meta["release_id"].strip():
        raise ValueError("Missing release identifier")
    if task == "classification":
        if not finite_number(meta.get("threshold")) or not .5 <= meta["threshold"] <= math.nextafter(1., math.inf):
            raise ValueError("Invalid safety-oriented threshold")
        if meta.get("probability_semantics") != "P(Potability=1)":
            raise ValueError("Probability semantics must be P(Potability=1)")
    elif meta.get("threshold") is not None:
        raise ValueError("Regression has no classification threshold")
    ranges = meta.get("reference_ranges")
    if not isinstance(ranges, dict) or set(ranges) != set(features):
        raise ValueError("Reference ranges must match exact feature schema")
    for feature, bounds in ranges.items():
        if not isinstance(bounds, dict):
            raise ValueError(f"Invalid reference bounds for {feature}")
        values = [bounds.get(key) for key in ("min", "q01", "median", "q99", "max")]
        if all(v is None for v in values):
            continue
        if not all(finite_number(v) for v in values) or values != sorted(values):
            raise ValueError(f"Invalid reference bounds for {feature}")
    if not isinstance(meta.get("evaluation"), dict) or not isinstance(meta.get("limitations"), list):
        raise ValueError("Missing evaluation evidence/limitations")
    if check_versions:
        check_runtime(meta.get("training_versions"))
    return meta
