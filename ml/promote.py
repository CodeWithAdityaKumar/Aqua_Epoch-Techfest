"""Package trusted local Phase 2 pickles as an immutable competition demo release.

Run ``python -m ml.promote --source reports/phase2_validated --dataset DATA.csv
--output artifacts/phase3``. Only use candidate pickles from a trusted local
evaluation run: hashes establish provenance, not protection from malicious pickle.
No estimator is fitted and no source evidence is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import joblib

from ml.quality import PhysicalSanitizer, load_quality_data, valid_ph_target
from ml.serving_contract import (
    MODEL_SPECS, RUNTIME_PACKAGES, SCHEMA_VERSION, SOURCE_IDS,
    check_runtime, finite_number, validate_metadata,
)

FIT_SCOPE = "Primary training partition only; heldout rows not refitted."
LIMITATIONS = [
    "Competition demonstration only; not regulatory approval, BIS certification, or a drinking-water safety guarantee.",
    "Internal validation only: Phase 1 explored this dataset; no independent external validation.",
    "Probabilities are uncalibrated P(Potability=1), not confidence in the predicted class.",
    "Measurement units and provenance are undocumented; physical sanitization is an operational freshwater assumption.",
    "Reference ranges describe sanitized training inputs, not regulatory limits; positive outliers remain.",
    "CV and OOF threshold selection add uncertainty; descriptive holdout and raw-target metrics are not performance claims.",
    "Physical-valid pH target filtering changes the cohort and does not establish measurement truth.",
]


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_file(source, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source path traversal is forbidden")
    path = source / relative
    if not path.resolve().is_relative_to(source):
        raise ValueError("Source symlink escapes the evidence directory")
    if not path.is_file():
        raise ValueError(f"Missing source evidence: {relative}")
    return path


def _row_ids(ids, frame, label):
    if (not isinstance(ids, list) or not ids
            or any(type(row) is not int for row in ids)
            or len(set(ids)) != len(ids)
            or not set(ids).issubset(frame.index)):
        raise ValueError(f"Invalid {label} row IDs")
    return set(ids)


def _validate_configuration(result, model_id, report, frame, dataset_hash):
    task, mode, features = MODEL_SPECS[model_id]
    cohort = "binary_targets" if task == "classification" else "physical_valid_targets"
    expected = {"task": task, "mode": mode, "policy": "physical_sanitized", "cohort": cohort}
    config = result.get("selected_configuration")
    if not isinstance(config, dict):
        raise ValueError("Missing selected configuration")
    for field, value in expected.items():
        if result.get(field) != value or config.get(field) != value:
            raise ValueError(f"Incompatible source configuration: {field}")
    required = {
        "experiment_id": SOURCE_IDS[model_id], "feature_order": features,
        "fit_scope": FIT_SCOPE, "dataset_sha256": dataset_hash,
        "target_imputed": False, "phase": 2, "experimental": True, "approved": False,
    }
    for field, value in required.items():
        if field not in config or config[field] != value:
            raise ValueError(f"Incompatible source configuration: {field}")
    check_runtime(config.get("versions"))
    if config["versions"] != report["metadata"]["versions"]:
        raise ValueError("Source configuration versions differ from results")
    train = _row_ids(config.get("training_row_ids"), frame, "training")
    test = _row_ids(config.get("holdout_row_ids"), frame, "holdout")
    if train & test:
        raise ValueError("Training and holdout row IDs overlap")
    if result.get("train_count") != len(train) or result.get("holdout_count") != len(test):
        raise ValueError("Source row counts disagree")
    split = report.get("cohorts", {}).get(f"{task}/{cohort}", {}).get("split", {})
    if (split.get("train_row_ids") != config["training_row_ids"]
            or split.get("test_row_ids") != config["holdout_row_ids"]):
        raise ValueError("Source configuration disagrees with cohort split")
    if task == "regression" and not valid_ph_target(frame.loc[sorted(train | test)]).all():
        raise ValueError("Physical-valid regression split contains invalid targets")
    key = "f1" if task == "classification" else "rmse"
    candidates = result.get("cv_candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Missing training CV candidate evidence")
    scores = {name: candidate.get("mean_std", {}).get(key, {}).get("mean")
              for name, candidate in candidates.items()}
    if not all(finite_number(score) for score in scores.values()):
        raise ValueError("Invalid training CV selection scores")
    winner = (max if task == "classification" else min)(scores, key=scores.get)
    if config.get("selected_model") != winner or config.get("selection_metric") != f"training_cv_{key}":
        raise ValueError("Source selected model is not the training CV winner")
    threshold = config.get("threshold")
    if "threshold" not in config or result.get("threshold") != threshold:
        raise ValueError("Source threshold evidence disagrees")
    if task == "classification":
        if not isinstance(threshold, dict) or threshold.get("training_row_ids") != config["training_row_ids"]:
            raise ValueError("Threshold must come from primary training rows")
        value = threshold.get("value")
        if not finite_number(value) or not .5 <= value <= float.fromhex("0x1.0000000000001p+0"):
            raise ValueError("Invalid safety-oriented source threshold")
        if config.get("probability_semantics") != "P(Potability=1), NOT confidence in the predicted class":
            raise ValueError("Invalid original probability semantics")
    elif threshold is not None:
        raise ValueError("Regression source must not have a threshold")
    return config


def _reference_ranges(frame, config):
    training = frame.loc[config["training_row_ids"], config["feature_order"]]
    # This new stateless sanitizer records a schema only; the fitted pipeline is untouched.
    sanitized = PhysicalSanitizer().fit_transform(training)
    ranges = {}
    for feature in sanitized:
        values = sanitized[feature].dropna()
        ranges[feature] = (dict.fromkeys(("min", "q01", "median", "q99", "max"))
                           if values.empty else dict(zip(
                               ("min", "q01", "median", "q99", "max"),
                               map(float, values.quantile([0, .01, .5, .99, 1])))))
    return ranges


def _evaluation(result, raw_result=None):
    evidence = {
        "interpretation": "Contextual internal validation, not independent external performance or safety claims.",
        "selection_metric": result["selected_configuration"]["selection_metric"],
        "selected_model": result["selected_configuration"]["selected_model"],
        "cv_candidates": result["cv_candidates"],
        "selected_internal_holdout": result["selected_internal_holdout"],
        "dummy_internal_holdout": result["dummy_internal_holdout"],
        "threshold_selection": result["threshold"],
    }
    if raw_result is not None:
        evidence["raw_regression_cohort"] = {
            "source_experiment_id": raw_result["id"],
            "interpretation": "Separate raw-target cohort and independently selected model/split; not a paired comparison or deployed model evaluation.",
            "train_count": raw_result["train_count"], "holdout_count": raw_result["holdout_count"],
            "selected_model": raw_result["selected_configuration"]["selected_model"],
            "cv_candidates": raw_result["cv_candidates"],
            "selected_internal_holdout": raw_result["selected_internal_holdout"],
            "dummy_internal_holdout": raw_result["dummy_internal_holdout"],
        }
    return evidence


def promote(source, dataset, output):
    """Validate evidence before loading trusted candidates, then publish once."""
    source, dataset, output = Path(source).resolve(), Path(dataset).resolve(), Path(output)
    if output.is_symlink():
        raise ValueError("Release destination must not be a symlink")
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Release destination must be separate from source evidence")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be empty; refusing to overwrite {output}")
    results_bytes = _source_file(source, "results.json").read_bytes()
    report = json.loads(results_bytes)
    if report.get("status") != "completed":
        raise ValueError("Source results status must be completed")
    common = report.get("metadata", {})
    check_runtime(common.get("versions"))
    dataset_hash = _sha256(dataset.read_bytes())
    if common.get("dataset_sha256") != dataset_hash:
        raise ValueError("Dataset SHA-256 mismatch")
    frame, audit = load_quality_data(dataset)
    if audit["source_sha256"] != dataset_hash:
        raise ValueError("Dataset changed during promotion")
    experiments = report.get("experiments", [])
    by_id = {}
    for result in experiments:
        identifier = result.get("id")
        if not isinstance(identifier, str) or identifier in by_id:
            raise ValueError("Invalid or duplicate source experiment ID")
        by_id[identifier] = result
    configs = {}
    for model_id, source_id in SOURCE_IDS.items():
        if source_id not in by_id:
            raise ValueError(f"Missing source experiment: {source_id}")
        configs[model_id] = _validate_configuration(by_id[source_id], model_id, report, frame, dataset_hash)
        task, mode, _ = MODEL_SPECS[model_id]
        if task == "regression" and f"regression_raw_targets_{mode}_physical_sanitized" not in by_id:
            raise ValueError("Missing contextual raw-regression cohort evidence")
    # All JSON evidence and original runtime versions pass before any pickle load.
    release_id = str(uuid.uuid4())
    manifest = {"schema_version": SCHEMA_VERSION, "release_id": release_id,
                "intended_use": "competition_demo_only", "models": {}}
    promotion = {
        "schema_version": SCHEMA_VERSION, "release_id": release_id, "status": "demo_only",
        "intended_use": "competition_demo_only", "source_results_sha256": _sha256(results_bytes),
        "dataset_sha256": dataset_hash,
        "operational_policy": "Physical-sanitized full/quick classification and physical-valid-target physical-sanitized full/quick regression are prespecified operational choices, not holdout policy winners.",
        "model_selection": "Preserve each Phase 2 primary-training CV winner and training-only OOF classifier threshold. No holdout selection, recalibration, or estimator refitting.",
        "trusted_pickle_notice": "Load only trusted local pickles. Hashes verify release consistency, not pickle safety or source authenticity.",
        "limitations": LIMITATIONS, "models": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        for model_id, config in configs.items():
            source_id = SOURCE_IDS[model_id]
            relative = Path("candidates") / f"{source_id}.joblib"
            candidate_bytes = _source_file(source, relative).read_bytes()
            candidate = joblib.load(io.BytesIO(candidate_bytes))
            if not isinstance(candidate, dict) or set(candidate) != {"pipeline", "metadata"}:
                raise ValueError("Invalid original candidate bundle")
            # JSON canonicalization handles tuples represented as arrays in Phase 2 results.
            if json.dumps(candidate["metadata"], sort_keys=True, allow_nan=False) != json.dumps(config, sort_keys=True, allow_nan=False):
                raise ValueError("Candidate metadata disagrees with source results configuration")
            pipeline = candidate["pipeline"]
            sanitizer = getattr(pipeline, "named_steps", {}).get("sanitizer")
            if (not isinstance(sanitizer, PhysicalSanitizer)
                    or list(getattr(pipeline, "feature_names_in_", [])) != config["feature_order"]):
                raise ValueError("Candidate fitted pipeline schema/policy is incompatible")
            task, mode, features = MODEL_SPECS[model_id]
            if task == "classification" and list(getattr(pipeline, "classes_", [])) != [0, 1]:
                raise ValueError("Candidate classifier must have binary Potability classes")
            result = by_id[source_id]
            raw = by_id.get(f"regression_raw_targets_{mode}_physical_sanitized") if task == "regression" else None
            metadata = {
                "schema_version": SCHEMA_VERSION, "model_id": model_id, "release_id": release_id,
                "task": task, "mode": mode, "feature_order": features,
                "policy": "physical_sanitized", "fit_scope": "primary_training_only",
                "status": "demo_only", "calibrated": False,
                "source_experiment_id": source_id, "target_cohort": config["cohort"],
                "training_versions": config["versions"], "dataset_sha256": dataset_hash,
                "source_candidate_sha256": _sha256(candidate_bytes), "source_results_sha256": _sha256(results_bytes),
                "threshold": config["threshold"]["value"] if task == "classification" else None,
                "probability_semantics": "P(Potability=1)" if task == "classification" else None,
                "reference_ranges": _reference_ranges(frame, config),
                "evaluation": _evaluation(result, raw), "limitations": LIMITATIONS,
                "training_row_ids": config["training_row_ids"], "holdout_row_ids": config["holdout_row_ids"],
            }
            validate_metadata(metadata, model_id)
            filename = f"{model_id}.joblib"
            joblib.dump({"pipeline": pipeline, "metadata": metadata}, stage / filename)
            manifest["models"][model_id] = {
                "file": filename, "sha256": _sha256((stage / filename).read_bytes()), "metadata": metadata,
            }
            promotion["models"][model_id] = {
                "source_experiment_id": source_id, "source_candidate": relative.as_posix(),
                "source_results": "results.json", "source_candidate_sha256": metadata["source_candidate_sha256"],
                "source_results_sha256": metadata["source_results_sha256"],
                "original_feature_order": config["feature_order"],
                "train_count": len(config["training_row_ids"]), "holdout_count": len(config["holdout_row_ids"]),
                "selected_model": config["selected_model"], "selection_metric": config["selection_metric"],
                "threshold": metadata["threshold"], "evaluation": metadata["evaluation"],
            }
        _write_json(stage / "promotion.json", promotion)
        versions = common["versions"]
        requirements = [f"# Python {versions['python']} (matching major/minor required)"]
        requirements.extend(f"{package}=={versions[key]}" for key, package in RUNTIME_PACKAGES.items())
        (stage / "requirements-serving.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
        # The manifest is the final staged file. Same-filesystem rename publishes
        # the complete release; rename refuses to replace a nonempty directory.
        _write_json(stage / "manifest.json", manifest)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/phase2_validated"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase3"))
    args = parser.parse_args(argv)
    manifest = promote(args.source, args.dataset, args.output)
    print(f"Published demo-only release {manifest['release_id']} to {args.output}")
    return manifest


if __name__ == "__main__":
    main()
