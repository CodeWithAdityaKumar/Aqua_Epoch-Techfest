"""Reproducible hybrid classification/regression training pipeline.

The four persisted artifacts produced by this module are deliberately plain
scikit-learn Pipelines.  A later backend can load one with ``joblib.load`` and
call ``bundle["pipeline"].predict(...)`` after applying the recorded schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATASET_FILENAME = (
    "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - "
    "Aqua_Epoch_BIS_Water_Quality_Dataset.csv"
)
TARGET_CLASSIFICATION = "Potability"
TARGET_REGRESSION = "pH"
UI_FEATURES = ["Turbidity", "Hardness", "TDS"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def _as_python(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-compatible Python values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_as_python(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_as_python(dict(payload)), indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def load_and_clean_dataset(
    dataset_path: str | Path,
    target_column: str = TARGET_CLASSIFICATION,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the CSV and return numeric data plus an auditable cleaning summary.

    Cleaning intentionally only removes fully blank records and records whose
    Potability target is missing.  Feature missingness is retained for the
    sklearn imputer in each model Pipeline.
    """

    path = Path(dataset_path)
    raw = pd.read_csv(path)
    if target_column not in raw.columns:
        raise ValueError(f"Required target column {target_column!r} is absent")

    blank_mask = raw.isna().all(axis=1)
    numeric = _numeric_columns(raw)
    missing_target_mask = numeric[target_column].isna()
    keep_mask = ~(blank_mask | missing_target_mask)
    cleaned = numeric.loc[keep_mask].reset_index(drop=True)

    invalid_target = ~cleaned[target_column].isin([0, 1])
    if invalid_target.any():
        bad_values = sorted(cleaned.loc[invalid_target, target_column].unique().tolist())
        raise ValueError(f"Potability must contain only 0/1 after cleaning; found {bad_values}")

    removed_mask = ~keep_mask
    removed_rows = (raw.index[removed_mask] + 2).tolist()  # header is CSV row 1
    summary: Dict[str, Any] = {
        "source_filename": path.name,
        "source_sha256": _sha256(path),
        "target_column": target_column,
        "columns": list(raw.columns),
        "rows_read": int(len(raw)),
        "fully_blank_rows_removed": int(blank_mask.sum()),
        "missing_target_rows_removed": int(missing_target_mask.sum()),
        "missing_target_nonblank_rows_removed": int((missing_target_mask & ~blank_mask).sum()),
        "blank_and_missing_target_overlap": int((blank_mask & missing_target_mask).sum()),
        "rows_removed_total": int(removed_mask.sum()),
        "rows_after_cleaning": int(len(cleaned)),
        "removed_csv_row_numbers_1_indexed": removed_rows,
        "missing_feature_values_after_cleaning": {
            column: int(value) for column, value in cleaned.isna().sum().items() if value
        },
        "target_distribution_after_cleaning": {
            str(key): int(value) for key, value in cleaned[target_column].value_counts().sort_index().items()
        },
    }
    return cleaned, summary


def feature_sets(columns: Iterable[str]) -> Dict[str, Sequence[str]]:
    """Return the required full/UI feature schemas for both tasks."""

    columns = list(columns)
    expected = {TARGET_CLASSIFICATION, TARGET_REGRESSION, *UI_FEATURES}
    missing = sorted(expected.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    classification_full = [column for column in columns if column != TARGET_CLASSIFICATION]
    regression_full = [
        column for column in columns if column not in {TARGET_REGRESSION, TARGET_CLASSIFICATION}
    ]
    return {
        "classification_full": classification_full,
        "classification_ui": list(UI_FEATURES),
        "regression_full": regression_full,
        "regression_ui": list(UI_FEATURES),
    }


CLASSIFICATION_UI_FEATURES = list(UI_FEATURES)
REGRESSION_UI_FEATURES = list(UI_FEATURES)


def build_classifier_pipeline(
    random_state: int = RANDOM_STATE, n_estimators: int = 200, max_depth: int = 18
) -> Pipeline:
    """Build a portable classifier with in-Pipeline missing-value handling."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_regressor_pipeline(
    random_state: int = RANDOM_STATE, n_estimators: int = 200, max_depth: int = 18
) -> Pipeline:
    """Build a portable regressor with in-Pipeline missing-value handling."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _validation_metadata(frame: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Any]:
    bounds: Dict[str, Any] = {}
    for column in columns:
        values = frame[column].dropna()
        bounds[column] = {
            "observed_training_min": float(values.min()) if not values.empty else None,
            "observed_training_max": float(values.max()) if not values.empty else None,
            "required": True,
            "nullable": True,
        }
    return {
        "policy": "advisory_observed_training_range",
        "domain_safety_claim": False,
        "message": (
            "Bounds are observed training-data ranges for input-quality warnings only; "
            "they are not regulatory limits and do not certify water safety or potability."
        ),
        "features": bounds,
    }


def _schema(
    task: str,
    mode: str,
    target: str,
    columns: Sequence[str],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "task": task,
        "mode": mode,
        "target": target,
        "feature_order": list(columns),
        "feature_dtype": "numeric",
        "allow_missing_features": True,
        "preprocessing": {
            "imputer": "SimpleImputer(strategy='median', add_indicator=True)",
            "fit_scope": "training split only",
        },
        "input_validation": dict(validation),
    }


def _classification_metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_labels": [0, 1],
    }


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


def _save_bundle(
    artifact_dir: Path,
    name: str,
    pipeline: Pipeline,
    metadata: Mapping[str, Any],
) -> str:
    artifact_path = artifact_dir / f"{name}.joblib"
    bundle = {"pipeline": pipeline, "metadata": _as_python(dict(metadata))}
    joblib.dump(bundle, artifact_path)
    _write_json(artifact_dir / f"{name}.metadata.json", metadata)
    return artifact_path.name


def train_models(
    dataset_path: str | Path,
    artifact_dir: str | Path = "artifacts",
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    n_estimators: int = 200,
    max_depth: int = 18,
) -> Dict[str, Any]:
    """Train, evaluate, and persist all Phase 1 model variants."""

    dataset_path = Path(dataset_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame, cleaning = load_and_clean_dataset(dataset_path)
    sets = feature_sets(frame.columns)
    validation_all = _validation_metadata(frame, sets["classification_full"])

    classification_models: Dict[str, Any] = {}
    x_class = frame[sets["classification_full"]]
    y_class = frame[TARGET_CLASSIFICATION].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x_class,
        y_class,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class,
    )
    for mode, columns in (
        ("full", sets["classification_full"]),
        ("ui", sets["classification_ui"]),
    ):
        pipeline = build_classifier_pipeline(random_state, n_estimators, max_depth)
        pipeline.fit(x_train[columns], y_train)
        predictions = pipeline.predict(x_test[columns])
        probabilities = pipeline.predict_proba(x_test[columns])[:, 1]
        metrics = _classification_metrics(y_test, predictions, probabilities)
        metadata = {
            "artifact_version": "phase1-1",
            "task": "classification",
            "mode": mode,
            "target": TARGET_CLASSIFICATION,
            "feature_order": list(columns),
            "dataset_sha256": cleaning["source_sha256"],
            "cleaning_summary": cleaning,
            "split": {
                "strategy": "train_test_split_stratified",
                "test_size": test_size,
                "random_state": random_state,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
            },
            "model": "sklearn.ensemble.RandomForestClassifier",
            "model_parameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
            "metrics": metrics,
            "schema": _schema(
                "classification", mode, TARGET_CLASSIFICATION, columns,
                {key: validation_all[key] for key in ("policy", "domain_safety_claim", "message")}
                | {"features": {name: validation_all["features"][name] for name in columns}},
            ),
        }
        classification_models[f"classification_{mode}"] = {
            "pipeline": pipeline,
            "metadata": metadata,
            "artifact": _save_bundle(artifact_dir, f"classification_{mode}", pipeline, metadata),
        }

    regression_models: Dict[str, Any] = {}
    # pH is the regression target. Unlike feature missingness, a missing
    # regression target cannot be imputed without inventing labels, so those
    # rows are excluded from regression only and recorded separately.
    regression_target_missing = frame[TARGET_REGRESSION].isna()
    regression_frame = frame.loc[~regression_target_missing].reset_index(drop=True)
    regression_cleaning = dict(cleaning)
    regression_cleaning.update(
        {
            "regression_target": TARGET_REGRESSION,
            "regression_target_missing_rows_removed": int(regression_target_missing.sum()),
            "regression_rows_after_target_filter": int(len(regression_frame)),
            "regression_target_missing_row_numbers_after_base_cleaning_1_indexed": (
                (frame.index[regression_target_missing] + 1).tolist()
            ),
        }
    )
    x_reg = regression_frame[sets["regression_full"]]
    y_reg = regression_frame[TARGET_REGRESSION].astype(float)
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x_reg,
        y_reg,
        test_size=test_size,
        random_state=random_state,
    )
    for mode, columns in (
        ("full", sets["regression_full"]),
        ("ui", sets["regression_ui"]),
    ):
        pipeline = build_regressor_pipeline(random_state, n_estimators, max_depth)
        pipeline.fit(x_train_r[columns], y_train_r)
        predictions = pipeline.predict(x_test_r[columns])
        metrics = _regression_metrics(y_test_r, predictions)
        metadata = {
            "artifact_version": "phase1-1",
            "task": "regression",
            "mode": mode,
            "target": TARGET_REGRESSION,
            "feature_order": list(columns),
            "dataset_sha256": cleaning["source_sha256"],
            "cleaning_summary": regression_cleaning,
            "split": {
                "strategy": "train_test_split",
                "test_size": test_size,
                "random_state": random_state,
                "train_rows": len(x_train_r),
                "test_rows": len(x_test_r),
            },
            "model": "sklearn.ensemble.RandomForestRegressor",
            "model_parameters": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": 2,
            },
            "metrics": metrics,
            "schema": _schema(
                "regression", mode, TARGET_REGRESSION, columns,
                {key: validation_all[key] for key in ("policy", "domain_safety_claim", "message")}
                | {"features": {name: validation_all["features"][name] for name in columns}},
            ),
        }
        regression_models[f"regression_{mode}"] = {
            "pipeline": pipeline,
            "metadata": metadata,
            "artifact": _save_bundle(artifact_dir, f"regression_{mode}", pipeline, metadata),
        }

    model_manifest = {}
    for name, result in {**classification_models, **regression_models}.items():
        model_manifest[name] = {
            "artifact": result["artifact"],
            "task": result["metadata"]["task"],
            "mode": result["metadata"]["mode"],
            "target": result["metadata"]["target"],
            "feature_order": result["metadata"]["feature_order"],
            "metrics": result["metadata"]["metrics"],
        }

    manifest: Dict[str, Any] = {
        "artifact_version": "phase1-1",
        "pipeline": "hybrid_water_quality_classification_and_regression",
        "dataset": {
            "filename": dataset_path.name,
            "sha256": cleaning["source_sha256"],
        },
        "cleaning_summary": cleaning,
        "regression_cleaning_summary": regression_cleaning,
        "reproducibility": {
            "random_state": random_state,
            "test_size": test_size,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
        "ui_features": list(UI_FEATURES),
        "input_validation": validation_all,
        "models": model_manifest,
    }
    _write_json(
        artifact_dir / "cleaning_summary.json",
        {"base_cleaning": cleaning, "regression_target_filter": regression_cleaning},
    )
    _write_json(artifact_dir / "metrics.json", {name: entry["metrics"] for name, entry in model_manifest.items()})
    _write_json(artifact_dir / "metadata.json", manifest)
    return manifest


def load_artifact(artifact_path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate an artifact bundle for backend use."""

    bundle = joblib.load(artifact_path)
    if not isinstance(bundle, dict) or not {"pipeline", "metadata"}.issubset(bundle):
        raise ValueError(f"Invalid Aqua Epoch model artifact: {artifact_path}")
    return bundle


def validate_input_record(bundle: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate shape/type and return advisory out-of-range warnings.

    This function intentionally does not label an input safe or unsafe.  The
    observed training range is only an input-quality signal.
    """

    metadata = bundle["metadata"]
    schema = metadata.get("schema", metadata)
    feature_order = list(schema.get("feature_order", metadata.get("feature_order", [])))
    validation = schema.get("input_validation", {})
    errors = [f"missing required feature: {name}" for name in feature_order if name not in record]
    non_numeric = []
    warnings = []
    for name in feature_order:
        if name not in record or record[name] is None or (isinstance(record[name], str) and not record[name].strip()):
            continue
        try:
            value = float(record[name])
        except (TypeError, ValueError):
            non_numeric.append(name)
            continue
        bounds = validation.get("features", {}).get(name, {})
        low = bounds.get("observed_training_min")
        high = bounds.get("observed_training_max")
        if low is not None and value < low or high is not None and value > high:
            warnings.append(
                f"{name} is outside observed training range [{low}, {high}]; "
                "this is an advisory input-quality warning, not a safety determination"
            )
    errors.extend(f"non-numeric feature: {name}" for name in non_numeric)
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "domain_safety_claim": False,
        "message": validation.get("message"),
    }


def _resolve_dataset(path_argument: str | None) -> Path:
    if path_argument:
        return Path(path_argument)
    here = Path(__file__).resolve().parents[1]
    return here / DATASET_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Aqua Epoch Phase 1 ML artifacts")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the source CSV")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Artifact output directory")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=18)
    args = parser.parse_args()
    manifest = train_models(
        _resolve_dataset(args.dataset),
        args.artifacts,
        random_state=args.random_state,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    print(json.dumps({"artifacts": manifest["models"], "cleaning": manifest["cleaning_summary"]}, indent=2))


if __name__ == "__main__":
    main()
