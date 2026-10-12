"""Bounded Phase 2 experiments; all results are experimental internal validation.

Run with ``python -m ml.evaluate --dataset DATA.csv --output reports/phase2``.
The output directory must be empty. No production or Phase 1 artifacts are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from ml.quality import (
    FULL_FEATURES, QUICK_FEATURES, REGRESSION_FEATURES, PhysicalSanitizer,
    load_quality_data, valid_ph_target,
)

SEED = 2026
POLICIES = ("raw", "physical_sanitized")


def features_for(task, mode):
    features = list(QUICK_FEATURES if mode == "quick" else
                    FULL_FEATURES if task == "classification" else REGRESSION_FEATURES)
    forbidden = {"Potability"} if task == "classification" else {"Potability", "pH"}
    if forbidden.intersection(features):
        raise ValueError(f"Target leakage in {task} feature schema")
    return features


def build_pipeline(task, name, policy, trees=80, seed=SEED):
    """Every fit (including CV/stress fits) learns its own imputation statistics."""
    if task == "classification":
        models = {
            "dummy": DummyClassifier(strategy="prior"),
            "linear": LogisticRegression(max_iter=1500, random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=trees, max_depth=14, min_samples_leaf=3,
                n_jobs=2, random_state=seed),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, max_leaf_nodes=31, random_state=seed),
        }
    else:
        models = {
            "dummy": DummyRegressor(strategy="mean"),
            "linear": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=trees, max_depth=14, min_samples_leaf=3,
                n_jobs=2, random_state=seed),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=100, max_depth=8, max_leaf_nodes=31, random_state=seed),
        }
    # Raw keeps finite, physically implausible measurements, but infinities
    # cannot be consumed by sklearn's imputer and are treated as missing.
    sanitizer = (PhysicalSanitizer() if policy == "physical_sanitized" else
                 FunctionTransformer(pd.DataFrame.replace,
                                     kw_args={"to_replace": [np.inf, -np.inf], "value": np.nan}))
    steps = [("sanitizer", sanitizer),
             ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                       keep_empty_features=True))]
    if name == "linear":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps + [("model", models[name])])


def prepare_cohort(frame, task, cohort):
    """One shared row universe across modes/policies, excluding ambiguous groups.

    Check every model's original-column representation before splitting. This
    deliberately conservative rule also prevents quick/sanitized duplicates
    from straddling partitions. Target values are never imputed.
    """
    if not frame.index.is_unique:
        raise ValueError("Original row IDs must be unique")
    target = "Potability" if task == "classification" else "pH"
    eligible = (frame[target].isin([0, 1]) if task == "classification" else
                valid_ph_target(frame) if cohort == "physical_valid_targets" else
                np.isfinite(frame[target]))
    work = frame.loc[eligible].copy()
    views = {}
    conflicts = set()
    details = {}
    for mode in ("full", "quick"):
        for policy in POLICIES:
            key = f"{mode}/{policy}"
            view = work[features_for(task, mode)]
            view = (PhysicalSanitizer().fit_transform(view) if policy == "physical_sanitized"
                    else view.replace([np.inf, -np.inf], np.nan))
            views[key] = view
            grouped = view.assign(_target=work[target]).groupby(
                list(view.columns), dropna=False, sort=False)["_target"]
            bad = grouped.transform("nunique") > 1
            ids = work.index[bad].tolist()
            conflicts.update(ids)
            details[key] = {"conflicting_row_ids": ids,
                            "conflicting_groups": int((grouped.nunique() > 1).sum())}
    remaining = work.index[~work.index.isin(conflicts)]
    duplicate_ids = []
    for key, view in views.items():
        duplicates = view.loc[remaining].duplicated(keep="first")
        removed = remaining[duplicates].tolist()
        details[key]["duplicate_row_ids_removed"] = removed
        duplicate_ids.extend(removed)
        remaining = remaining[~duplicates]
    result = work.loc[remaining].sort_index()
    return result, {
        "target": target, "cohort": cohort, "input_count": len(frame),
        "ineligible_target_row_ids": frame.index[~eligible].tolist(),
        "conflicting_row_ids_removed": sorted(conflicts),
        "conflicting_rows_removed_count": len(conflicts),
        "duplicate_row_ids_removed": duplicate_ids,
        "duplicate_rows_removed_count": len(duplicate_ids),
        "retained_row_ids": result.index.tolist(), "retained_count": len(result),
        "group_audit": details,
        "deduplication": "Exclude conflicts in any mode/policy; keep first exact feature representative sequentially across all views, before split.",
    }


def make_split(frame, task, seed=SEED, block=False):
    ids = frame.index.to_numpy()
    if len(ids) < 10:
        raise ValueError("At least 10 deduplicated eligible rows are required")
    if block:
        boundary = int(len(ids) * .8)
        train, test = ids[:boundary], ids[boundary:]
    else:
        train, test = train_test_split(
            ids, test_size=.2, random_state=seed,
            stratify=frame["Potability"] if task == "classification" else None)
    return {"train_row_ids": train.tolist(), "test_row_ids": test.tolist(),
            "seed": None if block else seed, "kind": "source_order_80_20" if block else "random_80_20",
            "interpretation": "Internal holdout; NOT unseen external data. Phase 1 explored this dataset."}


def wilson_interval(successes, trials):
    if not trials:
        return [None, None]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def classification_metrics(y, probability, threshold=.5):
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else None,
        "brier_score": brier_score_loss(y, probability),
        "safe_precision": tp / (tp + fp) if tp + fp else 0.,
        "safe_recall": tp / (tp + fn) if tp + fn else None,
        "unsafe_recall": tn / (tn + fp) if tn + fp else None,
        "unsafe_to_safe_fpr": fp / (tn + fp) if tn + fp else None,
        "unsafe_fpr_wilson_95": wilson_interval(int(fp), int(tn + fp)),
        "safe_prediction_coverage": float(predicted.mean()),
        "confusion_matrix": {"actual_unsafe_predicted_unsafe": int(tn),
                             "actual_unsafe_predicted_safe": int(fp),
                             "actual_safe_predicted_unsafe": int(fn),
                             "actual_safe_predicted_safe": int(tp)},
    }


def choose_threshold(y, probability, max_fpr=.02):
    """Maximize safe recall at threshold >= .5 under the empirical OOF FPR cap.

    A safety-oriented operating point must not increase safe calls relative to
    the default operating point merely to spend an available error budget.
    """
    y, probability = np.asarray(y), np.asarray(probability)
    fallback = float(np.nextafter(1., np.inf))
    if len(np.unique(y)) < 2:
        return fallback
    fpr, recall, thresholds = roc_curve(y, probability, drop_intermediate=False)
    feasible = (fpr <= max_fpr) & np.isfinite(thresholds) & (thresholds <= 1) & (thresholds >= .5)
    if classification_metrics(y, probability, .5)["unsafe_to_safe_fpr"] <= max_fpr:
        return .5
    if not feasible.any() or recall[feasible].max() == 0:
        return fallback
    best_recall = recall[feasible].max()
    return float(thresholds[feasible & (recall == best_recall)].max())


def regression_metrics(y, predicted):
    return {"rmse": float(np.sqrt(mean_squared_error(y, predicted))),
            "mae": float(mean_absolute_error(y, predicted)),
            "r2": float(r2_score(y, predicted)) if len(y) > 1 else None}


def safe_probability(pipeline, X):
    classes = list(pipeline.classes_)
    probabilities = pipeline.predict_proba(X)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def cv_predictions(pipeline, X, y, task, folds, seed=SEED):
    cv = (StratifiedKFold(folds, shuffle=True, random_state=seed) if task == "classification"
          else KFold(folds, shuffle=True, random_state=seed))
    oof = np.full(len(y), np.nan)
    metrics, fold_ids = [], []
    for train, validation in cv.split(X, y):
        fitted = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        predicted = (safe_probability(fitted, X.iloc[validation]) if task == "classification"
                     else fitted.predict(X.iloc[validation]))
        oof[validation] = predicted
        metrics.append(classification_metrics(y.iloc[validation], predicted) if task == "classification"
                       else regression_metrics(y.iloc[validation], predicted))
        fold_ids.append({"train_row_ids": X.index[train].tolist(),
                         "validation_row_ids": X.index[validation].tolist()})
    scalar_keys = ("f1", "roc_auc", "brier_score", "safe_precision", "safe_recall",
                   "unsafe_recall", "unsafe_to_safe_fpr", "safe_prediction_coverage") if task == "classification" else ("rmse", "mae", "r2")
    summary = {key: {"mean": float(np.mean([m[key] for m in metrics if m[key] is not None])),
                     "std": float(np.std([m[key] for m in metrics if m[key] is not None]))}
               for key in scalar_keys}
    return oof, {"fold_metrics": metrics, "mean_std": summary}, fold_ids


def _evaluate_fitted(pipeline, X, y, task, threshold=.5):
    if task == "classification":
        probability = safe_probability(pipeline, X)
        return {"baseline_threshold_0_5": classification_metrics(y, probability),
                "conservative_threshold": classification_metrics(y, probability, threshold)}
    return regression_metrics(y, pipeline.predict(X))


def _skill(selected, baseline):
    denominator = baseline["rmse"]
    return {"rmse_skill_vs_dummy": 1 - selected["rmse"] / denominator if denominator else None,
            "better_than_dummy_rmse": selected["rmse"] < denominator,
            "interpretation": ("Better than mean dummy on this split." if selected["rmse"] < denominator
                               else "NOT better than mean dummy on this split.")}


def _threshold_record(y, oof, row_ids):
    threshold = choose_threshold(y, oof)
    return {"value": threshold, "baseline": .5,
            "provenance": "Selected configuration, training-only out-of-fold P(Potability=1); no test labels used.",
            "training_row_ids": list(row_ids), "empirical_unsafe_fpr_cap": .02,
            "rule": "Keep .5 if OOF FPR <= .02; otherwise maximize safe recall at threshold >= .5 under that cap; stricter threshold wins ties; >1 means no safe predictions.",
            "oof_baseline_metrics": classification_metrics(y, oof),
            "oof_threshold_metrics": classification_metrics(y, oof, threshold),
            "caveat": "Empirical constraint, not a safety guarantee. Wilson interval is descriptive; model/threshold selection adds uncertainty."}


def _stress_test(frame, task, features, selected_pipeline, folds, split):
    target = "Potability" if task == "classification" else "pH"
    train, test = split["train_row_ids"], split["test_row_ids"]
    X_train, y_train = frame.loc[train, features], frame.loc[train, target]
    X_test, y_test = frame.loc[test, features], frame.loc[test, target]
    record = {"split": split, "configuration_selection": "Fixed from primary training CV; no stress-test configuration selection."}
    threshold = .5
    if task == "classification":
        counts = y_train.value_counts()
        if len(counts) < 2 or counts.min() < folds:
            record["skipped"] = "Training partition has insufficient class examples for training-only threshold CV."
            return record
        oof, _, fold_ids = cv_predictions(selected_pipeline, X_train, y_train, task, folds)
        provenance = _threshold_record(y_train, oof, train)
        threshold = provenance["value"]
        record.update(threshold=provenance, threshold_cv_folds=fold_ids)
    fitted = clone(selected_pipeline).fit(X_train, y_train)
    record["selected_metrics"] = _evaluate_fitted(fitted, X_test, y_test, task, threshold)
    # Dummy comparisons do not compete for selection on stress holdouts.
    dummy = clone(selected_pipeline).set_params(model=(DummyClassifier(strategy="prior")
                     if task == "classification" else DummyRegressor(strategy="mean")))
    dummy.fit(X_train, y_train)
    record["dummy_metrics"] = _evaluate_fitted(dummy, X_test, y_test, task, threshold)
    if task == "regression":
        record.update(_skill(record["selected_metrics"], record["dummy_metrics"]))
    return record


def run_experiment(frame, task, cohort, mode, policy, split, args, common_metadata):
    started = time.perf_counter()
    features = features_for(task, mode)
    target = "Potability" if task == "classification" else "pH"
    train, test = split["train_row_ids"], split["test_row_ids"]
    X_train, y_train = frame.loc[train, features], frame.loc[train, target]
    X_test, y_test = frame.loc[test, features], frame.loc[test, target]
    if task == "classification" and (y_train.nunique() != 2 or y_train.value_counts().min() < args.folds):
        raise ValueError("Classification training data needs both classes with at least --folds rows each")
    candidates, predictions, pipelines = {}, {}, {}
    fold_ids = None
    for name in ("dummy", "linear", "random_forest", "hist_gradient_boosting"):
        pipeline = build_pipeline(task, name, policy, args.trees)
        oof, cv, fold_ids = cv_predictions(pipeline, X_train, y_train, task, args.folds)
        candidates[name] = cv
        predictions[name] = oof
        pipelines[name] = pipeline
    key = "f1" if task == "classification" else "rmse"
    selected_name = (max if task == "classification" else min)(
        candidates, key=lambda name: candidates[name]["mean_std"][key]["mean"])
    selected = pipelines[selected_name]
    threshold_record = (_threshold_record(y_train, predictions[selected_name], train)
                        if task == "classification" else None)
    threshold = threshold_record["value"] if threshold_record else .5
    # Only the CV winner and prespecified dummy ever see the primary holdout.
    selected.fit(X_train, y_train)
    baseline = pipelines["dummy"] if selected_name != "dummy" else selected
    if selected_name != "dummy":
        baseline.fit(X_train, y_train)
    selected_metrics = _evaluate_fitted(selected, X_test, y_test, task, threshold)
    dummy_metrics = _evaluate_fitted(baseline, X_test, y_test, task, threshold)
    identifier = f"{task}_{cohort}_{mode}_{policy}"
    metadata = {
        **common_metadata, "experiment_id": identifier, "task": task, "cohort": cohort,
        "mode": mode, "policy": policy, "feature_order": features,
        "selected_model": selected_name, "selection_metric": f"training_cv_{key}",
        "model_parameters": selected.named_steps["model"].get_params(),
        "threshold": threshold_record,
        "probability_semantics": "P(Potability=1), NOT confidence in the predicted class" if task == "classification" else None,
        "fit_scope": "Primary training partition only; heldout rows not refitted.",
        "training_row_ids": train, "holdout_row_ids": test,
        "target_imputed": False,
    }
    bundle_path = Path(args.output) / "candidates" / f"{identifier}.joblib"
    joblib.dump({"pipeline": selected, "metadata": metadata}, bundle_path)
    importance = permutation_importance(
        selected, X_test, y_test, n_repeats=args.permutation_repeats,
        random_state=SEED, n_jobs=1,
        scoring="roc_auc" if task == "classification" else "neg_root_mean_squared_error")
    result = {
        "id": identifier, "task": task, "cohort": cohort, "mode": mode, "policy": policy,
        "train_count": len(train), "holdout_count": len(test),
        "cv_candidates": candidates, "cv_folds": fold_ids,
        "selected_configuration": metadata,
        "threshold": threshold_record,
        "selected_internal_holdout": selected_metrics,
        "dummy_internal_holdout": dummy_metrics,
        "permutation_importance": {
            "scope": "Primary internal holdout, original input feature columns; descriptive, not used for selection.",
            "scoring": "roc_auc" if task == "classification" else "neg_root_mean_squared_error",
            "repeats": args.permutation_repeats,
            "features": {feature: {"mean": float(mean), "std": float(std)}
                         for feature, mean, std in zip(features, importance.importances_mean,
                                                       importance.importances_std)},
        },
        "candidate_bundle": str(bundle_path),
        "stress_tests": {},
    }
    if task == "regression":
        result.update(_skill(selected_metrics, dummy_metrics))
    for label, stress_split in (
        ("source_order", make_split(frame, task, block=True)),
        ("seed_7", make_split(frame, task, seed=7)),
        ("seed_91", make_split(frame, task, seed=91)),
    ):
        result["stress_tests"][label] = _stress_test(
            frame, task, features, pipelines[selected_name], args.folds, stress_split)
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n")


def _comparison_rows(result):
    selected = result["selected_configuration"]["selected_model"]
    for name, candidate in result["cv_candidates"].items():
        row = {key: result[key] for key in ("id", "task", "cohort", "mode", "policy")}
        row.update(candidate=name, selected=name == selected)
        for metric, statistics in candidate["mean_std"].items():
            for stat, value in statistics.items():
                row[f"cv_{metric}_{stat}"] = value
        # Nonwinning candidates deliberately have no holdout scores.
        if name == selected or name == "dummy":
            metrics = result["selected_internal_holdout" if name == selected else "dummy_internal_holdout"]
            if result["task"] == "classification":
                for operating_point, scores in metrics.items():
                    for metric, value in scores.items():
                        if isinstance(value, (int, float)):
                            row[f"internal_holdout_{operating_point}_{metric}"] = value
            else:
                row.update({f"internal_holdout_{metric}": value for metric, value in metrics.items()})
                if name == selected:
                    row.update({key: result[key] for key in ("rmse_skill_vs_dummy", "better_than_dummy_rmse")})
        yield row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/phase2"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=80)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    for name, lower, upper in (("folds", 2, 5), ("trees", 1, 200), ("permutation_repeats", 1, 10)):
        if not lower <= getattr(args, name) <= upper:
            parser.error(f"--{name.replace('_', '-')} must be between {lower} and {upper}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if "artifacts" in args.output.resolve().parts:
        raise ValueError("Phase 2 output must not be inside an artifacts directory")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be empty; refusing to overwrite {args.output}")
    frame, audit = load_quality_data(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidates").mkdir()
    started = time.perf_counter()
    metadata = {
        "phase": 2, "experimental": True, "approved": False,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "seed": SEED,
        "validation": "Internal holdout, NOT unseen external data; Phase 1 explored the dataset.",
    }
    report = {
        "status": "running",
        "metadata": metadata, "runtime_config": vars(args), "data_audit": audit,
        "protocol": {
            "selection": "Each mode/policy/cohort independently selects on training-only CV; classification max mean F1 at .5, regression min mean RMSE. No policy winner is chosen on holdout.",
            "calibration": "Brier score assesses probability error; no claim that probabilities are calibrated.",
            "target_cohorts": "Raw regression targets are finite pH; physical-valid targets additionally lie in [0,14]. Cohorts have independent task-specific splits and may differ in size; within each cohort every mode and feature policy shares identical rows.",
            "uncertainty": "CV standard deviation is across folds, not a confidence interval. FPR Wilson intervals are descriptive, not safety guarantees.",
            "stress": "Configuration fixed from primary CV; each stress threshold refit on that stress training partition OOF only. Stress is not independent external validation.",
            "limits": "No external validation or model promotion. Source order may not represent time/source. Deduplication across quick and sanitized views conservatively removes rows and changes population. OOF selection/threshold estimates are optimistic without nested CV. Physical-valid target filtering cannot establish measurement truth.",
            "cpu": "threadpool limit 2; RF n_jobs=2; permutation n_jobs=1; sequential experiments",
        },
        "cohorts": {}, "experiments": [],
    }
    _write_json(args.output / "runtime_config.json", {**metadata, **vars(args)})
    with threadpool_limits(limits=2):
        for task, cohort in (("classification", "binary_targets"),
                             ("regression", "raw_targets"),
                             ("regression", "physical_valid_targets")):
            data, cohort_audit = prepare_cohort(frame, task, cohort)
            split = make_split(data, task)
            report["cohorts"][f"{task}/{cohort}"] = {"audit": cohort_audit, "split": split}
            _write_json(args.output / "split_indices.json", report["cohorts"])
            for mode in ("full", "quick"):
                for policy in POLICIES:
                    print(f"Evaluating {task}/{cohort}/{mode}/{policy}", flush=True)
                    result = run_experiment(data, task, cohort, mode, policy, split, args, metadata)
                    report["experiments"].append(result)
                    report["runtime_seconds"] = time.perf_counter() - started
                    _write_json(args.output / "results.json", report)
                    pd.DataFrame([row for result in report["experiments"]
                                  for row in _comparison_rows(result)]).to_csv(args.output / "comparison.csv", index=False)
    report["status"] = "completed"
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(args.output / "results.json", report)
    print(f"Wrote {len(report['experiments'])} experimental comparisons to {args.output}", flush=True)
    return report


if __name__ == "__main__":
    main()
