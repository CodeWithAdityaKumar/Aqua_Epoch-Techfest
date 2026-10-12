"""Behavioral checks for Phase 2 split isolation and safety reporting."""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from ml.evaluate import (
    build_pipeline, choose_threshold, classification_metrics, cv_predictions,
    features_for, main, make_split, prepare_cohort, regression_metrics,
    run_experiment,
)
from ml.quality import FULL_FEATURES, PhysicalSanitizer


def synthetic_frame(n=80):
    rng = np.random.default_rng(812)
    frame = pd.DataFrame(rng.uniform(1, 10, (n, len(FULL_FEATURES))), columns=FULL_FEATURES,
                         index=np.arange(1000, 1000 + n))
    frame["Potability"] = np.arange(n) % 2
    frame["pH"] = 5 + frame["Turbidity"] / 5
    return frame


def test_shared_split_excludes_duplicates_and_all_conflicting_rows():
    frame = synthetic_frame()
    # Identical features, identical label: retain only the first source row.
    frame.loc[2000] = frame.loc[1000]
    # Quick-only conflicting group: remove both, although full features differ.
    frame.loc[1003, ["Turbidity", "Hardness", "TDS"]] = frame.loc[
        1002, ["Turbidity", "Hardness", "TDS"]]
    # Distinct raw values collapse under sanitization; these must not leak either.
    frame.loc[1005, ["Hardness", "TDS"]] = frame.loc[1004, ["Hardness", "TDS"]]
    frame.loc[1004, "Turbidity"] = -1
    frame.loc[1005, "Turbidity"] = -2
    data, audit = prepare_cohort(frame, "classification", "binary_targets")
    assert 2000 in audit["duplicate_row_ids_removed"]
    assert {1002, 1003}.issubset(audit["conflicting_row_ids_removed"])
    assert {1004, 1005}.issubset(audit["conflicting_row_ids_removed"])
    split = make_split(data, "classification")
    assert split == make_split(data, "classification")
    train, test = split["train_row_ids"], split["test_row_ids"]
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(data.index)
    for mode in ("quick", "full"):
        for physical in (False, True):
            view = data[features_for("classification", mode)]
            if physical:
                view = PhysicalSanitizer().fit_transform(view)
            assert not view.duplicated().any()
    assert "NOT unseen external" in split["interpretation"]


def test_fold_local_imputation(monkeypatch):
    X = pd.DataFrame({"Turbidity": [1., 3., np.nan, 100., 200., np.nan,
                                  400., 800., 900., 1000., np.nan, 1200.]},
                     index=np.arange(30, 42))
    y = pd.Series([0, 1] * 6, index=X.index)
    observed = []
    original_fit = SimpleImputer.fit

    def record_fit(self, data, y=None):
        result = original_fit(self, data, y)
        observed.append((data.index.tolist(), self.statistics_.copy()))
        return result

    monkeypatch.setattr(SimpleImputer, "fit", record_fit)
    pipeline = build_pipeline("classification", "linear", "raw")
    oof, summary, folds = cv_predictions(pipeline, X, y, "classification", 3)
    assert np.isfinite(oof).all()
    assert len(observed) == 3
    for (ids, statistics), fold in zip(observed, folds):
        assert ids == fold["train_row_ids"]
        assert set(ids).isdisjoint(fold["validation_row_ids"])
        assert len(ids) < len(X)
        np.testing.assert_allclose(statistics, X.loc[ids].median().to_numpy())
    assert "f1" in summary["mean_std"]
    assert not hasattr(pipeline.named_steps["imputer"], "statistics_")


def test_regression_excludes_both_targets_and_never_imputes_target():
    frame = synthetic_frame()
    frame.loc[1000, "pH"] = np.nan
    frame.loc[1001, "pH"] = np.inf
    frame.loc[1002, "pH"] = -1.5
    frame.loc[1003, "pH"] = 18.2
    raw, _ = prepare_cohort(frame, "regression", "raw_targets")
    physical, _ = prepare_cohort(frame, "regression", "physical_valid_targets")
    assert 1000 not in raw.index and 1001 not in raw.index
    assert {1002, 1003}.issubset(raw.index)
    assert 1002 not in physical.index and 1003 not in physical.index
    assert raw.loc[1002, "pH"] == -1.5
    for mode in ("full", "quick"):
        assert {"Potability", "pH"}.isdisjoint(features_for("regression", mode))


def test_threshold_maximizes_safe_recall_subject_to_empirical_fpr():
    y = np.r_[np.zeros(100), np.ones(30)]
    probability = np.r_[np.linspace(.01, .9, 100), np.linspace(.65, .99, 30)]
    threshold = choose_threshold(y, probability)
    metrics = classification_metrics(y, probability, threshold)
    assert metrics["unsafe_to_safe_fpr"] <= .02
    for candidate in np.unique(probability):
        candidate_metrics = classification_metrics(y, probability, candidate)
        if candidate_metrics["unsafe_to_safe_fpr"] <= .02:
            assert candidate_metrics["safe_recall"] <= metrics["safe_recall"]
    assert metrics["unsafe_fpr_wilson_95"][1] > .02
    fallback = choose_threshold([0, 0, 1, 1], [.9, .9, .2, .3])
    assert fallback > 1
    assert classification_metrics([0, 0, 1, 1], [.9, .9, .2, .3], fallback)["safe_prediction_coverage"] == 0


def test_baseline_metrics_have_explicit_orientation_and_probability_semantics():
    result = classification_metrics([0, 0, 1, 1], [.1, .6, .7, .4])
    assert result["f1"] == .5
    assert result["safe_precision"] == .5
    assert result["safe_recall"] == .5
    assert result["unsafe_recall"] == .5
    assert result["unsafe_to_safe_fpr"] == .5
    assert result["safe_prediction_coverage"] == .5
    assert set(result["confusion_matrix"].values()) == {1}
    assert result["brier_score"] == pytest.approx((.01 + .36 + .09 + .36) / 4)
    assert regression_metrics([1, 3], [2, 2]) == {"rmse": 1., "mae": 1., "r2": 0.}


def test_conservative_threshold_never_increases_safe_calls_vs_default():
    y = np.r_[np.zeros(100), np.ones(100)]
    probabilities = np.r_[np.linspace(.01, .4, 100), np.linspace(.3, .99, 100)]
    threshold = choose_threshold(y, probabilities)
    assert threshold == .5
    assert np.all((probabilities >= threshold) <= (probabilities >= .5))


def test_selected_pipeline_bundle_and_stress_thresholds_use_training_only(tmp_path):
    frame, _ = prepare_cohort(synthetic_frame(60), "classification", "binary_targets")
    split = make_split(frame, "classification")
    (tmp_path / "candidates").mkdir()
    args = SimpleNamespace(output=tmp_path, trees=2, folds=2, permutation_repeats=1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        result = run_experiment(frame, "classification", "binary_targets", "quick", "raw",
                                split, args, {"experimental": True, "approved": False})
    assert set(result["threshold"]["training_row_ids"]).isdisjoint(split["test_row_ids"])
    assert len(result["cv_candidates"]) == 4
    bundle = joblib.load(result["candidate_bundle"])
    assert bundle["metadata"]["approved"] is False
    assert bundle["metadata"]["feature_order"] == ["Turbidity", "Hardness", "TDS"]
    assert bundle["pipeline"].predict(frame[bundle["metadata"]["feature_order"]]).shape == (60,)
    for stress in result["stress_tests"].values():
        train = set(stress["threshold"]["training_row_ids"])
        assert train.isdisjoint(stress["split"]["test_row_ids"])
        for fold in stress["threshold_cv_folds"]:
            assert set(fold["train_row_ids"]) <= train
            assert set(fold["validation_row_ids"]) <= train


def test_output_refuses_existing_files_before_loading_dataset(tmp_path):
    sentinel = tmp_path / "phase1.joblib"
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        main(["--dataset", "does-not-exist.csv", "--output", str(tmp_path)])
    assert sentinel.read_bytes() == b"preserve me"


def test_regression_experiment_reports_dummy_skill_and_original_column_importance(tmp_path):
    frame, _ = prepare_cohort(synthetic_frame(60), "regression", "raw_targets")
    split = make_split(frame, "regression")
    (tmp_path / "candidates").mkdir()
    args = SimpleNamespace(output=tmp_path, trees=2, folds=2, permutation_repeats=1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        result = run_experiment(frame, "regression", "raw_targets", "quick", "physical_sanitized",
                                split, args, {"experimental": True, "approved": False})
    selected = result["selected_internal_holdout"]["rmse"]
    dummy = result["dummy_internal_holdout"]["rmse"]
    assert result["rmse_skill_vs_dummy"] == pytest.approx(1 - selected / dummy)
    assert result["better_than_dummy_rmse"] == (selected < dummy)
    assert set(result["permutation_importance"]["features"]) == {"Turbidity", "Hardness", "TDS"}
    assert result["threshold"] is None
    assert result["selected_configuration"]["target_imputed"] is False
