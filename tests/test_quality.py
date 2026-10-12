"""Synthetic data-quality checks; no dependency on the production dataset."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.quality import (
    FULL_FEATURES,
    QUICK_FEATURES,
    REGRESSION_FEATURES,
    PhysicalSanitizer,
    load_quality_data,
    valid_ph_target,
)


SCHEMA = FULL_FEATURES + ["Potability"]


def record(**updates):
    values = {column: 1.0 for column in FULL_FEATURES}
    values.update(pH=7.0, Potability=1)
    values.update(updates)
    return [values[column] for column in SCHEMA]


def dataset(tmp_path, rows, columns=None):
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(SCHEMA if columns is None else columns)
        writer.writerows(rows)
    return path


def test_counts_coercions_raw_values_and_stable_source_ids(tmp_path):
    path = dataset(tmp_path, [
        record(pH=0.2, Hardness=-3, Fluoride=-2, TDS=100001, Turbidity=21),
        [],  # A truly blank CSV line must count and retain its source position.
        record(pH=14.8, Iron="bad", Sodium="inf"),
        record(Potability=" ", pH=18.2, Hardness=-4),
        [" "] * len(SCHEMA),
        record(pH=22, Calcium="", Magnesium="-inf", Potability=0),
        record(pH=-1.5, TDS=100000, Turbidity=20),
    ])
    frame, audit = load_quality_data(path)

    assert list(frame.columns) == SCHEMA
    assert frame.index.tolist() == [0, 2, 5, 6]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
    assert frame.loc[0, "Hardness"] == -3
    assert frame.loc[0, "pH"] == 0.2
    assert frame.loc[2, "pH"] == 14.8
    assert np.isnan(frame.loc[2, "Iron"])
    assert np.isnan(frame.loc[2, "Sodium"])
    assert audit["rows_read"] == 7
    assert audit["fully_blank_rows_removed"] == 2
    assert audit["missing_target_rows_removed"] == 3
    assert audit["missing_target_nonblank_rows_removed"] == 1
    assert audit["rows_after_cleaning"] == 4
    assert audit["removed_source_ids"] == [1, 3, 4]
    assert audit["removed_csv_row_numbers_1_indexed"] == [3, 5, 6]
    assert audit["raw"]["negative_per_column"]["Hardness"] == 2
    assert audit["labelled"]["negative_per_column"]["Hardness"] == 1
    assert audit["raw"]["invalid_ph_observed"] == 4
    assert audit["labelled"]["invalid_ph_observed"] == 3
    assert audit["raw"]["suspect_ph_sentinel_frequencies"] == {
        "0.2": 1, "14.8": 1, "18.2": 1, "22.0": 1, "-1.5": 1,
    }
    assert audit["labelled"]["suspect_ph_sentinel_frequencies"]["18.2"] == 0
    assert audit["labelled"]["nonnumeric_coercions_per_column"]["Iron"] == 1
    assert audit["labelled"]["nonfinite_coercions_per_column"]["Sodium"] == 1
    assert audit["labelled"]["nonfinite_coercions_per_column"]["Magnesium"] == 1
    assert audit["labelled"]["missing_per_column_before_coercion"]["Calcium"] == 1
    assert audit["labelled"]["missing_per_column_after_coercion"]["Iron"] == 1
    assert audit["labelled"]["paired_negative_hardness_and_fluoride"] == 1
    assert audit["labelled"]["exploratory_spikes"] == {
        "tds_above_100000": 1, "turbidity_above_20": 1, "overlap": 1, "union": 1,
    }
    assert set(audit["units"].values()) == {"unknown"}
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("columns", [
    SCHEMA[:-1], SCHEMA + ["extra"], list(reversed(SCHEMA)),
    ["ph"] + SCHEMA[1:], ["Hardness"] + SCHEMA[1:],
])
def test_schema_requires_exact_names_number_and_order(tmp_path, columns):
    with pytest.raises(ValueError, match="schema must exactly match"):
        load_quality_data(dataset(tmp_path, [], columns))


@pytest.mark.parametrize("label", ["unknown", "NaN", "inf", "-inf", -1, 2, 0.5])
def test_supplied_invalid_labels_are_rejected_not_silently_dropped(tmp_path, label):
    with pytest.raises(ValueError, match=r"finite 0/1 labels; invalid CSV lines: \[3\]"):
        load_quality_data(dataset(tmp_path, [record(), record(Potability=label)]))


def test_malformed_record_cannot_create_implicit_pandas_index(tmp_path):
    with pytest.raises(ValueError, match="has 20 fields"):
        load_quality_data(dataset(tmp_path, [record() + [5]]))


def test_duplicate_feature_conflicts_include_missing_values_and_source_ids(tmp_path):
    same = record(Iron="")
    frame, audit = load_quality_data(dataset(tmp_path, [
        same, [], same, record(Iron="", Potability=0), record(TDS=2),
    ]))
    duplicates = audit["duplicates"]
    assert len(frame) == 4  # Audit reports rather than deletes duplicates.
    assert duplicates["feature_duplicate_groups"] == 1
    assert duplicates["rows_in_duplicate_groups"] == 3
    assert duplicates["redundant_feature_rows"] == 2
    assert duplicates["conflicting_label_groups"] == 1
    assert duplicates["rows_in_conflicting_label_groups"] == 3
    assert duplicates["conflicts"] == [{
        "source_ids": [0, 2, 3], "csv_line_numbers": [2, 4, 5], "labels": [0, 1],
    }]


def test_sanitizer_preserves_frame_order_indices_and_positive_outliers():
    frame = pd.DataFrame({
        "Turbidity": [np.inf, -2, 5000, 0, 20, np.nan],
        "pH": [0.2, 0, 14, 14.8, -1.5, -np.inf],
        "Hardness": [-3, 0, 1e9, 5, np.nan, 3],
    }, index=[3, 8, 12, 15, 16, 22])
    original = frame.copy(deep=True)
    sanitizer = PhysicalSanitizer()
    result = sanitizer.fit_transform(frame)
    assert_frame_equal(frame, original)
    assert result.index.equals(frame.index)
    assert result.columns.equals(frame.columns)
    assert result["pH"].iloc[:3].tolist() == [0.2, 0, 14]
    assert result["pH"].iloc[3:].isna().all()
    assert result.loc[12, "Hardness"] == 1e9
    assert result.loc[12, "Turbidity"] == 5000
    assert result.loc[8, "Hardness"] == 0
    assert result.loc[[3, 8], "Turbidity"].isna().all()
    assert sanitizer.get_feature_names_out().tolist() == list(frame.columns)
    assert not hasattr(clone(sanitizer), "feature_names_in_")


def test_sanitizer_rejects_unfitted_or_mismatched_schema():
    frame = pd.DataFrame({"pH": [7], "TDS": [200]})
    with pytest.raises(NotFittedError):
        PhysicalSanitizer().transform(frame)
    sanitizer = PhysicalSanitizer().fit(frame)
    with pytest.raises(ValueError, match="order"):
        sanitizer.transform(frame[["TDS", "pH"]])
    with pytest.raises(TypeError, match="DataFrame"):
        sanitizer.transform(frame.to_numpy())
    with pytest.raises(ValueError, match="Unknown"):
        PhysicalSanitizer().fit(pd.DataFrame({"Potability": [1]}))


def test_pipeline_imputation_fits_training_only_and_transform_does_not_refit():
    training = pd.DataFrame({"pH": [6, 8, -1.5], "TDS": [10, 30, -3]})
    holdout = pd.DataFrame({"pH": [22, 0.2], "TDS": [-7, 1e9]}, index=[50, 70])
    pipeline = Pipeline([
        ("physical", PhysicalSanitizer()),
        ("imputer", SimpleImputer(strategy="median")),
    ])
    pipeline.fit(training)
    statistics = pipeline.named_steps["imputer"].statistics_.copy()
    schema = pipeline.named_steps["physical"].feature_names_in_.copy()
    transformed = pipeline.transform(holdout)
    np.testing.assert_array_equal(statistics, [7, 20])
    np.testing.assert_array_equal(transformed, [[7, 20], [0.2, 1e9]])
    np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, statistics)
    np.testing.assert_array_equal(pipeline.named_steps["physical"].feature_names_in_, schema)


def test_regression_target_mask_is_inclusive_and_source_aligned():
    frame = pd.DataFrame({"pH": [0, 14, 0.2, -1.5, 14.8, np.inf, np.nan, "bad"]}, index=range(5, 13))
    result = valid_ph_target(frame)
    assert result.index.equals(frame.index)
    assert result.dtype == bool
    assert result.tolist() == [True, True, True, False, False, False, False, False]
    assert len(FULL_FEATURES) == 18
    assert QUICK_FEATURES == ["Turbidity", "Hardness", "TDS"]
    assert REGRESSION_FEATURES == FULL_FEATURES[1:]


def test_outlier_sensitivity_views_and_tentative_label_comparison(tmp_path):
    rows = [
        record(TDS=1, Turbidity=3, pH=7, Potability=0),
        record(TDS=2, Turbidity=2, pH=0.2),
        record(TDS=3, Turbidity=1, pH=22),
        record(TDS=200000, Turbidity=100, pH=""),
    ]
    _, audit = load_quality_data(dataset(tmp_path, rows))
    views = audit["statistical_views"]
    assert views["raw"]["rows"] == 4
    assert views["raw"]["pearson_correlations"]["TDS"]["Turbidity"] > 0.99
    assert views["spike_excluded"]["rows"] == 3
    assert views["spike_excluded"]["pearson_correlations"]["TDS"]["Turbidity"] == pytest.approx(-1)
    assert views["raw"]["descriptive_summary"]["pH"]["count"] == 3
    assert views["physical_sanitized"]["descriptive_summary"]["pH"]["count"] == 2
    assert views["raw"]["pearson_correlations"]["Iron"]["TDS"] is None
    assert views["physical_sanitized"]["pairwise_observation_counts"]["pH"]["TDS"] == 2
    disagreements = audit["tentative_ph_label_disagreements"]
    assert disagreements["potable_outside_range"] == 2
    assert disagreements["nonpotable_inside_range"] == 1
    assert disagreements["missing_ph_excluded"] == 1
    assert "not a BIS rule" in disagreements["interpretation"]
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize("rows", [[], [[], [""] * len(SCHEMA)]])
def test_empty_labelled_dataset_is_auditable_without_json_nan(tmp_path, rows):
    frame, audit = load_quality_data(dataset(tmp_path, rows))
    assert frame.empty
    assert audit["rows_after_cleaning"] == 0
    assert audit["duplicates"]["feature_duplicate_groups"] == 0
    assert audit["statistical_views"]["raw"]["descriptive_summary"]["pH"]["mean"] is None
    json.dumps(audit, allow_nan=False)
