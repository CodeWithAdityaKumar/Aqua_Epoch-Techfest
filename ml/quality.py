"""Auditable dataset loading and conservative, split-safe physical sanitization.

Units are undocumented. The pH policy is a broad freshwater operational
assumption, not universal pH physics or BIS certification. Positive outliers
and the suspicious pH=0.2 sentinel are never automatically clipped or removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


DATASET_FILENAME = (
    "Aqua_Epoch_BIS_Water_Quality_Dataset.xls - "
    "Aqua_Epoch_BIS_Water_Quality_Dataset.csv"
)
FULL_FEATURES = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity", "Calcium", "Magnesium",
    "Sodium", "Potassium", "Chloride", "Nitrate", "Fluoride", "Iron", "Alkalinity",
]
QUICK_FEATURES = ["Turbidity", "Hardness", "TDS"]
REGRESSION_FEATURES = [column for column in FULL_FEATURES if column != "pH"]
TARGET = "Potability"
SCHEMA = FULL_FEATURES + [TARGET]
PH_SENTINELS = [0.2, 14.8, 18.2, 22.0, -1.5]


def valid_ph_target(frame: pd.DataFrame) -> pd.Series:
    """Return an index-aligned mask for finite pH in the inclusive [0, 14] range."""
    values = pd.to_numeric(frame["pH"], errors="coerce")
    return (np.isfinite(values) & values.between(0, 14)).fillna(False).astype(bool)


class PhysicalSanitizer(TransformerMixin, BaseEstimator):
    """Replace invalid measurements with NaN without learning value thresholds.

    Fit records only the ordered DataFrame schema; transform never fits, imputes,
    removes rows, clips spikes, or mutates the input. Put this before an imputer
    inside a pipeline fitted on training rows only. Feature subsets are allowed.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PhysicalSanitizer":
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PhysicalSanitizer requires a pandas DataFrame")
        if X.columns.has_duplicates or not len(X.columns):
            raise ValueError("Feature names must be unique and nonempty")
        unknown = [column for column in X.columns if column not in FULL_FEATURES]
        if unknown:
            raise ValueError(f"Unknown measurement features: {unknown}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        self._validate_frame(X)
        if list(X.columns) != self.feature_names_in_.tolist():
            raise ValueError("Feature names and order must match the fitted schema")
        result = X.apply(pd.to_numeric, errors="coerce").astype(float)
        result = result.where(np.isfinite(result))
        for column in result:
            if column == "pH":
                result[column] = result[column].where(result[column].between(0, 14))
            else:
                result[column] = result[column].where(result[column] >= 0)
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != self.feature_names_in_.tolist():
            raise ValueError("input_features must match the fitted schema")
        return self.feature_names_in_.copy()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is None or pd.isna(value):
        return None
    return str(value) if isinstance(value, Path) else value


def _counts(mask: pd.DataFrame) -> dict[str, int]:
    return {column: int(count) for column, count in mask.sum().items()}


def _physical_counts(frame: pd.DataFrame) -> dict[str, Any]:
    ph = frame["pH"]
    high_tds = frame["TDS"] > 100000
    high_turbidity = frame["Turbidity"] > 20
    return {
        "negative_per_column": _counts(frame[FULL_FEATURES] < 0),
        "invalid_ph_observed": int((ph.notna() & ~valid_ph_target(frame)).sum()),
        "missing_ph": int(ph.isna().sum()),
        "valid_ph_target_rows": int(valid_ph_target(frame).sum()),
        "suspect_ph_sentinel_frequencies": {
            str(value): int(ph.eq(value).sum()) for value in PH_SENTINELS
        },
        "paired_negative_hardness_and_fluoride": int(
            ((frame["Hardness"] < 0) & (frame["Fluoride"] < 0)).sum()
        ),
        "exploratory_spikes": {
            "tds_above_100000": int(high_tds.sum()),
            "turbidity_above_20": int(high_turbidity.sum()),
            "overlap": int((high_tds & high_turbidity).sum()),
            "union": int((high_tds | high_turbidity).sum()),
        },
    }


def _duplicate_summary(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_mask = frame.duplicated(FULL_FEATURES, keep=False)
    groups = []
    conflicts = []
    for _, group in frame.loc[duplicate_mask].groupby(FULL_FEATURES, dropna=False, sort=False):
        record = {
            "source_ids": group.index.tolist(),
            "csv_line_numbers": (group.index + 2).tolist(),
            "labels": sorted(group[TARGET].unique().tolist()),
        }
        groups.append(record)
        if len(record["labels"]) > 1:
            conflicts.append(record)
    return {
        "definition": "Exact equality of all numeric feature values; missing equals missing. No rows removed.",
        "rows_in_duplicate_groups": int(duplicate_mask.sum()),
        "redundant_feature_rows": int(frame.duplicated(FULL_FEATURES).sum()),
        "feature_duplicate_groups": len(groups),
        "groups": groups,
        "conflicting_label_groups": len(conflicts),
        "rows_in_conflicting_label_groups": sum(len(group["source_ids"]) for group in conflicts),
        "conflicts": conflicts,
    }


def _descriptive_view(frame: pd.DataFrame) -> dict[str, Any]:
    # Pairwise complete observations; undefined constant/empty correlations are null.
    finite = frame.where(np.isfinite(frame))
    observed = finite.notna().astype(np.int64)
    return {
        "rows": len(finite),
        "descriptive_summary": finite.describe().to_dict(),
        "pearson_correlations": finite.corr(method="pearson", min_periods=2).to_dict(),
        "pairwise_observation_counts": observed.T.dot(observed).to_dict(),
    }


def load_quality_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the exact 18-feature + Potability CSV schema and return an audit.

    Original pandas row indices are stable source IDs (CSV line = index + 2).
    Blank lines are read, not skipped. Empty/whitespace labels remove a row;
    supplied labels that are nonnumeric, nonfinite or outside {0, 1} fail.
    Nonnumeric/nonfinite features become NaN with separate conversion counts.
    Negative values and out-of-range pH remain unchanged in the returned frame.
    All counts distinguish raw records from retained labelled records.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        if header != SCHEMA:
            raise ValueError(f"Dataset schema must exactly match ordered columns {SCHEMA}; found {header}")
        for source_id, row in enumerate(reader):
            if row and len(row) != len(SCHEMA) and not (len(row) == 1 and not row[0].strip()):
                raise ValueError(f"CSV line {source_id + 2} has {len(row)} fields; expected {len(SCHEMA)}")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    raw = raw.replace(r"^\s*$", np.nan, regex=True)
    blank = raw.isna().all(axis=1)
    missing_label = raw[TARGET].isna()
    numeric = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    invalid_label = ~missing_label & ~numeric[TARGET].isin([0, 1])
    if invalid_label.any():
        bad_rows = (raw.index[invalid_label] + 2).tolist()
        raise ValueError(f"Potability must contain only finite 0/1 labels; invalid CSV lines: {bad_rows}")

    nonnumeric = raw.notna() & numeric.isna()
    nonfinite = numeric.notna() & ~np.isfinite(numeric)
    numeric = numeric.where(np.isfinite(numeric))
    keep = ~missing_label & ~blank
    frame = numeric.loc[keep].copy()
    frame[TARGET] = frame[TARGET].astype(int)

    sanitized = PhysicalSanitizer().fit_transform(frame[FULL_FEATURES])
    sanitized[TARGET] = frame[TARGET]
    spike = (frame["TDS"] > 100000) | (frame["Turbidity"] > 20)
    observed_ph = frame["pH"].notna()
    in_tentative_range = frame["pH"].between(6.5, 8.5)
    potable_outside = frame[TARGET].eq(1) & observed_ph & ~in_tentative_range
    nonpotable_inside = frame[TARGET].eq(0) & in_tentative_range
    audit = {
        "source_filename": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns": SCHEMA,
        "source_id_policy": "Original pandas index, never reset; CSV line = source_id + 2; blank lines retained on read.",
        "units": {column: "unknown" for column in FULL_FEATURES},
        "policy": {
            "physical_sanitization": "Nonfinite and negative non-pH measurements and pH outside [0,14] become NaN.",
            "interpretation": "Broad freshwater operational assumption, not universal pH physics or BIS certification.",
            "raw_returned_frame": "Nonnumeric/nonfinite values become NaN; negative and out-of-range physical values retained.",
            "positive_outliers": "Never clipped or automatically removed; pH=0.2 retained.",
            "statistics_scope": "Retained labelled rows only; descriptive audit is not fitted model preprocessing.",
            "correlations": "Pearson, pairwise complete finite observations; no causal or safety interpretation.",
            "spike_exclusion": "TDS > 100000 OR Turbidity > 20, exploratory sensitivity analysis only; unknown units.",
        },
        "rows_read": len(raw),
        "fully_blank_rows_removed": int(blank.sum()),
        "missing_target_rows_removed": int(missing_label.sum()),
        "missing_target_nonblank_rows_removed": int((missing_label & ~blank).sum()),
        "rows_removed_total": int((~keep).sum()),
        "rows_after_cleaning": len(frame),
        "removed_source_ids": raw.index[~keep].tolist(),
        "removed_csv_row_numbers_1_indexed": (raw.index[~keep] + 2).tolist(),
        "target_distribution_after_cleaning": {
            str(label): int(count) for label, count in frame[TARGET].value_counts().sort_index().items()
        },
        "raw": {
            "rows": len(raw),
            "missing_per_column_before_coercion": _counts(raw.isna()),
            "nonnumeric_coercions_per_column": _counts(nonnumeric),
            "nonfinite_coercions_per_column": _counts(nonfinite),
            "missing_per_column_after_coercion": _counts(numeric.isna()),
            **_physical_counts(numeric),
        },
        "labelled": {
            "rows": len(frame),
            "missing_per_column_before_coercion": _counts(raw.loc[keep].isna()),
            "nonnumeric_coercions_per_column": _counts(nonnumeric.loc[keep]),
            "nonfinite_coercions_per_column": _counts(nonfinite.loc[keep]),
            "missing_per_column_after_coercion": _counts(frame.isna()),
            "missing_per_column_after_physical_sanitization": _counts(sanitized.isna()),
            **_physical_counts(frame),
        },
        "duplicates": _duplicate_summary(frame),
        "tentative_ph_label_disagreements": {
            "range": [6.5, 8.5],
            "interpretation": "Tentative pH-only comparison, not a BIS rule or proof of incorrect labels; other contaminants may determine Potability.",
            "potable_outside_range": int(potable_outside.sum()),
            "nonpotable_inside_range": int(nonpotable_inside.sum()),
            "missing_ph_excluded": int((~observed_ph).sum()),
            "potable_outside_source_ids": frame.index[potable_outside].tolist(),
            "nonpotable_inside_source_ids": frame.index[nonpotable_inside].tolist(),
        },
        "statistical_views": {
            "raw": _descriptive_view(frame),
            "physical_sanitized": _descriptive_view(sanitized),
            "spike_excluded": _descriptive_view(frame.loc[~spike]),
            "physical_sanitized_spike_excluded": _descriptive_view(sanitized.loc[~spike]),
        },
    }
    return frame, _json_safe(audit)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / DATASET_FILENAME)
    parser.add_argument("--output", type=Path, default=Path("reports/phase2/data_audit.json"))
    args = parser.parse_args(argv)
    _, audit = load_quality_data(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {audit['rows_read']} records; retained {audit['rows_after_cleaning']}. Wrote {args.output}")


if __name__ == "__main__":
    main()
