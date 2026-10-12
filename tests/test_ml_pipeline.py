from pathlib import Path

import numpy as np
import pandas as pd

from ml.train import (
    DATASET_FILENAME,
    build_classifier_pipeline,
    build_regressor_pipeline,
    feature_sets,
    load_and_clean_dataset,
    validate_input_record,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / DATASET_FILENAME


def test_cleaning_is_auditable_and_retains_feature_missingness():
    frame, summary = load_and_clean_dataset(DATASET)

    assert summary["rows_read"] == 26500
    assert summary["fully_blank_rows_removed"] == 3888
    assert summary["missing_target_rows_removed"] == 3889
    assert summary["rows_after_cleaning"] == 22611
    assert frame["Potability"].notna().all()
    assert summary["missing_feature_values_after_cleaning"]
    assert len(summary["removed_csv_row_numbers_1_indexed"]) == summary["rows_removed_total"]


def test_regression_schema_excludes_pH_and_potability():
    frame, _ = load_and_clean_dataset(DATASET)
    sets = feature_sets(frame.columns)

    assert len(sets["classification_full"]) == 18
    assert len(sets["regression_full"]) == 17
    assert "pH" not in sets["regression_full"]
    assert "Potability" not in sets["regression_full"]
    assert sets["classification_ui"] == ["Turbidity", "Hardness", "TDS"]
    assert sets["regression_ui"] == ["Turbidity", "Hardness", "TDS"]


def test_missing_values_are_handled_inside_both_pipelines():
    x_class = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [2.0, 3.0, np.nan, 5.0]})
    y_class = [0, 1, 0, 1]
    classifier = build_classifier_pipeline(random_state=42, n_estimators=8, max_depth=6)
    classifier.fit(x_class, y_class)
    assert classifier.predict(pd.DataFrame({"a": [np.nan], "b": [4.0]})).shape == (1,)

    y_reg = [6.0, 6.5, 7.0, 7.5]
    regressor = build_regressor_pipeline(random_state=42, n_estimators=8, max_depth=6)
    regressor.fit(x_class, y_reg)
    assert regressor.predict(pd.DataFrame({"a": [2.0], "b": [np.nan]})).shape == (1,)


def test_validation_is_advisory_and_does_not_claim_safety():
    bundle = {
        "metadata": {
            "feature_order": ["Turbidity"],
            "schema": {
                "feature_order": ["Turbidity"],
                "input_validation": {
                    "domain_safety_claim": False,
                    "message": "Observed bounds only",
                    "features": {
                        "Turbidity": {
                            "observed_training_min": 0.0,
                            "observed_training_max": 10.0,
                        }
                    },
                },
            },
        }
    }
    result = validate_input_record(bundle, {"Turbidity": 11})
    assert result["valid"]
    assert result["warnings"]
    assert result["domain_safety_claim"] is False
