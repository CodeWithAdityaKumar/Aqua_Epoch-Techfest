"""Named-frame inference and conservative demo-only assessment policy."""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from .artifacts import ArtifactUnavailable, LoadedModel


def _warning(code, message, features=()):
    return {"code": code, "message": message, "features": list(features)}


def _inputs(model: LoadedModel, payload: dict[str, Any]):
    metadata = model.metadata
    features = metadata["feature_order"]
    missing, outside, unusual, unknown = [], [], [], []
    for feature in features:
        value = payload[feature]
        reference = metadata["reference_ranges"][feature]
        if value is None:
            missing.append(feature)
        elif reference["min"] is None:
            unknown.append(feature)
        elif not reference["min"] <= value <= reference["max"]:
            outside.append(feature)
        elif not reference["q01"] <= value <= reference["q99"]:
            unusual.append(feature)
    messages = []
    if missing:
        messages.append(_warning("missing_measurements", "Explicitly missing inputs are handled by the trained pipeline; evidence is incomplete.", missing))
    if outside:
        messages.append(_warning("outside_training_range", "Inputs exceed the sanitized training minimum/maximum; model evidence is insufficient.", outside))
    if unusual:
        messages.append(_warning("unusual_measurements", "Inputs lie outside the training 1st–99th percentiles but within its minimum/maximum.", unusual))
    if unknown:
        messages.append(_warning("unknown_training_range", "Training reference ranges are unavailable for these inputs.", unknown))
    frame = pd.DataFrame([[np.nan if payload[f] is None else payload[f] for f in features]],
                         columns=features, dtype=float)
    return frame, messages, bool(missing or outside or unknown)


def _runtime_warning():
    return _warning("model_runtime_warning", "The model runtime emitted a warning; review the promoted release environment.")


def classify(model: LoadedModel, payload: dict[str, Any]) -> dict:
    model_id = model.metadata["model_id"]
    row, messages, abstain = _inputs(model, payload)
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            probabilities = np.asarray(model.pipeline.predict_proba(row))
        if (probabilities.shape != (1, 2) or probabilities.dtype.kind not in "iuf"
                or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-6)):
            raise ValueError("Invalid probability schema")
        probability = float(probabilities[0, 1])
    except Exception:
        raise ArtifactUnavailable("invalid_model_output", model_id) from None
    threshold = model.metadata["threshold"]
    # Use the recorded safety-oriented threshold, never estimator.predict's
    # default threshold. These remain dataset labels, not certification.
    class_id = int(probability >= threshold)
    if model.metadata["mode"] == "quick":
        abstain = True
        messages.append(_warning("limited_inputs", "Three inputs and poor Phase2 quick-model precision provide insufficient evidence for a safety assessment.", model.metadata["feature_order"]))
    if abs(probability - threshold) < .05:
        abstain = True
        messages.append(_warning("near_threshold", "Probability is within 0.05 of the decision threshold. This abstention heuristic is not calibrated confidence."))
    messages.extend([
        _warning("uncalibrated_probability", "probability_safe is uncalibrated P(Potability=1), not confidence or a guarantee of safety."),
        _warning("not_certification", "Competition demo only. Dataset-label prediction is not BIS certification or a drinking-water recommendation."),
    ])
    if emitted or model.runtime_warning:
        messages.append(_runtime_warning())
    return {"model_id": model_id, "class_id": class_id,
            "model_label": "safe" if class_id else "unsafe",
            "probability_safe": probability, "decision_threshold": threshold,
            "assessment": "insufficient_evidence" if abstain else ("predicted_safe" if class_id else "predicted_unsafe"),
            "warnings": messages}


def regress(model: LoadedModel, payload: dict[str, Any]) -> dict:
    model_id = model.metadata["model_id"]
    row, messages, _ = _inputs(model, payload)
    try:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            predictions = np.asarray(model.pipeline.predict(row))
        if (predictions.shape != (1,) or predictions.dtype.kind not in "iuf"
                or not np.isfinite(predictions[0]) or not 0 <= predictions[0] <= 14):
            raise ValueError("Invalid pH prediction")
        estimate = float(predictions[0])
    except Exception:
        raise ArtifactUnavailable("invalid_model_output", model_id) from None
    messages.extend([
        _warning("weak_regression", "pH regression has weak predictive evidence; reliability is limited and no calibrated interval is available."),
        _warning("not_lab_measurement", "Estimated pH is not a laboratory measurement and is never substituted for measured pH in classification."),
    ])
    if emitted or model.runtime_warning:
        messages.append(_runtime_warning())
    return {"model_id": model_id, "estimate": estimate, "reliability": "limited", "warnings": messages}
