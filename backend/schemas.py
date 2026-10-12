"""Exact, case-sensitive, required-but-nullable request schemas."""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from .config import FEATURE_BOUNDS, FULL_FEATURES, PH_REGRESSION_FEATURES, QUICK_FEATURES


class _StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    @field_validator("*", mode="before")
    @classmethod
    def numeric_or_null(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a numeric JSON measurement or explicit null")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("must be finite")
        return value

    @model_validator(mode="after")
    def at_least_one_measurement(self):
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("at least one measured value is required")
        return self


def _request(name, features):
    return create_model(name, __base__=_StrictRequest, **{
        feature: (float | None, Field(
            ..., ge=FEATURE_BOUNDS[feature][0], le=FEATURE_BOUNDS[feature][1],
            description="Required numeric measurement, or null only if genuinely missing. "
                        "Bounds are engineering checks, not BIS limits; units are undocumented."))
        for feature in features
    })


FullFeatureRequest = _request("FullFeatureRequest", FULL_FEATURES)
QuickFeatureRequest = _request("QuickFeatureRequest", QUICK_FEATURES)
PHFeatureRequest = _request("PHFeatureRequest", PH_REGRESSION_FEATURES)


class CombinedFeatureRequest(FullFeatureRequest):
    """Each requested task needs at least one of its own measured inputs."""

    @model_validator(mode="after")
    def regression_has_measurement(self):
        if all(getattr(self, feature) is None for feature in PH_REGRESSION_FEATURES):
            raise ValueError("combined inference requires at least one non-pH measurement for regression")
        return self


class WarningMessage(BaseModel):
    code: str
    message: str
    features: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    model_id: str
    class_id: Literal[0, 1]
    model_label: Literal["safe", "unsafe"]
    probability_safe: float = Field(ge=0, le=1)
    decision_threshold: float
    assessment: Literal["predicted_safe", "predicted_unsafe", "insufficient_evidence"]
    warnings: list[WarningMessage]


class RegressionResult(BaseModel):
    model_id: str
    estimate: float = Field(ge=0, le=14)
    reliability: Literal["limited"] = "limited"
    warnings: list[WarningMessage]


class HistoryReceipt(BaseModel):
    saved: bool
    id: str | None
    error: str | None


class ResponseEnvelope(BaseModel):
    release_id: str
    mode: Literal["full", "quick"]
    contract_version: str
    api_version: Literal["2.0.0"] = "2.0.0"
    history: HistoryReceipt | None = None


class ClassificationResponse(ResponseEnvelope, ClassificationResult):
    pass


class RegressionResponse(ResponseEnvelope, RegressionResult):
    pass


class PredictionResponse(ResponseEnvelope):
    classification: ClassificationResult
    regression: RegressionResult


FEATURE_LISTS = {"full": FULL_FEATURES, "quick": QUICK_FEATURES, "ph_full": PH_REGRESSION_FEATURES}
