"""Shared feature names and conservative input validation bounds.

The bounds below are broad physical-plausibility checks for API hygiene. They
are intentionally not presented as BIS drinking-water limits or as a safety
certification rule.
"""

from __future__ import annotations

from pathlib import Path

# These names and this order match the CSV dataset in the repository. Models
# are expected to have been trained with the same positional feature order.
FULL_FEATURES: tuple[str, ...] = (
    "pH",
    "Hardness",
    "TDS",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Calcium",
    "Magnesium",
    "Sodium",
    "Potassium",
    "Chloride",
    "Nitrate",
    "Fluoride",
    "Iron",
    "Alkalinity",
)

# pH is the regression target for the hybrid task, so a pH regressor uses the
# remaining dataset columns. The classifier receives all FULL_FEATURES.
PH_REGRESSION_FEATURES: tuple[str, ...] = tuple(
    feature for feature in FULL_FEATURES if feature != "pH"
)

# The challenge brief explicitly calls out these three metrics for the quick
# interaction path. Quick models are separate persisted artifacts because a
# full model cannot infer omitted features without an explicit preprocessing
# strategy.
QUICK_FEATURES: tuple[str, ...] = ("Turbidity", "Hardness", "TDS")

# Broad, non-negative screening ranges. Values are not official regulatory
# thresholds and should not be used to decide whether water is safe to drink.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "pH": (0.0, 14.0),
    "Hardness": (0.0, 5_000.0),
    "TDS": (0.0, 100_000.0),
    "Chloramines": (0.0, 100.0),
    "Sulfate": (0.0, 5_000.0),
    "Conductivity": (0.0, 100_000.0),
    "Organic_carbon": (0.0, 1_000.0),
    "Trihalomethanes": (0.0, 1_000.0),
    "Turbidity": (0.0, 1_000.0),
    "Calcium": (0.0, 5_000.0),
    "Magnesium": (0.0, 5_000.0),
    "Sodium": (0.0, 5_000.0),
    "Potassium": (0.0, 1_000.0),
    "Chloride": (0.0, 10_000.0),
    "Nitrate": (0.0, 1_000.0),
    "Fluoride": (0.0, 100.0),
    "Iron": (0.0, 1_000.0),
    "Alkalinity": (0.0, 5_000.0),
}

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "phase3"
