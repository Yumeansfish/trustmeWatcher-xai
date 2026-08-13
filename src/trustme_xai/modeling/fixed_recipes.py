"""Load the fixed compact production recipes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import (
    all_history_columns,
    history_column,
)

COMPACT_MODEL_VERSION = "compact_90_v1"
COMPACT_RECIPE_SCHEMA = "trustme_xai.fixed_recipes"
COMPACT_RECIPE_RESOURCE = "compact_90_v1.json"
COMPACT_FEATURE_COUNTS = {
    "mood_valence": 33,
    "arousal": 16,
    "restfulness": 18,
    "stress_management": 17,
    "productivity": 33,
    "engagement": 24,
    "overall_wellbeing": 16,
}


@dataclass(frozen=True)
class FixedTargetRecipe:
    """Store one immutable target recipe."""

    model_name: str
    prediction_frame: str
    gamma: float
    feature_columns: tuple[str, ...]


def _recipe(target: str, payload: Any) -> FixedTargetRecipe:
    if not isinstance(payload, dict):
        raise ValueError(f"{target} compact recipe must be an object")
    features = tuple(str(value) for value in payload.get("features", []))
    recipe = FixedTargetRecipe(
        model_name=str(payload.get("model", "")),
        prediction_frame=str(payload.get("prediction_frame", "")),
        gamma=float(payload.get("gamma", float("nan"))),
        feature_columns=features,
    )
    if recipe.prediction_frame not in {"absolute", "personal_residual"}:
        raise ValueError(f"{target} compact prediction frame is invalid")
    if not np.isfinite(recipe.gamma) or not 0.0 <= recipe.gamma <= 1.0:
        raise ValueError(f"{target} compact gamma is invalid")
    if len(features) != COMPACT_FEATURE_COUNTS[target]:
        raise ValueError(f"{target} compact feature count is invalid")
    if len(set(features)) != len(features):
        raise ValueError(f"{target} compact features must be unique")
    allowed = {*PRODUCTION_FEATURE_COLUMNS, *all_history_columns()}
    unknown = sorted(set(features) - allowed)
    if unknown:
        raise ValueError(f"{target} compact features are unknown: {unknown}")
    if history_column(target, "mean") not in features:
        raise ValueError(f"{target} compact recipe needs its history mean")
    return recipe


def load_compact_recipes() -> dict[str, FixedTargetRecipe]:
    """Load and validate the seven compact recipes.

    Returns:
        Target names mapped to fixed recipes.
    """
    artifact = resources.files("trustme_xai.modeling").joinpath(
        COMPACT_RECIPE_RESOURCE,
    )
    with artifact.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != COMPACT_RECIPE_SCHEMA:
        raise ValueError("compact recipe schema is not supported")
    if payload.get("model_version") != COMPACT_MODEL_VERSION:
        raise ValueError("compact recipe model version is not supported")
    target_payloads = payload.get("targets")
    if not isinstance(target_payloads, dict):
        raise ValueError("compact recipes need target objects")
    if list(target_payloads) != MODEL_TARGETS:
        raise ValueError("compact recipe targets do not match the contract")
    return {
        target: _recipe(target, target_payloads[target])
        for target in MODEL_TARGETS
    }
