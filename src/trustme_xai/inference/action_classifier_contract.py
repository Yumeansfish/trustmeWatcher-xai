"""Validate the packaged action-classifier artifact contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.action_allocations import (
    ACTION_ALLOCATION_COLUMNS,
)

ACTION_CLASSIFIER_SCHEMA_VERSION = 1
ACTION_CLASSIFIER_MODEL_VERSION = "action_classifier_v1"
ACTION_CLASSIFIER_LABEL_RULE = "high_if_score_gt_3_else_not_high"
ACTION_CLASSIFIER_CALIBRATION = "none"
CLASSIFIER_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
)


@dataclass(frozen=True)
class ActionClassifierArtifactFacts:
    """Expose validated facts needed by classifier inference."""

    model_version: str
    minimum_complete_history_rows: int
    label_rule: str
    probability_calibration: str
    test_macro_metrics: Mapping[str, float]


def validate_action_classifier_artifact(
    bundle: Any,
) -> ActionClassifierArtifactFacts:
    """Validate one loaded classifier artifact and return runtime facts."""
    if getattr(bundle, "schema_version", None) != ACTION_CLASSIFIER_SCHEMA_VERSION:
        raise ValueError("unsupported action classifier schema version")
    if getattr(bundle, "targets", None) != MODEL_TARGETS:
        raise ValueError("action classifier targets do not match the runtime")
    target_models = getattr(bundle, "target_models", None)
    if not isinstance(target_models, dict) or set(target_models) != set(MODEL_TARGETS):
        raise ValueError("action classifier target models do not match targets")

    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("action classifier metadata must be an object")
    model_version = metadata.get("model_version")
    if model_version != ACTION_CLASSIFIER_MODEL_VERSION:
        raise ValueError("action classifier model version does not match the runtime")
    if metadata.get("label_rule") != ACTION_CLASSIFIER_LABEL_RULE:
        raise ValueError("action classifier label rule does not match the runtime")
    if metadata.get("probability_calibration") != ACTION_CLASSIFIER_CALIBRATION:
        raise ValueError("action classifier calibration does not match the runtime")

    minimum_history = metadata.get("minimum_complete_history_rows")
    if minimum_history != 1:
        raise ValueError("action classifier must require one history row")
    raw_metrics = metadata.get("test_macro_metrics")
    if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(
        CLASSIFIER_METRIC_NAMES,
    ):
        raise ValueError("action classifier test metrics do not match the contract")
    metrics = {name: float(raw_metrics[name]) for name in CLASSIFIER_METRIC_NAMES}
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("action classifier test metrics must be finite")

    for target in MODEL_TARGETS:
        model = target_models[target]
        if getattr(model, "target", None) != target:
            raise ValueError(f"action classifier target mismatch: {target}")
        feature_columns = getattr(model, "feature_columns", None)
        if not isinstance(feature_columns, list):
            raise ValueError(f"{target} classifier features are invalid")
        missing = sorted(set(ACTION_ALLOCATION_COLUMNS) - set(feature_columns))
        if missing:
            raise ValueError(f"{target} classifier omits action allocations: {missing}")
        preprocessor = getattr(model, "preprocessor", None)
        if list(getattr(preprocessor, "feature_columns", ())) != list(
            ACTION_ALLOCATION_COLUMNS,
        ):
            raise ValueError(f"{target} classifier preprocessor does not match")

    return ActionClassifierArtifactFacts(
        model_version=model_version,
        minimum_complete_history_rows=minimum_history,
        label_rule=ACTION_CLASSIFIER_LABEL_RULE,
        probability_calibration=ACTION_CLASSIFIER_CALIBRATION,
        test_macro_metrics=MappingProxyType(metrics),
    )
