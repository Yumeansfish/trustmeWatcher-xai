"""Tests for the semantic seven-target prediction report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.contracts import MODEL_TARGETS, TARGET_TITLES
from trustme_xai.inference.model_runtime import ModelBundle, TargetModel
from trustme_xai.inference.prediction_report import build_prediction_report


class _IdentityPreprocessor:
    feature_columns = ["x"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


def _bundle() -> ModelBundle:
    target_models = {
        target: TargetModel(
            target=target,
            model_name="dummy",
            feature_set="test_features",
            feature_columns=list(_IdentityPreprocessor.feature_columns),
            prediction_frame="absolute",
            estimator=_ConstantEstimator(index / 2.0),
            preprocessor=_IdentityPreprocessor(),
            metrics={"validation_mse": 0.5, "test_mse": 0.75},
            fallback_score=3.0,
            score_range=(0.0, 6.0),
        )
        for index, target in enumerate(MODEL_TARGETS)
    }
    return ModelBundle(
        feature_set="test_features",
        feature_columns=["x"],
        targets=list(MODEL_TARGETS),
        target_models=target_models,
        metadata={"model_version": "test-v1", "normalization": "global"},
    )


def test_report_uses_canonical_targets_and_prediction_field() -> None:
    timestamp = pd.Timestamp("2026-07-01 10:30:00")
    row = pd.DataFrame(
        {"user_id": ["u1"], "timestamp": [timestamp], "x": [1.0]}
    )

    report = build_prediction_report(_bundle(), row, "u1", timestamp)

    assert report["schema_version"] == 3
    assert report["participant_id"] == "u1"
    assert report["model_version"] == "test-v1"
    assert [item["target"] for item in report["predictions"]] == MODEL_TARGETS
    for item in report["predictions"]:
        assert item["title"] == TARGET_TITLES[item["target"]]
        assert isinstance(item["prediction"], float)


def test_report_requires_one_matching_participant_row() -> None:
    timestamp = pd.Timestamp("2026-07-01 10:30:00")
    rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "timestamp": [timestamp, timestamp],
            "x": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="exactly one row"):
        build_prediction_report(_bundle(), rows, "u1", timestamp)
