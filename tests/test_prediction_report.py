"""Test the dashboard prediction report"""

from __future__ import annotations

import pandas as pd
import pytest

from trustme_xai.contracts import QUESTION_IDS, TARGET_COLUMNS
from trustme_xai.feature_pipeline.production_features import PRODUCTION_FEATURE_SET
from trustme_xai.inference.model_runtime import (
    CURRENT_MODEL_SHA256,
    ModelBundle,
    TargetModel,
)
from trustme_xai.inference.prediction_report import build_prediction_report


class IdentityPreprocessor:
    feature_columns = ["x"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table


class ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> list[float]:
        return [self.value] * len(table)


def test_prediction_report_uses_runtime_contract() -> None:
    target_models = {
        target: TargetModel(
            target=target,
            model_name="saved-model",
            estimator=ConstantEstimator(float(index)),
            preprocessor=IdentityPreprocessor(),
            metrics={},
        )
        for index, target in enumerate(TARGET_COLUMNS)
    }
    bundle = ModelBundle(
        feature_set=PRODUCTION_FEATURE_SET,
        feature_columns=["x"],
        targets=list(TARGET_COLUMNS),
        target_models=target_models,
        metadata={},
    )

    report = build_prediction_report(
        bundle,
        pd.DataFrame(
            {
                "user_id": ["user1"],
                "timestamp": [pd.Timestamp("2026-07-21 14:00:00")],
                "x": [1.0],
            },
        ),
        "user1",
        "2026-07-21T14:00:00+02:00",
    )

    assert report["window_minutes"] == 60
    assert report["feature_set"] == PRODUCTION_FEATURE_SET
    assert report["artifact_fingerprint"] == CURRENT_MODEL_SHA256
    assert [item["id"] for item in report["questions"]] == QUESTION_IDS
    assert [item["prediction"] for item in report["questions"]] == [
        float(index) for index in range(len(TARGET_COLUMNS))
    ]
    assert all("explanation" not in item for item in report["questions"])
    assert all("model" not in item for item in report["questions"])
    assert all("metrics" not in item for item in report["questions"])


def test_prediction_report_rejects_mismatched_identity() -> None:
    bundle = ModelBundle(
        feature_set=PRODUCTION_FEATURE_SET,
        feature_columns=[],
        targets=[],
        target_models={},
        metadata={},
    )
    features = pd.DataFrame(
        {
            "user_id": ["user1"],
            "timestamp": [pd.Timestamp("2026-07-21 14:00:00")],
        },
    )

    with pytest.raises(ValueError, match="user_id does not match"):
        build_prediction_report(
            bundle,
            features,
            "user2",
            "2026-07-21T14:00:00+02:00",
        )
