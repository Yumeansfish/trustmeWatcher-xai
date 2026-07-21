"""Test the dashboard prediction report"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from trustme_xai.contracts import QUESTION_IDS, TARGET_COLUMNS
from trustme_xai.feature_pipeline.production_features import PRODUCTION_FEATURE_SET
from trustme_xai.inference import prediction_report as report_module
from trustme_xai.inference.model_runtime import (
    CURRENT_MODEL_SHA256,
    FeaturePreprocessor,
    ModelBundle,
    TargetModel,
)
from trustme_xai.inference.prediction_report import build_prediction_report
from trustme_xai.inference.shap_explainer import LocalShapExplanation


def test_prediction_report_uses_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_explanation(
        target_model: TargetModel,
        feature_row: pd.DataFrame,
    ) -> tuple[float, LocalShapExplanation]:
        assert len(feature_row) == 1
        return float(TARGET_COLUMNS.index(target_model.target)), {
            "base_value": 0.0,
            "features": [],
        }

    monkeypatch.setattr(
        report_module,
        "explain_target_prediction",
        fake_explanation,
    )
    target_models = {
        target: TargetModel(
            target=target,
            model_name="saved-model",
            estimator=object(),
            preprocessor=cast(FeaturePreprocessor, object()),
            metrics={},
        )
        for target in TARGET_COLUMNS
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
