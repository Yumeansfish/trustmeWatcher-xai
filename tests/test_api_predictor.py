"""Test the runtime prediction API"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.inference import inference_service as predictor_module
from trustme_xai.inference.inference_service import (
    predict_current,
    run_inference,
)
from trustme_xai.inference.model_runtime import ModelBundle, TargetModel


class _IdentityPreprocessor:
    feature_columns = ["time_personal_distraction"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


def _model_bundle() -> ModelBundle:
    target_models = {
        target: TargetModel(
            target=target,
            model_name="dummy",
            feature_set="test_features",
            feature_columns=list(_IdentityPreprocessor.feature_columns),
            prediction_frame="absolute",
            estimator=_ConstantEstimator(value),
            preprocessor=_IdentityPreprocessor(),
            metrics={},
            fallback_score=3.0,
        )
        for target, value in (
            ("stress_management", 2.5),
            ("productivity", 4.0),
        )
    }
    return ModelBundle(
        feature_set="test_features",
        feature_columns=["time_personal_distraction"],
        targets=list(target_models),
        target_models=target_models,
        metadata={},
    )


def test_predict_current_returns_plain_target_mapping() -> None:
    current = pd.DataFrame({"user_id": ["u1"], "time_personal_distraction": [3.0]})

    predictions = predict_current(_model_bundle(), current)

    assert predictions == {"stress_management": 2.5, "productivity": 4.0}
    assert all(isinstance(value, float) for value in predictions.values())


@pytest.mark.parametrize("row_count", [0, 2])
def test_predict_current_requires_exactly_one_row(row_count: int) -> None:
    current = pd.DataFrame(
        {"time_personal_distraction": np.arange(row_count, dtype=float)}
    )

    with pytest.raises(ValueError, match="exactly one row"):
        predict_current(_model_bundle(), current)


def test_run_inference_connects_parser_features_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buckets: dict[str, dict[str, object]] = {
        "window": {"type": "currentwindow", "events": []},
        "input": {"type": "os.hid.input", "events": []},
    }
    as_of = pd.Timestamp("2026-01-10 12:00:00")
    features = pd.DataFrame(
        {
            "user_id": ["u1"],
            "timestamp": [as_of],
            "time_personal_distraction": [0.0],
        }
    )
    expected = {"questions": []}

    monkeypatch.setattr(
        predictor_module,
        "build_runtime_feature_row",
        lambda model, activitywatch_buckets, user_id, as_of,
        previous_questionnaire_times, past_self_reports: features,
    )
    monkeypatch.setattr(
        predictor_module,
        "build_prediction_report",
        lambda bundle, current, user_id, timestamp: expected,
    )

    result = run_inference(
        bundle=_model_bundle(),
        activitywatch_buckets=buckets,
        user_id="u1",
        as_of=as_of,
        previous_questionnaire_times=[],
    )

    assert result is expected
