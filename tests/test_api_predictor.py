"""Test the runtime prediction API"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import all_history_columns
from trustme_xai.inference import inference_service as predictor_module
from trustme_xai.inference.inference_service import (
    build_current_features,
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


def _empty_events() -> ParsedActivityWatchEvents:
    return ParsedActivityWatchEvents(
        window=pd.DataFrame(),
        web=pd.DataFrame(),
        input=pd.DataFrame(),
    )


def _active_events() -> ParsedActivityWatchEvents:
    return ParsedActivityWatchEvents(
        window=pd.DataFrame(
            {
                "user_id": ["u1"],
                "timestamp": [pd.Timestamp("2026-01-10 11:30:00")],
                "event_end": [pd.Timestamp("2026-01-10 12:00:00")],
                "app": ["Code"],
                "category": ["development"],
            },
        ),
        web=pd.DataFrame(),
        input=pd.DataFrame(),
    )


def test_predict_current_returns_plain_target_mapping() -> None:
    current = pd.DataFrame(
        {"user_id": ["u1"], "time_personal_distraction": [3.0]}
    )

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


def test_build_current_features_uses_only_three_latest_past_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_production_features(
        events: ParsedActivityWatchEvents,
        requests: pd.DataFrame,
        *,
        include_actionable_categories: bool = False,
    ) -> pd.DataFrame:
        assert include_actionable_categories is False
        captured["events"] = events
        captured["requests"] = requests.copy()
        result = requests.copy()
        for column in PRODUCTION_FEATURE_COLUMNS:
            result[column] = 0.0
        result["time_personal_distraction"] = np.arange(
            len(requests), dtype=float
        )
        return result

    monkeypatch.setattr(
        predictor_module,
        "build_production_features",
        fake_build_production_features,
    )
    events = _active_events()
    current = pd.Timestamp("2026-01-10 12:00:00")
    previous = [
        pd.Timestamp("2026-01-09 12:00:00"),
        pd.Timestamp("2026-01-01 12:00:00"),
        pd.Timestamp("2026-01-08 12:00:00"),
        pd.Timestamp("2026-01-08 12:00:00"),
        pd.Timestamp("2026-01-11 12:00:00"),
        pd.Timestamp("2026-01-07 12:00:00"),
        current,
    ]

    result = build_current_features(
        events,
        "u1",
        current,
        previous,
    )

    requests = captured["requests"]
    assert isinstance(requests, pd.DataFrame)
    assert list(requests["timestamp"]) == [
        pd.Timestamp("2026-01-07 12:00:00"),
        pd.Timestamp("2026-01-08 12:00:00"),
        pd.Timestamp("2026-01-09 12:00:00"),
        current,
    ]
    assert list(requests["user_id"]) == ["u1"] * 4
    captured_events = captured["events"]
    assert isinstance(captured_events, ParsedActivityWatchEvents)
    assert len(captured_events.window) == 1
    assert result.loc[0, "user_id"] == "u1"
    assert result.loc[0, "timestamp"] == current
    assert result.loc[0, "time_personal_distraction"] == 3.0
    assert list(result.columns) == [
        "user_id",
        "timestamp",
        *PRODUCTION_FEATURE_COLUMNS,
        *all_history_columns(),
    ]


def test_build_current_features_rejects_no_recent_activity() -> None:
    with pytest.raises(ValueError, match="no window or web activity"):
        build_current_features(
            _empty_events(),
            "u1",
            "2026-01-10 12:00:00",
            [],
        )


def test_build_current_features_ignores_future_activity() -> None:
    events = _active_events()
    future = events.window.copy()
    future["timestamp"] = pd.Timestamp("2026-01-10 12:30:00")
    future["event_end"] = pd.Timestamp("2026-01-10 13:00:00")
    events = events._replace(window=future)

    with pytest.raises(ValueError, match="no window or web activity"):
        build_current_features(
            events,
            "u1",
            "2026-01-10 12:00:00",
            [],
        )


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
        "build_current_features_from_buckets",
        lambda raw, user_id, timestamp, questionnaire_times, self_reports: features,
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
