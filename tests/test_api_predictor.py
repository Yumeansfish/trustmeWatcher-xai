"""Test the runtime prediction API"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.inference import inference_service as predictor_module
from trustme_xai.inference.inference_service import (
    build_current_features,
    predict_current,
    run_inference,
)
from trustme_xai.inference.model_runtime import ModelBundle, TargetModel


class _IdentityPreprocessor:
    feature_columns = ["x"]

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
            estimator=_ConstantEstimator(value),
            preprocessor=_IdentityPreprocessor(),
            metrics={},
        )
        for target, value in (("q8_stress", 2.5), ("q9_productivity", 4.0))
    }
    return ModelBundle(
        feature_set="test_features",
        feature_columns=["x"],
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
    current = pd.DataFrame({"user_id": ["u1"], "x": [3.0]})

    predictions = predict_current(_model_bundle(), current)

    assert predictions == {"q8": 2.5, "q9": 4.0}
    assert all(isinstance(value, float) for value in predictions.values())


@pytest.mark.parametrize("row_count", [0, 2])
def test_predict_current_requires_exactly_one_row(row_count: int) -> None:
    current = pd.DataFrame({"x": np.arange(row_count, dtype=float)})

    with pytest.raises(ValueError, match="exactly one row"):
        predict_current(_model_bundle(), current)


def test_build_current_features_uses_only_three_latest_past_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_production_features(
        events: ParsedActivityWatchEvents,
        requests: pd.DataFrame,
    ) -> pd.DataFrame:
        captured["events"] = events
        captured["requests"] = requests.copy()
        return requests.assign(x=np.arange(len(requests), dtype=float))

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
    assert result.to_dict(orient="records") == [
        {"user_id": "u1", "timestamp": current, "x": 3.0},
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
    buckets: dict[str, dict[str, object]] = {"window": {"events": []}}
    events = _active_events()
    as_of = pd.Timestamp("2026-01-10 12:00:00")
    features = pd.DataFrame({"user_id": ["u1"], "timestamp": [as_of]})
    expected = {"questions": []}

    monkeypatch.setattr(
        predictor_module,
        "parse_activitywatch_events",
        lambda raw, user_id: events,
    )
    monkeypatch.setattr(
        predictor_module,
        "build_current_features",
        lambda parsed, user_id, timestamp, history: features,
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
