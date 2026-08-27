"""Exercise the optional action-classifier runtime."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import trustme_xai.inference as inference_interface
from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.inference.action_classifier import (
    ActionClassifierRuntime,
    load_packaged_action_classifier,
)


class _DevelopmentProbabilityModel:
    def __init__(self, gain_per_minute: float = 0.015) -> None:
        self.gain_per_minute = gain_per_minute
        self.calls = 0

    def probability_high(self, table: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.clip(
            0.2
            + table["action_minutes_development"].to_numpy(dtype=float)
            * self.gain_per_minute,
            0.0,
            1.0,
        )


class _DeterministicClassifierModels:
    targets = list(MODEL_TARGETS)
    model_version = "action-classifier-test-v1"
    minimum_complete_history_rows = 0

    def __init__(self, gain_per_minute: float) -> None:
        self.estimators = {
            target: _DevelopmentProbabilityModel(gain_per_minute)
            for target in MODEL_TARGETS
        }

    def predict_high(self, table: pd.DataFrame) -> dict[str, float]:
        return {
            target: float(self.probability_high(target, table)[0])
            for target in self.targets
        }

    def probability_high(self, target: str, table: pd.DataFrame) -> np.ndarray:
        return self.estimators[target].probability_high(table)


def _runtime(
    gain_per_minute: float = 0.015,
) -> tuple[ActionClassifierRuntime, dict[str, _DevelopmentProbabilityModel]]:
    models = _DeterministicClassifierModels(gain_per_minute)
    return ActionClassifierRuntime._from_models(models), models.estimators


def _event(
    timestamp: str,
    duration_minutes: float,
    app: str,
    title: str,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "duration": duration_minutes * 60.0,
        "data": {"app": app, "title": title},
    }


def _buckets() -> dict[str, dict[str, object]]:
    return {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                _event("2026-07-30T09:30:00+02:00", 10, "Code", "main.py"),
                _event("2026-07-30T09:40:00+02:00", 10, "Zoom", "Meeting"),
                _event("2026-07-30T09:50:00+02:00", 5, "Steam", "Game"),
                _event("2026-07-30T09:55:00+02:00", 5, "ChatGPT", "Chat"),
            ],
        },
    }


def _personal_distraction_buckets() -> dict[str, dict[str, object]]:
    return {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                _event(
                    "2026-07-30T09:30:00+02:00",
                    30,
                    "Firefox",
                    "reddit.com",
                ),
            ],
        },
    }


def test_prepare_snapshot_calculates_canonical_action_allocations() -> None:
    runtime, _ = _runtime()

    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    assert snapshot.model_version == "action-classifier-test-v1"
    assert snapshot.action_allocation == {
        "development": 10.0,
        "writing": 0.0,
        "research": 0.0,
        "communication": 10.0,
        "media": 5.0,
        "personal_distraction": 0.0,
        "other": 5.0,
    }
    assert [prediction.target for prediction in snapshot.predictions] == MODEL_TARGETS
    assert all(
        prediction.probability_high == pytest.approx(0.35)
        for prediction in snapshot.predictions
    )
    assert all(prediction.state == "not_high" for prediction in snapshot.predictions)


def test_counterfactual_scores_only_the_requested_target() -> None:
    runtime, estimators = _runtime()
    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_personal_distraction_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )
    calls_after_prepare = {
        target: estimator.calls for target, estimator in estimators.items()
    }

    suggestion = runtime.generate_counterfactual(snapshot, "productivity")

    assert suggestion.target == "productivity"
    assert suggestion.strength == "Greatly improve"
    assert suggestion.baseline_probability_high == pytest.approx(0.2)
    assert suggestion.projected_probability_high == pytest.approx(0.5)
    assert suggestion.delta_probability_high == pytest.approx(0.3)
    assert [change.category for change in suggestion.changes] == [
        "development",
        "personal_distraction",
    ]
    assert [change.delta_minutes for change in suggestion.changes] == [20.0, -20.0]
    assert estimators["productivity"].calls > calls_after_prepare["productivity"]
    assert all(
        estimator.calls == calls_after_prepare[target]
        for target, estimator in estimators.items()
        if target != "productivity"
    )


def test_counterfactual_keeps_current_allocation_when_effect_is_too_small() -> None:
    runtime, _ = _runtime(gain_per_minute=0.001)
    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_personal_distraction_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    suggestion = runtime.generate_counterfactual(snapshot, "stress_management")

    assert suggestion.strength == "Keep"
    assert suggestion.projected_probability_high == pytest.approx(
        suggestion.baseline_probability_high,
    )
    assert suggestion.delta_probability_high == 0.0
    assert suggestion.changes == ()


def test_counterfactual_rejects_an_unknown_target() -> None:
    runtime, _ = _runtime()
    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    with pytest.raises(ValueError, match="target not found"):
        runtime.generate_counterfactual(snapshot, "unknown")


def test_snapshot_exposes_copies_of_mapping_values() -> None:
    runtime, _ = _runtime()
    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    allocation: Mapping[str, float] = snapshot.action_allocation
    assert dict(allocation)["development"] == 10.0
    with pytest.raises(TypeError):
        allocation["development"] = 0.0  # type: ignore[index]


def test_snapshot_id_is_stable_for_backend_cache_keys() -> None:
    runtime, _ = _runtime()
    first = runtime.prepare_snapshot(
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )
    second = runtime.prepare_snapshot(
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )
    changed = runtime.prepare_snapshot(
        activitywatch_buckets=_personal_distraction_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_id != changed.snapshot_id
    suggestion = runtime.generate_counterfactual(first, "productivity")
    assert suggestion.snapshot_id == first.snapshot_id


def test_deployed_action_classifier_supports_all_targets_end_to_end() -> None:
    runtime = load_packaged_action_classifier()
    history = {
        "timestamp": "2026-07-29T10:00:00+02:00",
        **{target: 3.0 for target in MODEL_TARGETS},
    }

    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_personal_distraction_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
        past_self_reports=[history],
    )

    assert runtime.model_version == "action_classifier_v1"
    assert [prediction.target for prediction in snapshot.predictions] == MODEL_TARGETS
    assert all(
        0.0 <= prediction.probability_high <= 1.0
        for prediction in snapshot.predictions
    )
    for target in MODEL_TARGETS:
        suggestion = runtime.generate_counterfactual(snapshot, target)
        assert suggestion.target == target
        assert suggestion.delta_probability_high >= 0.0
        total_delta = sum(change.delta_minutes for change in suggestion.changes)
        assert total_delta == pytest.approx(0.0)
        if suggestion.strength == "Keep":
            assert suggestion.changes == ()
        elif suggestion.strength == "Improve slightly":
            assert 0.10 <= suggestion.delta_probability_high < 0.25
        else:
            assert suggestion.delta_probability_high >= 0.25


def test_public_interface_hides_classifier_artifact_representation() -> None:
    runtime = inference_interface.load_packaged_action_classifier()
    history = {
        "timestamp": "2026-07-29T10:00:00+02:00",
        **{target: 3.0 for target in MODEL_TARGETS},
    }

    snapshot = runtime.prepare_snapshot(
        activitywatch_buckets=_personal_distraction_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
        past_self_reports=[history],
    )
    suggestion = runtime.generate_counterfactual(snapshot, "productivity")

    assert isinstance(runtime, ActionClassifierRuntime)
    assert runtime.model_version == "action_classifier_v1"
    assert suggestion.target == "productivity"
    assert not hasattr(runtime, "target_models")
    assert not hasattr(runtime, "metadata")
    assert "ActionClassifierBundle" not in inference_interface.__all__
    assert "ActionClassifierTargetModel" not in inference_interface.__all__
    assert "prepare_classifier_snapshot" not in inference_interface.__all__
    assert "generate_target_counterfactual" not in inference_interface.__all__
