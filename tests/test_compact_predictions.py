"""Tests for the fixed compact prediction bundle."""

from __future__ import annotations

import json
from importlib import resources

import joblib
import numpy as np
import pandas as pd
import pytest

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    EVENT_DERIVED_ACTIVITY_FEATURE_COLUMNS,
)
from trustme_xai.inference.compact_contract import (
    load_compact_contract,
)
from trustme_xai.inference.inference_service import run_inference
from trustme_xai.inference.model_runtime import TargetModel, load_model_bundle


class _IdentityPreprocessor:
    feature_columns = ["history__productivity__mean", "input_load"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


def test_compact_artifact_contract_is_locked() -> None:
    contract = load_compact_contract()

    assert contract.model_version == "compact_aw_v2"
    assert contract.feature_set == "compact_aw_v2"
    assert contract.minimum_complete_history_rows == 1
    assert contract.counterfactual_targets == ()
    assert contract.normalization == "per_user"
    assert list(contract.targets) == MODEL_TARGETS
    assert {
        target: (
            target_contract.model_name,
            target_contract.prediction_frame,
            target_contract.gamma,
            len(target_contract.feature_columns),
        )
        for target, target_contract in contract.targets.items()
    } == {
        "mood_valence": ("forest_leaf5", "absolute", 1.0, 33),
        "arousal": ("extra_leaf2", "personal_residual", 1.0, 18),
        "restfulness": ("hist_leaf10", "absolute", 1.0, 20),
        "stress_management": ("hist_leaf20", "absolute", 1.0, 18),
        "productivity": ("extra_leaf2", "personal_residual", 0.75, 33),
        "engagement": ("hist_leaf10", "absolute", 0.75, 24),
        "overall_wellbeing": ("forest_leaf5", "absolute", 0.75, 16),
    }
    assert {
        target: len(target_contract.feature_columns)
        for target, target_contract in contract.targets.items()
    } == {
        "mood_valence": 33,
        "arousal": 18,
        "restfulness": 20,
        "stress_management": 18,
        "productivity": 33,
        "engagement": 24,
        "overall_wellbeing": 16,
    }


def test_every_compact_target_uses_event_derived_activitywatch_features() -> None:
    contract = load_compact_contract()
    activity_features = set(EVENT_DERIVED_ACTIVITY_FEATURE_COLUMNS)
    selected_activity = {
        target: set(target_contract.feature_columns).intersection(activity_features)
        for target, target_contract in contract.targets.items()
    }

    assert all(selected_activity.values())
    assert {
        target: selected_activity[target]
        for target in ("arousal", "restfulness", "stress_management")
    } == {
        "arousal": {
            "context_24h_ratio_personal_distraction",
            "current60_state_share_4",
        },
        "restfulness": {
            "current60_state_share_4",
            "prior7d_state_share_5",
        },
        "stress_management": {
            "current60_state_share_4",
            "prior24h_state_share_0",
        },
    }


def test_aw_target_artifact_metrics_beat_the_previous_bundle() -> None:
    artifact = resources.files("trustme_xai").joinpath("current.joblib")
    with resources.as_file(artifact) as path:
        bundle = load_model_bundle(path)
    previous = {
        "arousal": (0.8910882037142006, 0.5638086506946122),
        "restfulness": (0.8556202257450447, 0.7706652760864541),
        "stress_management": (0.6939748226293037, 0.9724502977806789),
    }

    for target, (validation_mse, test_mse) in previous.items():
        metrics = bundle.target_models[target].metrics
        assert metrics["validation_mse"] < validation_mse
        assert metrics["test_mse"] < test_mse


def test_compact_contract_rejects_changed_runtime_metadata() -> None:
    artifact = resources.files("trustme_xai").joinpath("current.joblib")
    with resources.as_file(artifact) as path:
        bundle = joblib.load(path)
    bundle.metadata = {**bundle.metadata, "normalization": "global"}

    with pytest.raises(ValueError, match="normalization does not match"):
        load_compact_contract().validate_bundle(bundle)


@pytest.mark.parametrize(
    ("prediction_frame", "estimator_value", "expected"),
    [
        ("absolute", 5.0, 4.0),
        ("personal_residual", 2.0, 2.5),
    ],
)
def test_target_model_applies_saved_gamma(
    prediction_frame: str,
    estimator_value: float,
    expected: float,
) -> None:
    model = TargetModel(
        target="productivity",
        model_name="dummy",
        feature_set="test",
        feature_columns=list(_IdentityPreprocessor.feature_columns),
        prediction_frame=prediction_frame,
        estimator=_ConstantEstimator(estimator_value),
        preprocessor=_IdentityPreprocessor(),
        metrics={},
        fallback_score=3.0,
        blend_gamma=0.75,
        score_range=(0.0, 6.0),
    )
    table = pd.DataFrame(
        {"history__productivity__mean": [1.0], "input_load": [0.0]},
    )

    assert model.predict(table).iloc[0] == pytest.approx(expected)


def test_behavior_model_records_train_only_provenance() -> None:
    artifact = resources.files("trustme_xai.feature_pipeline").joinpath(
        "behavior_state_model.json",
    )
    with artifact.open(encoding="utf-8") as handle:
        provenance = json.load(handle)["provenance"]

    assert provenance["fit_hour_rows"] == 4497
    assert provenance["n_clusters"] == 6
    assert provenance["n_init"] == 20
    assert provenance["random_state"] == 42
    assert provenance["validation_or_test_activity_used"] is False
    assert provenance["state_feature_parity_max_abs_diff"] < 1e-12


def test_compact_model_predicts_without_self_report_history() -> None:
    artifact = resources.files("trustme_xai").joinpath("current.joblib")
    with resources.as_file(artifact) as path:
        bundle = load_model_bundle(path)
    buckets = {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                {
                    "timestamp": "2026-07-30T09:30:00+02:00",
                    "duration": 30 * 60.0,
                    "data": {"app": "Code", "title": "main.py"},
                }
            ],
        }
    }

    report = run_inference(
        bundle=bundle,
        activitywatch_buckets=buckets,
        user_id="new_user",
        as_of="2026-07-30T10:00:00+02:00",
    )

    assert [
        prediction["target"] for prediction in report["predictions"]
    ] == MODEL_TARGETS
    assert all(
        0.0 <= prediction["prediction"] <= 6.0
        for prediction in report["predictions"]
    )
