"""Tests for zero-sum counterfactual search and its bucket service."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.inference import counterfactual_service as service_module
from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.counterfactual_service import run_counterfactual
from trustme_xai.inference.model_runtime import ModelBundle, TargetModel


class _IdentityPreprocessor:
    feature_columns = ["time_development", "time_personal_distraction"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ProductivityEstimator:
    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return (
            1.0
            + table["time_development"].to_numpy(dtype=float) / 10.0
            - table["time_personal_distraction"].to_numpy(dtype=float) / 20.0
        )


def _bundle() -> ModelBundle:
    target = TargetModel(
        target="productivity",
        model_name="dummy",
        feature_set="test_features",
        feature_columns=list(_IdentityPreprocessor.feature_columns),
        prediction_frame="absolute",
        estimator=_ProductivityEstimator(),
        preprocessor=_IdentityPreprocessor(),
        metrics={},
        fallback_score=3.0,
        score_range=(0.0, 6.0),
    )
    return ModelBundle(
        feature_set="test_features",
        feature_columns=list(_IdentityPreprocessor.feature_columns),
        targets=["productivity"],
        target_models={"productivity": target},
        metadata={},
    )


def _feature_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "timestamp": pd.Timestamp("2026-07-29 14:30:00"),
                "time_development": 5.0,
                "time_personal_distraction": 25.0,
            }
        ]
    )


def test_find_counterfactual_preserves_the_time_budget() -> None:
    result = find_counterfactual(
        bundle=_bundle(),
        feature_row=_feature_row(),
        target="productivity",
        desired_score=3.0,
    )

    assert result["participant_id"] == "u1"
    assert result["target"] == "productivity"
    assert isinstance(result["projected_score"], float)
    assert abs(sum(shift["delta_minutes"] for shift in result["shifts"])) < 1e-4


def test_run_counterfactual_builds_features_from_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_module,
        "build_runtime_feature_row",
        lambda model, activitywatch_buckets, user_id, as_of,
        previous_questionnaire_times, past_self_reports: _feature_row(),
    )

    result = run_counterfactual(
        bundle=_bundle(),
        activitywatch_buckets={},
        user_id="u1",
        as_of="2026-07-29T14:30:00",
        target="productivity",
        desired_score=3.0,
    )

    assert result["target"] == "productivity"
    assert "projected_score" in result


def test_search_does_not_treat_unrelated_minute_features_as_actionable() -> None:
    row = _feature_row()
    row["time_development"] = 0.0
    row["time_personal_distraction"] = 0.0
    row["minutes_since_prev1_window"] = 120.0

    result = find_counterfactual(
        bundle=_bundle(),
        feature_row=row,
        target="productivity",
        desired_score=3.0,
    )

    assert result["success"] is False
    assert result["shifts"] == []
