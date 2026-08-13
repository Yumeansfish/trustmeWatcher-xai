"""Tests for the fixed compact prediction bundle."""

from __future__ import annotations

import json
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.inference.inference_service import _require_model_history
from trustme_xai.inference.model_runtime import TargetModel
from trustme_xai.modeling.fixed_recipes import (
    COMPACT_FEATURE_COUNTS,
    load_compact_recipes,
)


class _IdentityPreprocessor:
    feature_columns = ["history__productivity__mean", "input_load"]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


def test_compact_recipe_manifest_is_locked() -> None:
    recipes = load_compact_recipes()

    assert list(recipes) == MODEL_TARGETS
    assert {
        target: (
            recipe.model_name,
            recipe.prediction_frame,
            recipe.gamma,
            len(recipe.feature_columns),
        )
        for target, recipe in recipes.items()
    } == {
        "mood_valence": ("forest_leaf5", "absolute", 1.0, 33),
        "arousal": ("extra_leaf2", "personal_residual", 1.0, 16),
        "restfulness": ("hist_leaf10", "absolute", 1.0, 18),
        "stress_management": ("extra_leaf5", "absolute", 1.0, 17),
        "productivity": ("extra_leaf2", "personal_residual", 0.75, 33),
        "engagement": ("hist_leaf10", "absolute", 0.75, 24),
        "overall_wellbeing": ("forest_leaf5", "absolute", 0.75, 16),
    }
    assert {
        target: len(recipe.feature_columns)
        for target, recipe in recipes.items()
    } == COMPACT_FEATURE_COUNTS


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


def test_compact_bundle_requires_one_complete_earlier_checkin() -> None:
    bundle = SimpleNamespace(
        targets=MODEL_TARGETS,
        metadata={
            "model_version": "compact_90_v1",
            "minimum_complete_history_rows": 1,
        },
    )
    as_of = pd.Timestamp("2026-01-02 12:00:00")

    with pytest.raises(ValueError, match="one complete earlier StreamDeck check-in"):
        _require_model_history(bundle, [], as_of)

    report = {
        "timestamp": "2026-01-01 12:00:00",
        **{target: 3.0 for target in MODEL_TARGETS},
    }
    _require_model_history(bundle, [report], as_of)


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
