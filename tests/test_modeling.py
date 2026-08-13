"""Test runtime model loading, normalization, and prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trustme_xai.inference.model_runtime import (
    ModelBundle,
    PerUserStandardizer,
    TargetModel,
    load_model_bundle,
    save_model_bundle,
)


class _IdentityPreprocessor:
    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _OffsetSumEstimator:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return table.sum(axis=1).to_numpy(dtype=float) + self.offset


def _model_bundle() -> ModelBundle:
    feature_columns = ["x", "y"]
    target_models = {
        target: TargetModel(
            target=target,
            model_name="dummy",
            feature_set="test_features",
            feature_columns=feature_columns,
            prediction_frame="absolute",
            estimator=_OffsetSumEstimator(offset),
            preprocessor=_IdentityPreprocessor(feature_columns),
            metrics={},
            fallback_score=3.0,
        )
        for target, offset in (("mood_valence", 0.0), ("productivity", 1.0))
    }
    return ModelBundle(
        feature_set="test_features",
        feature_columns=feature_columns,
        targets=list(target_models),
        target_models=target_models,
        metadata={},
    )


def test_model_bundle_predicts_all_targets() -> None:
    table = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "x": [1.0, 3.0],
            "y": [2.0, 4.0],
        },
    )

    predictions = _model_bundle().predict(table)

    assert list(predictions.columns) == ["mood_valence", "productivity"]
    np.testing.assert_allclose(predictions.to_numpy(), [[3.0, 4.0], [7.0, 8.0]])


def test_model_bundle_rejects_missing_features() -> None:
    table = pd.DataFrame({"user_id": ["u1"], "x": [1.0]})

    with pytest.raises(ValueError, match=r"missing model features: \['y'\]"):
        _model_bundle().predict(table)


def test_per_user_standardizer_uses_saved_stats_and_global_fallback() -> None:
    preprocessor = PerUserStandardizer(
        feature_columns=["x", "y"],
        user_stats_={
            "known": (
                pd.Series({"x": 1.0, "y": 2.0}),
                pd.Series({"x": 1.0, "y": 2.0}),
            ),
        },
        global_stats_=(
            pd.Series({"x": 10.0, "y": 20.0}),
            pd.Series({"x": 2.0, "y": 4.0}),
        ),
    )
    table = pd.DataFrame(
        {
            "user_id": ["known", "new"],
            "x": [3.0, 12.0],
            "y": [6.0, 28.0],
        },
    )

    transformed = preprocessor.transform(table)

    np.testing.assert_allclose(
        transformed[["x", "y"]].to_numpy(dtype=float),
        [[2.0, 2.0], [1.0, 2.0]],
    )


def test_model_bundle_round_trips_through_joblib(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    expected = _model_bundle()

    save_model_bundle(expected, path)
    loaded = load_model_bundle(path)

    assert loaded.feature_set == expected.feature_set
    assert loaded.targets == expected.targets
    assert loaded.feature_columns == expected.feature_columns


def test_load_model_bundle_accepts_schema_four_without_gamma(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "schema_four.joblib"
    expected = _model_bundle()
    expected.schema_version = 4
    for model in expected.target_models.values():
        model.__dict__.pop("blend_gamma", None)
    joblib.dump(expected, path)

    loaded = load_model_bundle(path)

    assert loaded.schema_version == 4
    assert all(model.blend_gamma == 1.0 for model in loaded.target_models.values())


def test_load_model_bundle_rejects_wrong_object(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump({"not": "a model"}, path)

    with pytest.raises(TypeError, match="not a ModelBundle"):
        load_model_bundle(path)
