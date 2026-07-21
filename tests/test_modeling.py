"""Test runtime model loading and prediction"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trustme_xai.inference import model_runtime as bundle_module
from trustme_xai.inference.model_runtime import (
    ModelBundle,
    PerUserStandardizer,
    TargetModel,
)
from trustme_xai.modeling.bundle import (
    ModelBundle as LegacyModelBundle,
)
from trustme_xai.modeling.bundle import (
    TargetModel as LegacyTargetModel,
)
from trustme_xai.modeling.preprocessing import (
    PerUserStandardizer as LegacyPerUserStandardizer,
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
            estimator=_OffsetSumEstimator(offset),
            preprocessor=_IdentityPreprocessor(feature_columns),
            metrics={},
        )
        for target, offset in (("q1_feelings", 0.0), ("q8_stress", 10.0))
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

    assert list(predictions.columns) == ["q1_feelings", "q8_stress"]
    np.testing.assert_allclose(
        predictions.to_numpy(),
        [[3.0, 13.0], [7.0, 17.0]],
    )


def test_legacy_joblib_classes_alias_runtime_classes() -> None:
    assert LegacyModelBundle is ModelBundle
    assert LegacyTargetModel is TargetModel
    assert LegacyPerUserStandardizer is PerUserStandardizer


def test_model_bundle_rejects_missing_features() -> None:
    table = pd.DataFrame({"user_id": ["u1"], "x": [1.0]})

    with pytest.raises(ValueError, match=r"missing model features: \['y'\]"):
        _model_bundle().predict(table)


def test_per_user_standardizer_uses_saved_stats() -> None:
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


def test_load_model_bundle_checks_hash_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"unexpected model bytes")

    def unexpected_load(path: Path) -> object:
        raise AssertionError(f"joblib.load should not read {path}")

    monkeypatch.setattr(bundle_module.joblib, "load", unexpected_load)

    with pytest.raises(ValueError, match="hash does not match"):
        bundle_module._load_model_bundle(path, expected_sha256="0" * 64)


def test_load_model_bundle_accepts_matching_hash_and_bundle_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"trusted model bytes"
    path = tmp_path / "model.joblib"
    path.write_bytes(payload)
    expected = _model_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(bundle_module.joblib, "load", lambda _: expected)

    loaded = bundle_module._load_model_bundle(path, expected_sha256=digest)

    assert loaded is expected


def test_load_model_bundle_rejects_matching_hash_with_wrong_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"trusted but invalid model bytes"
    path = tmp_path / "model.joblib"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(bundle_module.joblib, "load", lambda _: object())

    with pytest.raises(TypeError, match="not a ModelBundle"):
        bundle_module._load_model_bundle(path, expected_sha256=digest)
