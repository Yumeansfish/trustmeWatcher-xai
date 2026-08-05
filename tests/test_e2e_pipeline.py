"""Exercise the production feature, prediction, and counterfactual path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.inference.counterfactual_service import run_counterfactual
from trustme_xai.inference.inference_service import run_inference
from trustme_xai.inference.model_runtime import ModelBundle, TargetModel


class _IdentityPreprocessor:
    feature_columns = list(PRODUCTION_FEATURE_COLUMNS)

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.copy()


class _ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return np.full(len(table), self.value)


def _bundle() -> ModelBundle:
    models = {
        target: TargetModel(
            target=target,
            model_name="dummy",
            feature_set="test_features",
            feature_columns=list(PRODUCTION_FEATURE_COLUMNS),
            prediction_frame="absolute",
            estimator=_ConstantEstimator(3.0),
            preprocessor=_IdentityPreprocessor(),
            metrics={"validation_mse": 1.0, "test_mse": 1.0},
            fallback_score=3.0,
            score_range=(0.0, 6.0),
        )
        for target in MODEL_TARGETS
    }
    return ModelBundle(
        feature_set="test_features",
        feature_columns=list(PRODUCTION_FEATURE_COLUMNS),
        targets=list(MODEL_TARGETS),
        target_models=models,
        metadata={
            "model_version": "test-v1",
            "normalization": "global",
            "preprocessor_fit_scope": "train_plus_validation",
        },
    )


def _buckets() -> dict[str, dict[str, object]]:
    return {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                {
                    "timestamp": "2026-07-30T10:00:00+02:00",
                    "duration": 1800.0,
                    "data": {"app": "Code", "title": "Project"},
                }
            ],
        },
        "input": {
            "type": "os.hid.input",
            "hostname": "test-host",
            "events": [
                {
                    "timestamp": "2026-07-30T10:10:00+02:00",
                    "duration": 30.0,
                    "data": {
                        "presses": 20,
                        "clicks": 4,
                        "mouse_distance": 100.0,
                        "scroll_abs": 20.0,
                    },
                }
            ],
        },
    }


def test_end_to_end_runtime_contract() -> None:
    timestamp = pd.Timestamp("2026-07-30T10:30:00+02:00")

    report = run_inference(
        bundle=_bundle(),
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of=timestamp,
    )
    counterfactual = run_counterfactual(
        bundle=_bundle(),
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of=timestamp,
        target="productivity",
        desired_score=4.0,
    )

    assert [item["target"] for item in report["predictions"]] == MODEL_TARGETS
    assert counterfactual["target"] == "productivity"
    assert abs(
        sum(shift["delta_minutes"] for shift in counterfactual["shifts"])
    ) < 1e-4
