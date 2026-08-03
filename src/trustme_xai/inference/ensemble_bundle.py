"""Store fitted model bundles for live prediction"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

import numpy as np
import pandas as pd

ENSEMBLE_BUNDLE_SCHEMA_VERSION = 2
NUM_BLOCKS = 5


class TargetMetric(StrEnum):
    STRESS = "stress"
    FATIGUE = "fatigue"
    VALENCE = "valence"
    AROUSAL = "arousal"
    PRODUCTIVITY = "productivity"
    ENGAGEMENT = "engagement"
    OVERALL_WELLBEING = "overall_wellbeing"


class ModelFamily(StrEnum):
    RIDGE = "ridge"
    LASSO = "lasso"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FOREST = "random_forest"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"



def _empty_metadata() -> dict[str, str]:
    return {}


@dataclass
class EnsembleTargetModel:
    """Store fitted regressors for one target metric"""

    target: TargetMetric
    family: ModelFamily
    block_models: list[Any]
    user_means: dict[str, float]
    global_mean: float
    feature_columns: list[str]
    avg_val_mse: float

    def predict(self, feature_row: pd.DataFrame) -> float:
        """Compute live prediction from fitted model

        Args:
            feature_row: single-row pd.DataFrame containing required model features

        Returns:
            continuous float prediction
        """
        missing = [c for c in self.feature_columns if c not in feature_row.columns]
        if missing:
            raise ValueError(f"feature_row missing required columns for {self.target}: {missing}")

        X = feature_row[self.feature_columns].to_numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)

        block_preds = [float(model.predict(X)[0]) for model in self.block_models]
        return float(np.mean(block_preds))


@dataclass
class EnsembleBundle:
    """Store fitted model bundle across all target metrics"""


    targets: list[TargetMetric]
    target_models: dict[TargetMetric, EnsembleTargetModel]
    feature_columns: list[str]
    metadata: dict[str, str] = field(default_factory=_empty_metadata)
    schema_version: int = ENSEMBLE_BUNDLE_SCHEMA_VERSION

    def predict_target(self, feature_row: pd.DataFrame, target: TargetMetric | str) -> float:
        """Predict continuous score for one specified target

        Args:
            feature_row: single-row pd.DataFrame
            target: target metric name

        Returns:
            averaged float prediction
        """
        metric_key = TargetMetric(target)
        if metric_key not in self.target_models:
            raise ValueError(f"target {target} not found in model bundle")
        return self.target_models[metric_key].predict(feature_row)

    def predict(self, feature_row: pd.DataFrame) -> dict[str, float]:
        """Predict continuous scores across all saved targets

        Args:
            feature_row: single-row pd.DataFrame

        Returns:
            dict mapping target string to continuous prediction
        """
        return {
            str(target): self.target_models[target].predict(feature_row)
            for target in self.targets
        }
