"""Load the dashboard model and apply its saved preprocessing"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd

CURRENT_MODEL_SHA256 = (
    "4a7db8320c610f33fc1394f5a2d240fd160dc5b90eb32393a92408d0bde2bbf1"
)


class FeaturePreprocessor(Protocol):
    feature_columns: list[str]

    def transform(self, table: pd.DataFrame) -> pd.DataFrame: ...


def numeric_features(table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    values = table[feature_columns].astype(float)
    return cast(pd.DataFrame, values.replace([np.inf, -np.inf], np.nan).fillna(0.0))


@dataclass
class PerUserStandardizer:
    feature_columns: list[str]
    user_stats_: dict[str, tuple[pd.Series, pd.Series]] = field(default_factory=dict)
    global_stats_: tuple[pd.Series, pd.Series] | None = None

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.global_stats_ is None:
            raise RuntimeError("standardizer must be fitted before transform")
        if "user_id" not in table.columns:
            raise ValueError("per-user standardizer needs user_id")

        out = table.copy()
        values = numeric_features(out, self.feature_columns)
        global_mean, global_std = self.global_stats_
        for user_id, rows in out.groupby("user_id", sort=False):
            mean, std = self.user_stats_.get(str(user_id), (global_mean, global_std))
            out.loc[rows.index, self.feature_columns] = (
                values.loc[rows.index] - mean
            ) / std
        return cast(pd.DataFrame, out)


@dataclass
class TargetModel:
    target: str
    # These fields preserve the current.joblib serialization contract.
    model_name: str
    estimator: Any
    preprocessor: FeaturePreprocessor
    metrics: dict[str, float]

    def predict(self, table: pd.DataFrame) -> pd.Series:
        transformed = self.preprocessor.transform(table)
        prediction = self.estimator.predict(
            transformed[self.preprocessor.feature_columns]
        )
        return pd.Series(prediction, index=table.index, name=self.target)


@dataclass
class ModelBundle:
    feature_set: str
    feature_columns: list[str]
    targets: list[str]
    target_models: dict[str, TargetModel]
    metadata: dict[str, Any]

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column for column in self.feature_columns if column not in table.columns
        ]
        if missing:
            raise ValueError(f"table is missing model features: {missing}")
        return pd.concat(
            [self.target_models[target].predict(table) for target in self.targets],
            axis=1,
        )


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load one trusted model bundle

    Args:
        path: path to the model artifact

    Returns:
        loaded model bundle
    """

    return _load_model_bundle(path, CURRENT_MODEL_SHA256)


def _load_model_bundle(path: str | Path, expected_sha256: str) -> ModelBundle:
    """Load a model using an explicit digest for isolated tests"""

    model_path = Path(path)
    payload = model_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("model artifact hash does not match the trusted digest")

    loaded = joblib.load(BytesIO(payload))
    if not isinstance(loaded, ModelBundle):
        raise TypeError("loaded artifact is not a ModelBundle")
    return loaded
