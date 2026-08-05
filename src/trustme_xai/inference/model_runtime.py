"""Store fitted production models and feature normalization"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

MODEL_BUNDLE_SCHEMA_VERSION = 3
MIN_SCORE = 0.0
MAX_SCORE = 6.0


class FeaturePreprocessor(Protocol):
    """Define the feature normalization interface"""

    feature_columns: list[str]

    def fit(
        self,
        table: pd.DataFrame,
        row_split: pd.Series,
    ) -> FeaturePreprocessor: ...

    def transform(self, table: pd.DataFrame) -> pd.DataFrame: ...


def numeric_features(
    table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Read finite numeric model features

    Args:
        table: feature table
        feature_columns: columns required by the model

    Returns:
        pd.DataFrame with finite float values
    """
    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise ValueError(f"missing model features: {missing}")
    values = table[feature_columns].apply(pd.to_numeric, errors="raise").astype(float)
    bad_columns = [
        column
        for column in feature_columns
        if not np.isfinite(values[column].to_numpy(dtype=float)).all()
    ]
    if bad_columns:
        raise ValueError(f"model features are not finite: {bad_columns}")
    return values


def mean_and_std(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate feature means and safe standard deviations

    Args:
        values: finite numeric features

    Returns:
        column means and standard deviations
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0).replace(0.0, 1.0)
    return mean, std


@dataclass
class GlobalStandardizer:
    """Normalize all participants with saved training statistics"""

    feature_columns: list[str]
    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    def fit(
        self,
        table: pd.DataFrame,
        row_split: pd.Series,
    ) -> GlobalStandardizer:
        """Fit global statistics on training rows

        Args:
            table: complete feature table
            row_split: split name for each row

        Returns:
            fitted GlobalStandardizer
        """
        if not table.index.equals(row_split.index):
            raise ValueError("row_split index must match the feature table")
        train = numeric_features(
            table.loc[row_split == "train"],
            self.feature_columns,
        )
        if train.empty:
            raise ValueError("global standardizer needs training rows")
        self.mean_, self.std_ = mean_and_std(train)
        return self

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Normalize one feature table

        Args:
            table: feature table to normalize

        Returns:
            normalized pd.DataFrame
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("global standardizer must be fitted")
        result = table.copy()
        values = numeric_features(result, self.feature_columns)
        result[self.feature_columns] = (values - self.mean_) / self.std_
        return result


@dataclass
class PerUserStandardizer:
    """Normalize each participant with saved training statistics"""

    feature_columns: list[str]
    user_stats_: dict[str, tuple[pd.Series, pd.Series]] = field(
        default_factory=dict,
    )
    global_stats_: tuple[pd.Series, pd.Series] | None = None

    def fit(
        self,
        table: pd.DataFrame,
        row_split: pd.Series,
    ) -> PerUserStandardizer:
        """Fit participant statistics on training rows

        Args:
            table: complete feature table
            row_split: split name for each row

        Returns:
            fitted PerUserStandardizer
        """
        if not table.index.equals(row_split.index):
            raise ValueError("row_split index must match the feature table")
        if "user_id" not in table.columns:
            raise ValueError("per-user standardizer needs user_id")
        train = table.loc[row_split == "train"]
        if train.empty:
            raise ValueError("per-user standardizer needs training rows")

        self.user_stats_ = {}
        self.global_stats_ = mean_and_std(
            numeric_features(train, self.feature_columns),
        )
        for user_id, rows in train.groupby("user_id", sort=False):
            self.user_stats_[str(user_id)] = mean_and_std(
                numeric_features(rows, self.feature_columns),
            )
        return self

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Normalize one feature table by participant

        Args:
            table: feature table to normalize

        Returns:
            normalized pd.DataFrame
        """
        if not self.user_stats_ or self.global_stats_ is None:
            raise RuntimeError("per-user standardizer must be fitted")
        if "user_id" not in table.columns:
            raise ValueError("per-user standardizer needs user_id")
        users = table["user_id"]
        if bool(users.isna().any()) or bool(users.astype(str).str.strip().eq("").any()):
            raise ValueError("per-user standardizer needs non-empty user_id")

        result = table.copy()
        values = numeric_features(result, self.feature_columns)
        normalized = values.copy()
        for user_id, rows in result.groupby("user_id", sort=False):
            mean, std = self.user_stats_.get(
                str(user_id),
                self.global_stats_,
            )
            normalized.loc[rows.index] = (values.loc[rows.index] - mean) / std
        result[self.feature_columns] = normalized
        return result


def make_preprocessor(
    normalization: str,
    feature_columns: list[str],
) -> FeaturePreprocessor:
    """Build one feature preprocessor

    Args:
        normalization: global or per_user
        feature_columns: model feature columns

    Returns:
        new feature preprocessor
    """
    if normalization == "global":
        return GlobalStandardizer(list(feature_columns))
    if normalization == "per_user":
        return PerUserStandardizer(list(feature_columns))
    raise ValueError(f"unknown normalization strategy: {normalization}")


def clip_predictions(values: object) -> np.ndarray:
    """Clip finite predictions to the display range

    Args:
        values: estimator predictions

    Returns:
        one clipped float array
    """
    predictions = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(predictions).all():
        raise ValueError("model predictions must be finite")
    return np.clip(predictions, MIN_SCORE, MAX_SCORE)


@dataclass
class TargetModel:
    """Store the selected estimator for one positive target"""

    target: str
    model_name: str
    estimator: Any
    preprocessor: FeaturePreprocessor
    metrics: dict[str, int | float]
    score_range: tuple[float, float] | None = None

    @property
    def feature_columns(self) -> list[str]:
        """Return ordered model feature names

        Returns:
            saved feature columns
        """
        return list(self.preprocessor.feature_columns)

    def predict(self, table: pd.DataFrame) -> pd.Series:
        """Predict one positive target

        Args:
            table: rows with model features

        Returns:
            pd.Series with clipped predictions
        """
        transformed = self.preprocessor.transform(table)
        values = self.estimator.predict(transformed[self.feature_columns])
        predictions = np.asarray(values, dtype=float).reshape(-1)
        if not np.isfinite(predictions).all():
            raise ValueError("model predictions must be finite")
        if self.score_range is not None:
            predictions = np.clip(predictions, *self.score_range)
        return pd.Series(
            predictions,
            index=table.index,
            name=self.target,
            dtype=float,
        )


@dataclass
class ModelBundle:
    """Store all seven fitted production models"""

    feature_set: str
    feature_columns: list[str]
    targets: list[str]
    target_models: dict[str, TargetModel]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MODEL_BUNDLE_SCHEMA_VERSION

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """Predict every saved target

        Args:
            table: rows with all model features

        Returns:
            pd.DataFrame with one column per target
        """
        numeric_features(table, self.feature_columns)
        return pd.concat(
            [self.target_models[target].predict(table) for target in self.targets],
            axis=1,
        )

    def predict_target(self, table: pd.DataFrame, target: str) -> float:
        """Predict one target for one row

        Args:
            table: one row with model features
            target: saved target name

        Returns:
            clipped prediction
        """
        if len(table) != 1:
            raise ValueError("predict_target needs exactly one row")
        if target not in self.target_models:
            raise ValueError(f"target not found in model bundle: {target}")
        return float(self.target_models[target].predict(table).iloc[0])


def validate_model_bundle(bundle: ModelBundle) -> None:
    """Validate one saved model bundle

    Args:
        bundle: model bundle to validate

    Returns:
        None
    """
    if bundle.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported model bundle schema: {bundle.schema_version}")
    if not bundle.feature_set.strip():
        raise ValueError("model bundle needs a feature set")
    if not bundle.feature_columns:
        raise ValueError("model bundle needs feature columns")
    if len(set(bundle.feature_columns)) != len(bundle.feature_columns):
        raise ValueError("model bundle feature columns must be unique")
    if not bundle.targets or len(set(bundle.targets)) != len(bundle.targets):
        raise ValueError("model bundle targets must be unique and non-empty")
    if set(bundle.target_models) != set(bundle.targets):
        raise ValueError("model bundle target keys do not match targets")

    for target in bundle.targets:
        model = bundle.target_models[target]
        if not isinstance(model, TargetModel):
            raise TypeError(f"{target} is not a TargetModel")
        if model.target != target:
            raise ValueError(f"target model key does not match: {target}")
        if model.feature_columns != bundle.feature_columns:
            raise ValueError(f"{target} feature columns do not match the bundle")
        if not model.model_name:
            raise ValueError(f"{target} has no model name")

    if bundle.feature_set == "production_25":
        from trustme_xai.contracts import MODEL_TARGETS
        from trustme_xai.feature_pipeline.production_features import (
            PRODUCTION_FEATURE_COLUMNS,
        )

        if bundle.targets != MODEL_TARGETS:
            raise ValueError("production bundle targets do not match the contract")
        if bundle.feature_columns != PRODUCTION_FEATURE_COLUMNS:
            raise ValueError("production bundle features do not match the contract")
        model_version = bundle.metadata.get("model_version")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("production bundle is missing model_version")
        if bundle.metadata.get("preprocessor_fit_scope") != "train_plus_validation":
            raise ValueError("production preprocessors must use development rows")
        if any(
            bundle.target_models[target].score_range != (MIN_SCORE, MAX_SCORE)
            for target in bundle.targets
        ):
            raise ValueError("production targets must use the 0..6 score range")


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    """Save one model bundle

    Args:
        bundle: fitted model bundle
        path: output joblib path

    Returns:
        None
    """
    validate_model_bundle(bundle)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load and validate one model bundle

    Args:
        path: saved joblib path

    Returns:
        loaded ModelBundle
    """
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, ModelBundle):
        raise TypeError(f"loaded object is not a ModelBundle: {type(loaded)}")
    validate_model_bundle(loaded)
    return loaded
