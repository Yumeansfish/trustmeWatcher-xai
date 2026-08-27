"""Load production models and apply saved feature normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

MODEL_BUNDLE_SCHEMA_VERSION = 5
SUPPORTED_MODEL_BUNDLE_SCHEMA_VERSIONS = frozenset({4, MODEL_BUNDLE_SCHEMA_VERSION})
MIN_SCORE = 0.0
MAX_SCORE = 6.0


class FeaturePreprocessor(Protocol):
    """Define the runtime feature-normalization interface."""

    feature_columns: list[str]

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


def numeric_model_features(
    table: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Read numeric model features while keeping missing history

    Args:
        table: feature table
        feature_columns: columns required by the model

    Returns:
        numeric model input with missing values unchanged
    """
    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise ValueError(f"missing model features: {missing}")
    values = table[feature_columns].apply(pd.to_numeric, errors="raise").astype(float)
    infinite = np.isinf(values.to_numpy(dtype=float)).any(axis=0)
    bad_columns = [
        column
        for column, is_infinite in zip(feature_columns, infinite, strict=True)
        if is_infinite
    ]
    if bad_columns:
        raise ValueError(f"model features are infinite: {bad_columns}")
    return values


@dataclass
class GlobalStandardizer:
    """Normalize all participants with artifact-provided statistics."""

    feature_columns: list[str]
    mean_: pd.Series | None = None
    std_: pd.Series | None = None

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Normalize one feature table

        Args:
            table: feature table to normalize

        Returns:
            normalized pd.DataFrame
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("model artifact has no global normalization statistics")
        result = table.copy()
        values = numeric_features(result, self.feature_columns)
        result[self.feature_columns] = (values - self.mean_) / self.std_
        return result


@dataclass
class PerUserStandardizer:
    """Normalize each participant with artifact-provided statistics."""

    feature_columns: list[str]
    user_stats_: dict[str, tuple[pd.Series, pd.Series]] = field(
        default_factory=dict,
    )
    global_stats_: tuple[pd.Series, pd.Series] | None = None

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Normalize one feature table by participant

        Args:
            table: feature table to normalize

        Returns:
            normalized pd.DataFrame
        """
        if not self.user_stats_ or self.global_stats_ is None:
            raise RuntimeError(
                "model artifact has no per-user normalization statistics",
            )
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


@dataclass
class TargetModel:
    """Store the loaded estimator for one positive target."""

    target: str
    model_name: str
    feature_set: str
    feature_columns: list[str]
    prediction_frame: str
    estimator: Any
    preprocessor: FeaturePreprocessor
    metrics: dict[str, int | float]
    fallback_score: float
    blend_gamma: float = 1.0
    score_range: tuple[float, float] | None = None

    def predict(self, table: pd.DataFrame) -> pd.Series:
        """Predict one positive target

        Args:
            table: rows with model features

        Returns:
            pd.Series with clipped predictions
        """
        transformed = self.preprocessor.transform(table)
        features = numeric_model_features(transformed, self.feature_columns)
        values = self.estimator.predict(features)
        predictions = np.asarray(values, dtype=float).reshape(-1)
        baseline: np.ndarray | None = None
        if self.prediction_frame == "personal_residual" or self.blend_gamma < 1.0:
            from trustme_xai.feature_pipeline.self_report_features import (
                history_column,
            )

            baseline = pd.to_numeric(
                table[history_column(self.target, "mean")],
                errors="coerce",
            ).fillna(self.fallback_score).to_numpy(dtype=float)
        if self.prediction_frame == "personal_residual":
            if baseline is None:
                raise RuntimeError("personal residual prediction needs a baseline")
            predictions += baseline
        if self.blend_gamma < 1.0:
            if baseline is None:
                raise RuntimeError("blended prediction needs a baseline")
            predictions = baseline + self.blend_gamma * (predictions - baseline)
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
    """Store all seven production estimators loaded from an artifact."""

    feature_set: str
    feature_columns: list[str]
    targets: list[str]
    target_models: dict[str, TargetModel]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MODEL_BUNDLE_SCHEMA_VERSION

    @property
    def model_version(self) -> str:
        """Return the validated artifact model version."""
        return str(self.metadata["model_version"])

    @property
    def minimum_complete_history_rows(self) -> int:
        """Return the number of complete prior reports required by inference."""
        return int(self.metadata.get("minimum_complete_history_rows", 0))

    @property
    def normalization(self) -> str:
        """Return the validated runtime normalization description."""
        return str(self.metadata["normalization"])

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        """Predict every saved target

        Args:
            table: rows with all model features

        Returns:
            pd.DataFrame with one column per target
        """
        numeric_model_features(table, self.feature_columns)
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
    if bundle.schema_version not in SUPPORTED_MODEL_BUNDLE_SCHEMA_VERSIONS:
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
        if not model.feature_set:
            raise ValueError(f"{target} has no feature set")
        if not model.feature_columns:
            raise ValueError(f"{target} has no feature columns")
        if len(set(model.feature_columns)) != len(model.feature_columns):
            raise ValueError(f"{target} feature columns must be unique")
        if not set(model.feature_columns).issubset(bundle.feature_columns):
            raise ValueError(f"{target} features are not in the bundle schema")
        if not set(model.preprocessor.feature_columns).issubset(
            bundle.feature_columns,
        ):
            raise ValueError(f"{target} preprocessor features are not in the bundle")
        if model.prediction_frame not in {"absolute", "personal_residual"}:
            raise ValueError(f"{target} has an unknown prediction frame")
        if not np.isfinite(model.fallback_score):
            raise ValueError(f"{target} has no finite fallback score")
        if not np.isfinite(model.blend_gamma) or not 0.0 <= model.blend_gamma <= 1.0:
            raise ValueError(f"{target} has an invalid blend gamma")
        if not model.model_name:
            raise ValueError(f"{target} has no model name")

    if bundle.feature_set == "compact_90_v1":
        from trustme_xai.inference.compact_contract import load_compact_contract

        load_compact_contract().validate_bundle(bundle)

    if bundle.feature_set == "production_25_history":
        from trustme_xai.contracts import MODEL_TARGETS
        from trustme_xai.feature_pipeline.production_features import (
            PRODUCTION_FEATURE_COLUMNS,
        )
        from trustme_xai.feature_pipeline.self_report_features import (
            all_history_columns,
            history_column,
        )

        if bundle.targets != MODEL_TARGETS:
            raise ValueError("production bundle targets do not match the contract")
        expected = [*PRODUCTION_FEATURE_COLUMNS, *all_history_columns()]
        if bundle.feature_columns != expected:
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
        for target in bundle.targets:
            model = bundle.target_models[target]
            if not set(model.feature_columns).intersection(
                PRODUCTION_FEATURE_COLUMNS,
            ):
                raise ValueError(f"{target} does not use ActivityWatch features")
            if history_column(target, "mean") not in model.feature_columns:
                raise ValueError(f"{target} does not use its past self-reports")


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
