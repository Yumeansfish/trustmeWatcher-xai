"""Explain one prediction with local shap values"""

from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
import shap  # type: ignore[reportMissingTypeStubs]
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from trustme_xai.inference.model_runtime import TargetModel


class ShapFeatureValue(TypedDict):
    """Store one feature contribution"""

    feature: str
    raw_value: float
    shap_value: float


class LocalShapExplanation(TypedDict):
    """Store one local shap explanation"""

    base_value: float
    features: list[ShapFeatureValue]


def explain_target_prediction(
    target_model: TargetModel,
    feature_row: pd.DataFrame,
) -> tuple[float, LocalShapExplanation]:
    """Explain one target prediction with local shap values

    Args:
        target_model: fitted model for one question
        feature_row: one row containing the model features

    Returns:
        prediction and local shap values sorted by impact
    """

    if len(feature_row) != 1:
        raise ValueError("feature_row must contain exactly one row")

    feature_columns = target_model.preprocessor.feature_columns
    missing = [
        column for column in feature_columns if column not in feature_row.columns
    ]
    if missing:
        raise ValueError(f"feature_row is missing model features: {missing}")

    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    estimator = target_model.estimator
    if not isinstance(
        estimator,
        (DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, LGBMRegressor, XGBRegressor),
    ):

        raise TypeError(f"SHAP does not support model type: {type(estimator).__name__}")

    transformed = target_model.preprocessor.transform(feature_row)
    model_input = transformed[feature_columns]
    prediction_values = np.asarray(estimator.predict(model_input), dtype=float).reshape(
        -1
    )
    if prediction_values.shape != (1,):
        raise ValueError("target model must return one prediction")
    prediction = float(prediction_values[0])

    tree_explainer = cast(Any, shap.TreeExplainer)(estimator)
    shap_values = np.asarray(
        tree_explainer.shap_values(model_input, check_additivity=True),
        dtype=float,
    )
    expected_values = np.asarray(tree_explainer.expected_value, dtype=float).reshape(-1)
    expected_shape = (1, len(feature_columns))
    if shap_values.shape != expected_shape:
        raise ValueError(
            f"SHAP returned shape {shap_values.shape} instead of {expected_shape}",
        )
    if expected_values.shape != (1,):
        raise ValueError("SHAP must return one base value")

    base_value = float(expected_values[0])
    contributions = shap_values[0]
    explained_prediction = base_value + float(contributions.sum())
    if not np.isclose(explained_prediction, prediction, rtol=1e-5, atol=1e-6):
        raise ValueError("SHAP values do not add up to the model prediction")

    raw_values = feature_row[feature_columns].iloc[0].to_numpy(dtype=float)
    features: list[ShapFeatureValue] = [
        {
            "feature": feature,
            "raw_value": float(raw_value),
            "shap_value": float(shap_value),
        }
        for feature, raw_value, shap_value in zip(
            feature_columns,
            raw_values,
            contributions,
            strict=True,
        )
    ]
    features.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return prediction, {
        "base_value": base_value,
        "features": features,
    }
