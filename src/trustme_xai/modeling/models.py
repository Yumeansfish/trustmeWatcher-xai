"""Define regression models used by production training"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = (
    "ridge_0.1",
    "ridge_1",
    "ridge_10",
    "extra_leaf2",
    "extra_leaf5",
    "extra_leaf10",
    "forest_leaf5",
    "forest_leaf10",
    "hist_leaf10",
    "hist_leaf20",
)


@dataclass(frozen=True)
class ModelSpec:
    """Store one model name and factory"""

    name: str
    factory: Callable[[], Any]


def model_specs(
    names: Sequence[str] = MODEL_NAMES,
    random_state: int = 42,
) -> tuple[ModelSpec, ...]:
    """Build selected model specifications

    Args:
        names: model names to include
        random_state: seed used by learned models

    Returns:
        selected ModelSpec values
    """
    def ridge(alpha: float) -> Any:
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            Ridge(alpha=alpha),
        )

    def imputed(estimator: Any) -> Any:
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            estimator,
        )

    factories: dict[str, Callable[[], Any]] = {
        "ridge_0.1": lambda: ridge(0.1),
        "ridge_1": lambda: ridge(1.0),
        "ridge_10": lambda: ridge(10.0),
        "extra_leaf2": lambda: imputed(
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=0.7,
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "extra_leaf5": lambda: imputed(
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_features=0.7,
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "extra_leaf10": lambda: imputed(
            ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                max_features=0.7,
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "forest_leaf5": lambda: imputed(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                max_features=0.7,
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "forest_leaf10": lambda: imputed(
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                max_features=0.7,
                n_jobs=1,
                random_state=random_state,
            ),
        ),
        "hist_leaf10": lambda: imputed(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=random_state,
            ),
        ),
        "hist_leaf20": lambda: imputed(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=3.0,
                early_stopping=False,
                random_state=random_state,
            ),
        ),
    }
    if not names:
        raise ValueError("at least one model must be selected")
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    unknown = [name for name in names if name not in factories]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return tuple(ModelSpec(name, factories[name]) for name in names)


def default_model_specs(random_state: int = 42) -> tuple[ModelSpec, ...]:
    """Build every production model specification

    Args:
        random_state: seed used by learned models

    Returns:
        all production ModelSpec values
    """
    return model_specs(random_state=random_state)
