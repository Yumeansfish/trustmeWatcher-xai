"""Define regression models used by production training"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

MODEL_NAMES = (
    "ridge",
    "lasso",
    "gradient_boosting",
    "random_forest",
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
    factories: dict[str, Callable[[], Any]] = {
        "ridge": lambda: Ridge(alpha=1.0),
        "lasso": lambda: Lasso(alpha=0.01, max_iter=20_000),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            max_depth=3,
            n_estimators=50,
            random_state=random_state,
        ),
        "random_forest": lambda: RandomForestRegressor(
            max_depth=5,
            n_estimators=50,
            n_jobs=1,
            random_state=random_state,
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
