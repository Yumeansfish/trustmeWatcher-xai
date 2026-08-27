"""Build the action allocations used by the binary classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

ACTION_ALLOCATION_SOURCES = {
    "action_minutes_development": ("time_development",),
    "action_minutes_writing": ("time_writing",),
    "action_minutes_research": ("time_research",),
    "action_minutes_communication": (
        "time_communication",
        "time_meetings",
    ),
    "action_minutes_media": (
        "time_media",
        "time_game_distraction",
    ),
    "action_minutes_personal_distraction": (
        "time_personal_distraction",
    ),
    "action_minutes_other": (
        "time_ai_assistant",
        "time_browser_uncategorized",
        "time_calendar_tasks",
        "time_file_management",
        "time_other",
        "time_system_admin",
    ),
}

ACTION_ALLOCATION_COLUMNS = tuple(ACTION_ALLOCATION_SOURCES)
ACTION_ALLOCATION_NAMES = {
    column: column.removeprefix("action_minutes_")
    for column in ACTION_ALLOCATION_COLUMNS
}
ACTION_WINDOW_MINUTES = 60.0
NON_NEGATIVE_TOLERANCE = 1e-9
WINDOW_TOLERANCE = 1e-4
CHANGE_EPSILON = 1e-12


@dataclass(frozen=True)
class ActionChange:
    """Describe one category's minutes before and after a suggestion."""

    category: str
    baseline_minutes: float
    counterfactual_minutes: float
    delta_minutes: float


@dataclass(frozen=True)
class ActionAllocation:
    """Represent one valid allocation across the seven action categories."""

    _values: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self._values, dtype=float)
        if values.shape != (len(ACTION_ALLOCATION_COLUMNS),):
            raise ValueError("action allocation must contain all seven categories")
        _validate_allocation_values(values.reshape(1, -1))
        object.__setattr__(self, "_values", tuple(float(value) for value in values))

    @classmethod
    def from_feature_row(
        cls,
        row: Mapping[str, object] | pd.Series,
    ) -> ActionAllocation:
        """Read and validate an Action allocation from one feature row."""
        missing = sorted(
            column for column in ACTION_ALLOCATION_COLUMNS if column not in row
        )
        if missing:
            raise ValueError(f"action allocation columns are missing: {missing}")
        return cls(tuple(float(row[column]) for column in ACTION_ALLOCATION_COLUMNS))

    @property
    def minutes(self) -> Mapping[str, float]:
        """Return immutable minutes keyed by canonical category name."""
        return MappingProxyType(
            {
                ACTION_ALLOCATION_NAMES[column]: self._values[index]
                for index, column in enumerate(ACTION_ALLOCATION_COLUMNS)
            },
        )

    def swap_scenarios(
        self,
        feature_row: pd.DataFrame,
        step_minutes: float,
    ) -> pd.DataFrame:
        """Build every legal zero-sum swap in deterministic category order."""
        if len(feature_row) != 1:
            raise ValueError("feature_row must contain exactly one row")
        step = float(step_minutes)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_minutes must be finite and positive")

        template = feature_row.iloc[0].to_dict()
        scenarios: list[dict[str, object]] = []
        for donor, available in enumerate(self._values):
            if available + CHANGE_EPSILON < step:
                continue
            for receiver in range(len(ACTION_ALLOCATION_COLUMNS)):
                if donor == receiver:
                    continue
                candidate = list(self._values)
                candidate[donor] -= step
                candidate[receiver] += step
                validated = ActionAllocation(tuple(candidate))
                scenario = dict(template)
                scenario.update(
                    zip(
                        ACTION_ALLOCATION_COLUMNS,
                        validated._values,
                        strict=True,
                    ),
                )
                scenarios.append(scenario)

        if not scenarios:
            return feature_row.iloc[0:0].copy()
        return pd.DataFrame.from_records(scenarios, columns=feature_row.columns)

    def changes_to(self, other: ActionAllocation) -> tuple[ActionChange, ...]:
        """Describe category changes from this allocation to another."""
        changes: list[ActionChange] = []
        for index, column in enumerate(ACTION_ALLOCATION_COLUMNS):
            baseline = self._values[index]
            counterfactual = other._values[index]
            delta = counterfactual - baseline
            if abs(delta) <= CHANGE_EPSILON:
                continue
            changes.append(
                ActionChange(
                    category=ACTION_ALLOCATION_NAMES[column],
                    baseline_minutes=baseline,
                    counterfactual_minutes=counterfactual,
                    delta_minutes=delta,
                ),
            )
        return tuple(changes)


def _validate_allocation_values(allocation: np.ndarray) -> None:
    if not np.isfinite(allocation).all():
        raise ValueError("action allocations must be finite")
    if bool((allocation < -NON_NEGATIVE_TOLERANCE).any()):
        raise ValueError("action allocations must not be negative")
    if bool(
        (allocation.sum(axis=1) > ACTION_WINDOW_MINUTES + WINDOW_TOLERANCE).any(),
    ):
        raise ValueError("action allocations exceed the 60-minute window")


def add_action_allocations(table: pd.DataFrame) -> pd.DataFrame:
    """Add the seven classifier action allocations.

    Args:
        table: window feature rows with legacy activity category minutes

    Returns:
        rows with the classifier action allocation columns
    """
    required = {
        source
        for sources in ACTION_ALLOCATION_SOURCES.values()
        for source in sources
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"action allocation source columns are missing: {missing}")

    result = table.copy()
    for column, sources in ACTION_ALLOCATION_SOURCES.items():
        values = result[list(sources)].apply(pd.to_numeric, errors="raise")
        result[column] = values.sum(axis=1)

    allocation = result[list(ACTION_ALLOCATION_COLUMNS)].to_numpy(dtype=float)
    _validate_allocation_values(allocation)
    return result
