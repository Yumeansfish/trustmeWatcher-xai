"""Enforce the zero-sum counterfactual constraints"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

IMMUTABLE_FEATURES: frozenset[str] = frozenset(
    {
        "timestamp",
        "user_id",
        "hour_of_day",
        "is_morning",
        "is_afternoon",
        "sleep_hours",
        "minutes_since_prev1_window",
    },
)

ACTIONABLE_CATEGORIES: tuple[str, ...] = (
    "time_personal_distraction",
    "time_media",
    "time_communication",
    "time_other",
    "time_development",
    "time_writing",
    "time_research",
)

DERIVED_FOCUS_METRICS: tuple[str, ...] = (
    "mean_focus_block_minutes",
    "short_fragment_count_2m",
    "app_switch_count",
    "input_load",
)

WINDOW_TIME_BUDGET: float = 60.0
TOLERANCE: float = 1e-4


def is_immutable(feature_name: str) -> bool:
    """Check if feature is immutable

    Args:
        feature_name: name of feature column

    Returns:
        True if feature is immutable
    """
    return feature_name in IMMUTABLE_FEATURES


def check_time_budget(
    feature_row: Mapping[str, Any],
    proposed_deltas: Mapping[str, float],
) -> bool:
    """Assert proposed time shifts satisfy non-negativity and 60-minute window limits

    Args:
        feature_row: original feature row mapping
        proposed_deltas: dictionary of proposed signed minute shifts

    Returns:
        True if all physical constraints are satisfied
    """
    if abs(sum(proposed_deltas.values())) > TOLERANCE:
        return False

    new_active_sum = 0.0
    for cat in ACTIONABLE_CATEGORIES:
        new_val = float(feature_row.get(cat, 0.0)) + float(
            proposed_deltas.get(cat, 0.0),
        )
        if new_val < -TOLERANCE:
            return False
        new_active_sum += new_val

    afk = float(feature_row.get("minutes_afk", feature_row.get("afk_minutes", 0.0)))
    return new_active_sum <= (WINDOW_TIME_BUDGET - afk + TOLERANCE)


def validate_counterfactual(
    original_row: Mapping[str, Any],
    modified_row: Mapping[str, Any],
) -> None:
    """Validate physical counterfactual constraints

    Args:
        original_row: baseline feature mapping
        modified_row: proposed counterfactual feature mapping

    Raises:
        ValueError: if any physical invariant is violated
    """
    for f in IMMUTABLE_FEATURES:
        changed = (
            f in original_row
            and f in modified_row
            and not _same_value(original_row[f], modified_row[f])
        )
        if changed:
            raise ValueError(f"immutable feature modified: {f}")

    deltas = {
        cat: float(modified_row.get(cat, 0.0)) - float(original_row.get(cat, 0.0))
        for cat in ACTIONABLE_CATEGORIES
    }
    if abs(sum(deltas.values())) > TOLERANCE:
        total = sum(deltas.values())
        raise ValueError(
            f"zero-sum conservation invariant violated: sum(delta) = {total}",
        )

    for cat in ACTIONABLE_CATEGORIES:
        if float(modified_row.get(cat, 0.0)) < -TOLERANCE:
            raise ValueError(f"non-negativity invariant violated for category {cat}")

    active = sum(float(modified_row.get(cat, 0.0)) for cat in ACTIONABLE_CATEGORIES)
    if active > WINDOW_TIME_BUDGET + TOLERANCE:
        raise ValueError(f"time budget upper bound invariant violated: {active} > 60.0")


def _same_value(left: Any, right: Any) -> bool:
    """Treat two missing scalar values as unchanged."""
    try:
        if bool(pd.isna(left)) and bool(pd.isna(right)):
            return True
        return bool(left == right)
    except (TypeError, ValueError):
        return False
