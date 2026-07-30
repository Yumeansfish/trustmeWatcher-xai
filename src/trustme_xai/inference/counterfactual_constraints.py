"""Physical simplex constraints and feature taxonomy for Zero-Sum counterfactual recourse"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
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
    "time_development",
    "time_writing",
    "time_communication",
    "time_personal_distraction",
    "time_media",
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
        feature_name: name of the feature column

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
    # 1. Zero-Sum Conservation Invariant: sum(deltas) == 0
    total_delta = sum(proposed_deltas.values())
    if abs(total_delta) > TOLERANCE:
        return False

    # 2. Non-negativity & window capacity invariants
    new_active_sum = 0.0
    for cat in ACTIONABLE_CATEGORIES:
        baseline = float(feature_row.get(cat, 0.0))
        delta = float(proposed_deltas.get(cat, 0.0))
        new_val = baseline + delta
        if new_val < -TOLERANCE:
            return False
        new_active_sum += new_val

    afk_minutes = float(feature_row.get("afk_minutes", 0.0))
    if new_active_sum > (WINDOW_TIME_BUDGET - afk_minutes + TOLERANCE):
        return False

    return True


def validate_counterfactual(
    original_row: Mapping[str, Any],
    modified_row: Mapping[str, Any],
) -> None:
    """Fail-fast guard clause validating all physical invariants between original and counterfactual

    Args:
        original_row: baseline feature mapping
        modified_row: proposed counterfactual feature mapping

    Raises:
        ValueError: if any physical invariant is violated
    """
    # Immutable feature invariant: deltas for immutable features must be 0
    for feature in IMMUTABLE_FEATURES:
        if feature in original_row and feature in modified_row:
            if original_row[feature] != modified_row[feature]:
                raise ValueError(f"immutable feature modified: {feature}")

    # Zero-sum conservation invariant
    deltas = {
        cat: float(modified_row.get(cat, 0.0)) - float(original_row.get(cat, 0.0))
        for cat in ACTIONABLE_CATEGORIES
    }
    total_delta = sum(deltas.values())
    if abs(total_delta) > TOLERANCE:
        raise ValueError(f"zero-sum conservation invariant violated: sum(delta) = {total_delta}")

    # Non-negativity invariant
    for cat in ACTIONABLE_CATEGORIES:
        val = float(modified_row.get(cat, 0.0))
        if val < -TOLERANCE:
            raise ValueError(f"non-negativity invariant violated for category {cat}: {val}")

    # Time budget upper bound invariant
    active_total = sum(float(modified_row.get(cat, 0.0)) for cat in ACTIONABLE_CATEGORIES)
    if active_total > WINDOW_TIME_BUDGET + TOLERANCE:
        raise ValueError(f"time budget upper bound invariant violated: {active_total} > 60.0")
