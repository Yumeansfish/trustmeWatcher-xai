"""Build canonical ActivityWatch feature rows for runtime inference."""

from __future__ import annotations

import pandas as pd

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.action_allocations import (
    ACTION_ALLOCATION_COLUMNS,
    add_action_allocations,
)
from trustme_xai.feature_pipeline.behavior_state_features import (
    add_behavior_state_features,
    build_hourly_behavior_table,
    load_production_behavior_state_model,
)
from trustme_xai.feature_pipeline.history_features import (
    add_rolling_history_features,
    build_context_features,
)
from trustme_xai.feature_pipeline.window_features import build_window_features

PRODUCTION_WINDOW_MINUTES = 60
PRODUCTION_FEATURE_SET = "production_25"

ACTIONABLE_CATEGORY_COLUMNS = [
    "time_personal_distraction",
    "time_media",
    "time_communication",
    "time_other",
    "time_development",
    "time_writing",
    "time_research",
]

PRODUCTION_FEATURE_COLUMNS = [
    "current60_state_share_0",
    "current60_state_share_2",
    "current60_state_share_4",
    "prior24h_state_share_4",
    "short_fragment_count_2m",
    "current60_state_active_hours",
    "prior7d_state_share_5",
    "minutes_since_prev1_window",
    "context_24h_ratio_personal_distraction",
    "input_active_ratio",
    "mean_focus_block_minutes",
    "prior24h_state_share_5",
    "context_7d_ratio_media",
    "current_minus_rolling_mean_3_mean_focus_block_minutes",
    "current60_state_distance_median",
    "context_7d_ratio_ai_assistant",
    "context_7d_ratio_personal_distraction",
    "time_personal_distraction",
    "context_7d_ratio_development",
    "input_load",
    "prior24h_state_share_0",
    "rolling_mean_3_input_load",
    "ratio_communication",
    "current_minus_rolling_mean_3_switch_rate",
    "context_7d_ratio_research",
]

# This feature comes from questionnaire request times rather than AW events.
EVENT_DERIVED_ACTIVITY_FEATURE_COLUMNS = [
    column
    for column in PRODUCTION_FEATURE_COLUMNS
    if column != "minutes_since_prev1_window"
]

RUNTIME_ACTIVITY_FEATURE_COLUMNS = [
    *PRODUCTION_FEATURE_COLUMNS,
    *[
        column
        for column in ACTIONABLE_CATEGORY_COLUMNS
        if column not in PRODUCTION_FEATURE_COLUMNS
    ],
    *ACTION_ALLOCATION_COLUMNS,
]


def build_production_features(
    events: ParsedActivityWatchEvents,
    timestamps: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical activity features for prepared runtime inputs.

    Args:
        events: validated ActivityWatch tables clipped at the latest request
        timestamps: normalized rows containing user_id and timestamp

    Returns:
        rows containing score, actionable, and action-allocation features
    """
    if timestamps.empty:
        raise ValueError("timestamps must contain at least one row")
    table = build_window_features(
        events,
        timestamps,
        PRODUCTION_WINDOW_MINUTES,
    )
    table = build_context_features(
        events,
        table,
        current_window_minutes=PRODUCTION_WINDOW_MINUTES,
    )
    table = add_rolling_history_features(table)
    table = add_behavior_state_features(
        table,
        build_hourly_behavior_table(events),
        load_production_behavior_state_model(),
        current_window_minutes=PRODUCTION_WINDOW_MINUTES,
        events=events,
    )
    table = add_action_allocations(table)

    missing = [
        column
        for column in RUNTIME_ACTIVITY_FEATURE_COLUMNS
        if column not in table.columns
    ]
    if missing:
        raise ValueError(f"feature table is missing required columns: {missing}")
    output_columns = ["user_id", "timestamp", *RUNTIME_ACTIVITY_FEATURE_COLUMNS]
    return table[output_columns].copy()
