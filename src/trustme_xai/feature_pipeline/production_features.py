"""Build the feature row used by the dashboard model"""

from __future__ import annotations

import pandas as pd

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.behavior_state_features import (
    add_behavior_state_features,
    build_hourly_behavior_table,
    load_production_behavior_state_model,
)
from trustme_xai.feature_pipeline.event_contract import (
    clip_events_at,
    normalize_request_times,
    prepare_inference_events,
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


def build_production_features(
    events: ParsedActivityWatchEvents,
    timestamps: pd.DataFrame,
    *,
    include_actionable_categories: bool = False,
) -> pd.DataFrame:
    """Build the 25 model features

    Args:
        events: parsed activitywatch tables
        timestamps: rows containing user_id and timestamp

    Returns:
        pd.DataFrame with production model features
    """

    normalized_events = prepare_inference_events(events)
    normalized_timestamps = normalize_request_times(timestamps)
    if normalized_timestamps.empty:
        raise ValueError("timestamps must contain at least one row")
    normalized_events = clip_events_at(
        normalized_events,
        normalized_timestamps["timestamp"].max(),
    )
    table = build_window_features(
        normalized_events,
        normalized_timestamps,
        PRODUCTION_WINDOW_MINUTES,
    )
    table = build_context_features(
        normalized_events,
        table,
        current_window_minutes=PRODUCTION_WINDOW_MINUTES,
    )
    table = add_rolling_history_features(table)
    table = add_behavior_state_features(
        table,
        build_hourly_behavior_table(normalized_events),
        load_production_behavior_state_model(),
        current_window_minutes=PRODUCTION_WINDOW_MINUTES,
        events=normalized_events,
    )

    missing = [
        column for column in PRODUCTION_FEATURE_COLUMNS if column not in table.columns
    ]
    if missing:
        raise ValueError(f"feature table is missing required columns: {missing}")
    output_columns = ["user_id", "timestamp", *PRODUCTION_FEATURE_COLUMNS]
    if include_actionable_categories:
        output_columns.extend(
            column
            for column in ACTIONABLE_CATEGORY_COLUMNS
            if column not in output_columns
        )
    return table[output_columns].copy()
