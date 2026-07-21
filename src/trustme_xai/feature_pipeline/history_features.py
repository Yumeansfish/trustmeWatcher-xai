"""Build the earlier activity ratios used by inference"""

from __future__ import annotations

import pandas as pd

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.window_features import (
    allocate_activity_minutes,
    choose_event_slices,
    divide_or_zero,
    select_window_events,
)

ROLLING_SOURCE_COLUMNS = [
    "mean_focus_block_minutes",
    "switch_rate",
    "input_load",
]

CONTEXT_CATEGORIES = {
    "24h": ("personal_distraction",),
    "7d": (
        "media",
        "ai_assistant",
        "personal_distraction",
        "development",
        "research",
    ),
}


def _context_ratios(
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    name: str,
    days: int,
) -> dict[str, float]:
    allocation = allocate_activity_minutes(
        choose_event_slices(events, start, end),
        float(days * 24 * 60),
    )
    return {
        f"context_{name}_ratio_{category}": divide_or_zero(
            allocation.category_minutes.get(category, 0.0),
            allocation.total_minutes,
        )
        for category in CONTEXT_CATEGORIES[name]
    }


def build_context_features(
    events: ParsedActivityWatchEvents,
    feature_table: pd.DataFrame,
    current_window_minutes: int = 60,
) -> pd.DataFrame:
    """Add the model context ratios

    Args:
        events: parsed activitywatch tables
        feature_table: current feature rows
        current_window_minutes: size of the current window

    Returns:
        pd.DataFrame with earlier activity ratios
    """

    required = {"user_id", "timestamp"}
    if not required.issubset(feature_table.columns):
        raise ValueError("feature_table must contain user_id and timestamp")

    table = feature_table.copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    window_by_user = (
        dict(tuple(events.window.groupby("user_id"))) if not events.window.empty else {}
    )
    web_by_user = (
        dict(tuple(events.web.groupby("user_id"))) if not events.web.empty else {}
    )
    empty_window = pd.DataFrame(columns=events.window.columns)
    empty_web = pd.DataFrame(columns=events.web.columns)
    rows: list[dict[str, float]] = []

    for item in table.itertuples(index=False):
        user_id = str(getattr(item, "user_id"))
        timestamp = pd.Timestamp(getattr(item, "timestamp"))
        if pd.isna(timestamp):
            raise ValueError("feature_table contains an invalid timestamp")
        current_start = timestamp - pd.Timedelta(minutes=current_window_minutes)
        user_window = window_by_user.get(user_id, empty_window)
        user_web = web_by_user.get(user_id, empty_web)
        row: dict[str, float] = {}

        for name, days in (("24h", 1), ("7d", 7)):
            context_start = current_start - pd.Timedelta(days=days)
            selected = pd.concat(
                [
                    select_window_events(
                        user_window,
                        context_start,
                        current_start,
                    ),
                    select_window_events(
                        user_web,
                        context_start,
                        current_start,
                    ),
                ],
                ignore_index=True,
            )
            row.update(
                _context_ratios(
                    selected,
                    context_start,
                    current_start,
                    name,
                    days,
                )
            )
        rows.append(row)

    return pd.concat([table, pd.DataFrame(rows, index=table.index)], axis=1)


def add_rolling_history_features(
    feature_table: pd.DataFrame,
    source_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add features from earlier answers by the same user

    Args:
        feature_table: feature rows sorted during processing
        source_columns: columns used to calculate earlier averages

    Returns:
        pd.DataFrame with earlier answer features
    """

    required = {"user_id", "timestamp"}
    if not required.issubset(feature_table.columns):
        raise ValueError("feature_table must contain user_id and timestamp")

    table = feature_table.copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    table = table.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    grouped = table.groupby("user_id", sort=False)

    previous_timestamp = grouped["timestamp"].shift(1)
    gap_minutes = (table["timestamp"] - previous_timestamp).dt.total_seconds() / 60.0
    additions = [
        pd.DataFrame({"minutes_since_prev1_window": gap_minutes.fillna(0.0)}),
    ]

    columns = [
        col
        for col in (source_columns or ROLLING_SOURCE_COLUMNS)
        if col in table.columns
    ]
    if columns:
        values = (
            table[columns]
            .astype(float)
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
        )
        prior = grouped[columns].shift(1)
        rolling_mean = (
            prior.groupby(table["user_id"], sort=False)
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
        )
        rolling_mean.columns = [f"rolling_mean_3_{col}" for col in columns]
        current_minus = values - rolling_mean.to_numpy()
        current_minus.columns = [
            f"current_minus_rolling_mean_3_{col}" for col in columns
        ]
        additions.extend([rolling_mean, current_minus])

    return pd.concat([table, *additions], axis=1)
