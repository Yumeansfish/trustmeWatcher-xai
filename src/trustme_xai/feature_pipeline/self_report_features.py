"""Build causal features from past self-reports"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)

TIME_FEATURE_COLUMNS = [
    "history__hour_sin",
    "history__hour_cos",
    "history__dow_sin",
    "history__dow_cos",
]

TARGET_HISTORY_STATS = [
    "last",
    "mean",
    "mean_3",
    "std",
    "log_count",
    "log_gap_hours",
    "last_minus_mean",
]

DIRECT_ACTIVITY_FEATURE_COLUMNS = [
    column
    for column in PRODUCTION_FEATURE_COLUMNS
    if "_state_" not in column
]


def history_column(target: str, stat: str) -> str:
    """Build one history column name

    Args:
        target: model target name
        stat: history value name

    Returns:
        history column name
    """
    return f"history__{target}__{stat}"


def all_history_columns() -> list[str]:
    """List every generated history column

    Returns:
        ordered history column names
    """
    return [
        *[
            history_column(target, stat)
            for target in MODEL_TARGETS
            for stat in TARGET_HISTORY_STATS
        ],
        *TIME_FEATURE_COLUMNS,
    ]


def core_history_columns(target: str) -> list[str]:
    """List history columns for one target

    Args:
        target: model target name

    Returns:
        target history and time columns
    """
    return [
        *[
            history_column(target, stat)
            for stat in TARGET_HISTORY_STATS
        ],
        *TIME_FEATURE_COLUMNS,
    ]


def vector_history_columns(target: str) -> list[str]:
    """List compact history columns across all targets

    Args:
        target: model target name

    Returns:
        target history and cross-target columns
    """
    cross_target = [
        history_column(other, stat)
        for other in MODEL_TARGETS
        for stat in ("last", "mean", "last_minus_mean")
    ]
    return list(dict.fromkeys([*core_history_columns(target), *cross_target]))


def target_feature_sets(target: str) -> dict[str, list[str]]:
    """Build the selectable feature sets for one target

    Args:
        target: model target name

    Returns:
        feature set names mapped to ordered columns
    """
    core = core_history_columns(target)
    vector = vector_history_columns(target)
    return {
        "history_core_aw25": [*core, *PRODUCTION_FEATURE_COLUMNS],
        "history_vector_aw25": [*vector, *PRODUCTION_FEATURE_COLUMNS],
        "history_core_direct16": [*core, *DIRECT_ACTIVITY_FEATURE_COLUMNS],
    }


def add_self_report_features(
    table: pd.DataFrame,
    source_column: str = "answer_source",
    history_source: str = "streamdeck",
) -> pd.DataFrame:
    """Add features using only earlier self-reports

    Args:
        table: rows with model targets and timestamps
        source_column: column identifying the answer source
        history_source: source allowed to enter later rows

    Returns:
        pd.DataFrame with causal history features
    """
    required = {"user_id", "timestamp", source_column, *MODEL_TARGETS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"self-report table is missing columns: {missing}")
    if table.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("self-report rows need unique user timestamps")

    ordered = table.sort_values(
        ["user_id", "timestamp"],
        kind="stable",
    ).copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="raise")
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for user_id, user_rows in ordered.groupby("user_id", sort=False):
        values = {target: [] for target in MODEL_TARGETS}
        times = {target: [] for target in MODEL_TARGETS}

        for item in user_rows.itertuples(index=False):
            timestamp = pd.Timestamp(item.timestamp)
            row: dict[str, float | str | pd.Timestamp] = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "history__hour_sin": math.sin(
                    2 * math.pi * timestamp.hour / 24,
                ),
                "history__hour_cos": math.cos(
                    2 * math.pi * timestamp.hour / 24,
                ),
                "history__dow_sin": math.sin(
                    2 * math.pi * timestamp.dayofweek / 7,
                ),
                "history__dow_cos": math.cos(
                    2 * math.pi * timestamp.dayofweek / 7,
                ),
            }

            for target in MODEL_TARGETS:
                past = values[target]
                count = len(past)
                mean = float(np.mean(past)) if count else float("nan")
                last = past[-1] if count else float("nan")
                mean_3 = float(np.mean(past[-3:])) if count else float("nan")
                std = float(np.std(past, ddof=0)) if count > 1 else 0.0
                gap = (
                    (timestamp - times[target][-1]).total_seconds() / 3600.0
                    if count
                    else float("nan")
                )
                row[history_column(target, "last")] = last
                row[history_column(target, "mean")] = mean
                row[history_column(target, "mean_3")] = mean_3
                row[history_column(target, "std")] = std
                row[history_column(target, "log_count")] = math.log1p(count)
                row[history_column(target, "log_gap_hours")] = (
                    math.log1p(max(gap, 0.0))
                    if np.isfinite(gap)
                    else float("nan")
                )
                row[history_column(target, "last_minus_mean")] = last - mean

            rows.append(row)
            if getattr(item, source_column) != history_source:
                continue
            for target in MODEL_TARGETS:
                value = getattr(item, target)
                if pd.notna(value):
                    values[target].append(float(value))
                    times[target].append(timestamp)

    history = pd.DataFrame(rows)
    return table.merge(
        history,
        on=["user_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
