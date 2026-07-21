"""Validate and normalize inference event tables"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trustme_xai.contracts import (
    LOCAL_TIMEZONE,
    ParsedActivityWatchEvents,
)
from trustme_xai.feature_pipeline.activity_categories import unified_categories

EVENT_COLUMNS = {
    "window": {
        "user_id",
        "timestamp",
        "event_end",
        "app",
        "category",
    },
    "web": {
        "user_id",
        "timestamp",
        "event_end",
        "category",
    },
    "input": {
        "user_id",
        "timestamp",
        "event_end",
        "duration_seconds",
        "presses",
        "clicks",
        "mouse_distance",
        "scroll_abs",
    },
}
INPUT_NUMBER_COLUMNS = (
    "duration_seconds",
    "presses",
    "clicks",
    "mouse_distance",
    "scroll_abs",
)


def normalize_timestamp(value: object) -> pd.Timestamp:
    """Convert one timestamp to Zurich local time

    Args:
        value: timestamp value to normalize

    Returns:
        timezone-naive timestamp in Europe/Zurich
    """

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp must be valid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(LOCAL_TIMEZONE).tz_localize(None)
    return timestamp


def _prepare_event_table(name: str, table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name} events must be a DataFrame")
    if table.empty:
        return table.copy()

    missing = sorted(EVENT_COLUMNS[name] - set(table.columns))
    if missing:
        raise ValueError(f"{name} events are missing columns: {missing}")

    normalized = table.copy()
    for column in ("timestamp", "event_end"):
        normalized[column] = normalized[column].map(normalize_timestamp)
    if bool((normalized["event_end"] < normalized["timestamp"]).any()):
        raise ValueError(f"{name} events contain an end before their start")

    if bool(normalized["user_id"].isna().any()):
        raise ValueError(f"{name} events contain a missing user_id")
    normalized["user_id"] = normalized["user_id"].astype(str).str.strip()
    if bool(normalized["user_id"].eq("").any()):
        raise ValueError(f"{name} events contain a blank user_id")
    if name in {"window", "web"}:
        if bool(normalized["category"].isna().any()):
            raise ValueError(f"{name} events contain a missing category")
        normalized["category"] = normalized["category"].astype(str).str.strip()
        invalid_categories = sorted(
            set(normalized["category"]) - set(unified_categories()),
        )
        if invalid_categories:
            raise ValueError(
                f"{name} events contain unknown categories: {invalid_categories}",
            )
        normalized["source_type"] = name
        if "category_confidence" not in normalized.columns:
            normalized["category_confidence"] = "unknown"
    if name == "input":
        for column in INPUT_NUMBER_COLUMNS:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="raise",
            ).fillna(0.0)
        if not bool(
            np.isfinite(
                normalized[list(INPUT_NUMBER_COLUMNS)].to_numpy(dtype=float),
            ).all(),
        ):
            raise ValueError("input event values must be finite")
        if bool((normalized[list(INPUT_NUMBER_COLUMNS)] < 0).any().any()):
            raise ValueError("input event values must be non-negative")
    return normalized


def prepare_inference_events(
    events: ParsedActivityWatchEvents,
) -> ParsedActivityWatchEvents:
    """Validate and normalize all event tables used by inference

    Args:
        events: activity event tables

    Returns:
        event tables using Zurich local timestamps
    """

    return ParsedActivityWatchEvents(
        window=_prepare_event_table("window", events.window),
        web=_prepare_event_table("web", events.web),
        input=_prepare_event_table("input", events.input),
    )


def clip_events_at(
    events: ParsedActivityWatchEvents,
    end: object,
) -> ParsedActivityWatchEvents:
    """Drop or clip events after one inference time

    Args:
        events: validated inference event tables
        end: latest timestamp visible to inference

    Returns:
        event tables with no data after end
    """

    end_time = normalize_timestamp(end)

    def clip(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty:
            return table.copy()
        clipped = table.loc[table["timestamp"] < end_time].copy()
        clipped["event_end"] = clipped["event_end"].clip(upper=end_time)
        clipped = clipped.loc[clipped["event_end"] > clipped["timestamp"]]
        if "duration_seconds" in clipped.columns:
            clipped["duration_seconds"] = (
                clipped["event_end"] - clipped["timestamp"]
            ).dt.total_seconds()
        return clipped.reset_index(drop=True)

    return ParsedActivityWatchEvents(
        window=clip(events.window),
        web=clip(events.web),
        input=clip(events.input),
    )


def normalize_request_times(requests: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction request timestamps

    Args:
        requests: rows containing a timestamp column

    Returns:
        request rows using Zurich local timestamps
    """

    if "timestamp" not in requests.columns:
        raise ValueError("requests must contain a timestamp column")
    normalized = requests.copy()
    normalized["timestamp"] = normalized["timestamp"].map(normalize_timestamp)
    return normalized
