"""Build current-window features used by inference"""

from __future__ import annotations

from typing import NamedTuple, cast

import pandas as pd

from trustme_xai.contracts import ParsedActivityWatchEvents


class Allocation(NamedTuple):
    """Store active minutes by category"""

    total_minutes: float
    category_minutes: dict[str, float]


class EventSlice(NamedTuple):
    """Store one selected activity slice"""

    start: pd.Timestamp
    end: pd.Timestamp
    timestamp: pd.Timestamp
    category: str
    priority: int


def divide_or_zero(numerator: float, denominator: float) -> float:
    """Divide two values or return zero

    Args:
        numerator: value to divide
        denominator: value to divide by

    Returns:
        division result or zero
    """

    return 0.0 if denominator <= 0 else numerator / denominator


def count_changes(values: list[str]) -> int:
    """Count changes between adjacent values

    Args:
        values: ordered values

    Returns:
        number of adjacent changes
    """

    return sum(
        1 for previous, current in zip(values, values[1:]) if previous != current
    )


def select_window_events(
    events: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Select events that overlap one time window

    Args:
        events: event table
        window_start: window start time
        window_end: window end time

    Returns:
        pd.DataFrame with overlapping events
    """

    if events.empty:
        return events
    mask = (events["timestamp"] < window_end) & (events["event_end"] > window_start)
    return events[mask]


def column_sum(events: pd.DataFrame, column: str) -> float:
    """Sum one event column

    Args:
        events: event table
        column: column to sum

    Returns:
        column total
    """

    return float(events[column].sum())


def choose_event_slices(
    events: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[EventSlice]:
    """Choose one event for each overlapping time slice

    Args:
        events: window and web events
        window_start: window start time
        window_end: window end time

    Returns:
        selected activity slices
    """

    slices: list[EventSlice] = []
    for row in events.itertuples(index=False):
        start = cast(pd.Timestamp, getattr(row, "timestamp"))
        end = cast(pd.Timestamp, getattr(row, "event_end"))
        clip_start = max(start, window_start)
        clip_end = min(end, window_end)
        if clip_end <= clip_start:
            continue

        source = str(getattr(row, "source_type", "window"))
        if source == "afk":
            priority = 2
        elif source == "web":
            priority = 1
        else:
            priority = 0
        slices.append(
            EventSlice(
                start=clip_start,
                end=clip_end,
                timestamp=start,
                category=str(getattr(row, "category")),
                priority=priority,
            )
        )

    if not slices:
        return []

    boundaries = sorted({time for item in slices for time in (item.start, item.end)})
    ordered = sorted(slices, key=lambda item: item.start)
    chosen: list[EventSlice] = []
    active: list[EventSlice] = []
    next_record = 0

    for start, end in zip(boundaries, boundaries[1:]):
        while next_record < len(ordered) and ordered[next_record].start < end:
            active.append(ordered[next_record])
            next_record += 1
        active = [item for item in active if item.end > start]
        if active:
            source = max(active, key=lambda item: (item.priority, item.timestamp))
            chosen.append(source._replace(start=start, end=end))
    return chosen


def allocate_activity_minutes(
    slices: list[EventSlice],
    window_minutes: float,
) -> Allocation:
    """Group activity minutes by category

    Args:
        slices: selected activity slices
        window_minutes: maximum window length

    Returns:
        active minutes and category totals
    """

    if not slices:
        return Allocation(0.0, {})

    total = 0.0
    by_category: dict[str, float] = {}
    for item in slices:
        minutes = (item.end - item.start).total_seconds() / 60.0
        total += minutes
        by_category[item.category] = by_category.get(item.category, 0.0) + minutes
    return Allocation(min(total, window_minutes), by_category)


def category_segments(
    slices: list[EventSlice],
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Join adjacent slices with the same category

    Args:
        slices: selected activity slices

    Returns:
        joined category segments
    """

    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for item in slices:
        if (
            segments
            and segments[-1][2] == item.category
            and segments[-1][1] == item.start
        ):
            segments[-1] = (segments[-1][0], item.end, item.category)
        else:
            segments.append((item.start, item.end, item.category))
    return segments


def segment_features(
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]],
) -> dict[str, float]:
    """Calculate the focus features used by the model

    Args:
        segments: joined category segments

    Returns:
        focus feature values
    """

    if not segments:
        return {
            "mean_focus_block_minutes": 0.0,
            "short_fragment_count_2m": 0.0,
        }

    durations = [(end - start).total_seconds() / 60.0 for start, end, _ in segments]
    return {
        "mean_focus_block_minutes": float(sum(durations) / len(durations)),
        "short_fragment_count_2m": float(sum(duration < 2.0 for duration in durations)),
    }


def app_switch_count(window_events: pd.DataFrame) -> int:
    """Count app changes in one window

    Args:
        window_events: app window events

    Returns:
        number of app changes
    """

    if window_events.empty:
        return 0
    apps = window_events.sort_values("timestamp")["app"].astype(str).to_list()
    return count_changes(apps)


def input_features(
    input_events: pd.DataFrame,
    window_minutes: int,
) -> dict[str, float]:
    """Calculate the input features used by the model

    Args:
        input_events: input events in the window
        window_minutes: full window length

    Returns:
        input feature values
    """

    if input_events.empty:
        return {
            "input_active_ratio": 0.0,
            "input_load": 0.0,
        }

    keypress_count = column_sum(input_events, "presses")
    click_count = column_sum(input_events, "clicks")
    mouse_distance = column_sum(input_events, "mouse_distance")
    scroll_abs = column_sum(input_events, "scroll_abs")
    activity = input_events[["presses", "clicks", "mouse_distance", "scroll_abs"]].sum(
        axis=1
    )
    input_active_minutes = min(
        float(input_events.loc[activity > 0, "duration_seconds"].sum()) / 60.0,
        float(window_minutes),
    )
    input_load = (
        keypress_count + click_count + (scroll_abs / 100.0) + (mouse_distance / 1000.0)
    )
    return {
        "input_active_ratio": divide_or_zero(
            input_active_minutes,
            float(window_minutes),
        ),
        "input_load": input_load,
    }


def build_window_features(
    events: ParsedActivityWatchEvents,
    timestamps: pd.DataFrame,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """Build current features for each timestamp

    Args:
        events: parsed activitywatch tables
        timestamps: rows containing user_id and timestamp
        window_minutes: minutes before each timestamp

    Returns:
        pd.DataFrame with current model features
    """

    required = {"user_id", "timestamp"}
    if not required.issubset(timestamps.columns):
        raise ValueError("timestamps must contain user_id and timestamp columns")

    requests = timestamps.copy()
    requests["timestamp"] = pd.to_datetime(requests["timestamp"])
    by_user = {
        "window": (
            dict(tuple(events.window.groupby("user_id")))
            if not events.window.empty
            else {}
        ),
        "web": (
            dict(tuple(events.web.groupby("user_id"))) if not events.web.empty else {}
        ),
        "input": (
            dict(tuple(events.input.groupby("user_id")))
            if not events.input.empty
            else {}
        ),
    }
    empty = {
        "window": pd.DataFrame(columns=events.window.columns),
        "web": pd.DataFrame(columns=events.web.columns),
        "input": pd.DataFrame(columns=events.input.columns),
    }
    rows: list[dict[str, object]] = []

    for request in requests.itertuples(index=False):
        user_id = str(getattr(request, "user_id"))
        window_end = pd.Timestamp(getattr(request, "timestamp"))
        if pd.isna(window_end):
            raise ValueError("timestamps contains an invalid timestamp")
        window_start = window_end - pd.Timedelta(minutes=window_minutes)
        window_events = select_window_events(
            by_user["window"].get(user_id, empty["window"]),
            window_start,
            window_end,
        )
        web_events = select_window_events(
            by_user["web"].get(user_id, empty["web"]),
            window_start,
            window_end,
        )
        input_events = select_window_events(
            by_user["input"].get(user_id, empty["input"]),
            window_start,
            window_end,
        )
        selected_slices = choose_event_slices(
            pd.concat([window_events, web_events], ignore_index=True),
            window_start,
            window_end,
        )
        allocation = allocate_activity_minutes(
            selected_slices,
            float(window_minutes),
        )
        switches = float(app_switch_count(window_events))
        row: dict[str, object] = {
            "user_id": user_id,
            "timestamp": window_end,
            "window_minutes": window_minutes,
            "app_switch_count": switches,
            "total_active_minutes": allocation.total_minutes,
            "time_personal_distraction": allocation.category_minutes.get(
                "personal_distraction",
                0.0,
            ),
            "ratio_communication": divide_or_zero(
                allocation.category_minutes.get("communication", 0.0),
                allocation.total_minutes,
            ),
            "switch_rate": divide_or_zero(
                switches,
                allocation.total_minutes,
            ),
        }
        row.update(segment_features(category_segments(selected_slices)))
        row.update(input_features(input_events, window_minutes))
        rows.append(row)

    return pd.DataFrame(rows)
