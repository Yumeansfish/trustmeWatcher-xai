"""Parse live ActivityWatch events for inference"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import NamedTuple

import pandas as pd

from trustme_xai.contracts import (
    INPUT_COLUMNS,
    WEB_COLUMNS,
    WINDOW_COLUMNS,
    ParsedActivityWatchEvents,
)
from trustme_xai.feature_pipeline.activity_categories import (
    categorize_web_event,
    categorize_window_event,
)
from trustme_xai.feature_pipeline.event_contract import normalize_timestamp

MAX_EVENT_SECONDS = 12 * 60 * 60
SUPPORTED_BUCKET_TYPES = {
    "currentwindow": "window",
    "web.tab.current": "web",
    "os.hid.input": "input",
}


class _Snapshot(NamedTuple):
    timestamp: object
    duration_seconds: float
    data: Mapping[str, object]


def _number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _duration_seconds(value: object) -> float | None:
    seconds = value.total_seconds() if isinstance(value, timedelta) else _number(value)
    if 0 < seconds < MAX_EVENT_SECONDS:
        return seconds
    return None


def _event_snapshots(events: Sequence[object]) -> list[_Snapshot]:
    longest: dict[tuple[str, str], _Snapshot] = {}
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            raise TypeError("ActivityWatch events must be mappings")
        duration_seconds = _duration_seconds(raw_event.get("duration"))
        timestamp = raw_event.get("timestamp")
        raw_data = raw_event.get("data", {})
        if duration_seconds is None or timestamp in (None, ""):
            continue
        if not isinstance(raw_data, Mapping):
            raise TypeError("ActivityWatch event data must be a mapping")

        snapshot = _Snapshot(timestamp, duration_seconds, raw_data)
        key = (
            str(timestamp),
            json.dumps(raw_data, sort_keys=True, default=str),
        )
        previous = longest.get(key)
        if previous is None or duration_seconds > previous.duration_seconds:
            longest[key] = snapshot
    return list(longest.values())


def _base_row(
    bucket_id: str,
    user_id: str,
    hostname: str,
    source_type: str,
    snapshot: _Snapshot,
) -> dict[str, object]:
    timestamp = normalize_timestamp(snapshot.timestamp)
    event_end = timestamp + pd.Timedelta(seconds=snapshot.duration_seconds)
    return {
        "bucket": bucket_id,
        "user_id": user_id,
        "hostname": hostname,
        "timestamp": timestamp,
        "event_end": event_end,
        "date": str(timestamp.date()),
        "duration_seconds": snapshot.duration_seconds,
        "source_type": source_type,
    }


def _window_row(
    bucket_id: str,
    user_id: str,
    hostname: str,
    snapshot: _Snapshot,
) -> dict[str, object] | None:
    app = str(snapshot.data.get("app") or "unknown")
    if app.casefold() in {"loginwindow", "lockapp.exe"}:
        return None
    title = str(snapshot.data.get("title") or "")
    url = str(snapshot.data.get("url") or "")
    category = categorize_window_event(app, title, url)
    return {
        **_base_row(bucket_id, user_id, hostname, "window", snapshot),
        "app": app,
        "category": category.category,
        "category_confidence": category.confidence,
        "category_rule": category.rule,
        "title": title,
        "url": url,
    }


def _web_row(
    bucket_id: str,
    user_id: str,
    hostname: str,
    snapshot: _Snapshot,
) -> dict[str, object]:
    url = str(snapshot.data.get("url") or "")
    title = str(snapshot.data.get("title") or "")
    category = categorize_web_event(url, title)
    return {
        **_base_row(bucket_id, user_id, hostname, "web", snapshot),
        "domain": category.normalized_domain,
        "category": category.category,
        "category_confidence": category.confidence,
        "category_rule": category.rule,
        "url": url,
        "title": title,
        "audible": snapshot.data.get("audible"),
        "tab_count": snapshot.data.get(
            "tabCount",
            snapshot.data.get("tab_count"),
        ),
    }


def _input_row(
    bucket_id: str,
    user_id: str,
    hostname: str,
    snapshot: _Snapshot,
) -> dict[str, object]:
    delta_x = _number(snapshot.data.get("deltaX"))
    delta_y = _number(snapshot.data.get("deltaY"))
    scroll_x = _number(snapshot.data.get("scrollX"))
    scroll_y = _number(snapshot.data.get("scrollY"))
    return {
        **_base_row(bucket_id, user_id, hostname, "input", snapshot),
        "presses": _number(snapshot.data.get("presses")),
        "clicks": _number(snapshot.data.get("clicks")),
        "mouse_distance": math.hypot(delta_x, delta_y),
        "scroll_abs": abs(scroll_x) + abs(scroll_y),
    }


def _hostname(bucket: Mapping[str, object]) -> str:
    hostname = bucket.get("hostname")
    if isinstance(hostname, str) and hostname:
        return hostname
    metadata = bucket.get("data")
    if isinstance(metadata, Mapping):
        fallback = metadata.get("hostname")
        if isinstance(fallback, str) and fallback:
            return fallback
    return "unknown"


def _finalize(
    rows: list[dict[str, object]],
    columns: list[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .reindex(columns=columns)
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )


def parse_activitywatch_events(
    buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
) -> ParsedActivityWatchEvents:
    """Convert live ActivityWatch bucket events to inference tables

    Args:
        buckets: bucket metadata with an events sequence on each bucket
        user_id: participant identifier attached to every parsed event

    Returns:
        normalized window, web and input event tables
    """

    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be blank")

    rows: dict[str, list[dict[str, object]]] = {
        "window": [],
        "web": [],
        "input": [],
    }
    for bucket_id, bucket in buckets.items():
        source_type = SUPPORTED_BUCKET_TYPES.get(str(bucket.get("type") or ""))
        if source_type is None:
            continue
        raw_events = bucket.get("events", [])
        if not isinstance(raw_events, Sequence) or isinstance(
            raw_events,
            (str, bytes),
        ):
            raise TypeError(f"{bucket_id} events must be a sequence")

        hostname = _hostname(bucket)
        for snapshot in _event_snapshots(raw_events):
            if source_type == "window":
                row = _window_row(
                    bucket_id,
                    normalized_user_id,
                    hostname,
                    snapshot,
                )
            elif source_type == "web":
                row = _web_row(
                    bucket_id,
                    normalized_user_id,
                    hostname,
                    snapshot,
                )
            else:
                row = _input_row(
                    bucket_id,
                    normalized_user_id,
                    hostname,
                    snapshot,
                )
            if row is not None:
                rows[source_type].append(row)

    return ParsedActivityWatchEvents(
        window=_finalize(rows["window"], WINDOW_COLUMNS),
        web=_finalize(rows["web"], WEB_COLUMNS),
        input=_finalize(rows["input"], INPUT_COLUMNS),
    )
