"""Test the inference timestamp boundary"""

from __future__ import annotations

import pandas as pd
import pytest

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.event_contract import (
    clip_events_at,
    normalize_request_times,
    normalize_timestamp,
    prepare_inference_events,
)


def test_aware_timestamp_becomes_zurich_local_naive() -> None:
    timestamp = normalize_timestamp("2026-07-21T12:00:00Z")

    assert timestamp == pd.Timestamp("2026-07-21 14:00:00")
    assert timestamp.tzinfo is None


def test_naive_timestamp_keeps_local_wall_time() -> None:
    assert normalize_timestamp("2026-07-21 14:00:00") == pd.Timestamp(
        "2026-07-21 14:00:00",
    )


def test_invalid_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp must be valid"):
        normalize_timestamp(None)


def test_event_and_request_tables_share_one_time_basis() -> None:
    event = pd.DataFrame(
        {
            "user_id": ["user1"],
            "timestamp": ["2026-07-21T12:00:00Z"],
            "event_end": ["2026-07-21T12:05:00Z"],
            "app": ["Code"],
            "category": ["development"],
        },
    )
    events = ParsedActivityWatchEvents(
        window=event,
        web=pd.DataFrame(),
        input=pd.DataFrame(),
    )
    requests = pd.DataFrame({"timestamp": ["2026-07-21T12:10:00Z"]})

    normalized_events = prepare_inference_events(events)
    normalized_requests = normalize_request_times(requests)

    assert normalized_events.window.loc[0, "timestamp"] == pd.Timestamp(
        "2026-07-21 14:00:00",
    )
    assert normalized_events.window.loc[0, "event_end"] == pd.Timestamp(
        "2026-07-21 14:05:00",
    )
    assert normalized_requests.loc[0, "timestamp"] == pd.Timestamp(
        "2026-07-21 14:10:00",
    )


def test_nonempty_event_table_requires_runtime_columns() -> None:
    events = ParsedActivityWatchEvents(
        window=pd.DataFrame(
            {
                "timestamp": ["2026-07-21T12:00:00Z"],
                "event_end": ["2026-07-21T12:05:00Z"],
            },
        ),
        web=pd.DataFrame(),
        input=pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="window events are missing columns"):
        prepare_inference_events(events)


def test_input_event_values_must_be_non_negative() -> None:
    events = ParsedActivityWatchEvents(
        window=pd.DataFrame(),
        web=pd.DataFrame(),
        input=pd.DataFrame(
            {
                "user_id": ["user1"],
                "timestamp": ["2026-07-21T12:00:00Z"],
                "event_end": ["2026-07-21T12:00:30Z"],
                "duration_seconds": [30],
                "presses": [-1],
                "clicks": [0],
                "mouse_distance": [0],
                "scroll_abs": [0],
            },
        ),
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        prepare_inference_events(events)


def test_clip_events_at_removes_future_and_clips_overlap() -> None:
    events = ParsedActivityWatchEvents(
        window=pd.DataFrame(
            {
                "user_id": ["user1", "user1"],
                "timestamp": [
                    "2026-07-21T12:00:00Z",
                    "2026-07-21T13:00:00Z",
                ],
                "event_end": [
                    "2026-07-21T12:30:00Z",
                    "2026-07-21T13:30:00Z",
                ],
                "app": ["Code", "Code"],
                "category": ["development", "development"],
            },
        ),
        web=pd.DataFrame(),
        input=pd.DataFrame(),
    )
    prepared = prepare_inference_events(events)

    clipped = clip_events_at(prepared, "2026-07-21T12:15:00Z")

    assert len(clipped.window) == 1
    assert clipped.window.loc[0, "event_end"] == pd.Timestamp(
        "2026-07-21 14:15:00",
    )
