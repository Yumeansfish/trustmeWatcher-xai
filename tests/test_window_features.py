from __future__ import annotations

import pandas as pd
import pytest

from trustme_xai.contracts import (
    INPUT_COLUMNS,
    WEB_COLUMNS,
    WINDOW_COLUMNS,
    ParsedActivityWatchEvents,
)
from trustme_xai.feature_pipeline.window_features import build_window_features


def make_events() -> ParsedActivityWatchEvents:
    window = pd.DataFrame(
        [
            {
                "bucket": "window",
                "user_id": "user1",
                "hostname": "host",
                "timestamp": pd.Timestamp("2026-01-01 09:30:00"),
                "event_end": pd.Timestamp("2026-01-01 09:50:00"),
                "date": "2026-01-01",
                "duration_seconds": 20 * 60,
                "source_type": "window",
                "app": "Code",
                "category": "development",
                "category_confidence": "high",
                "category_rule": "app:code",
                "title": "main.py",
            },
            {
                "bucket": "window",
                "user_id": "user1",
                "hostname": "host",
                "timestamp": pd.Timestamp("2026-01-01 09:50:00"),
                "event_end": pd.Timestamp("2026-01-01 10:00:00"),
                "date": "2026-01-01",
                "duration_seconds": 10 * 60,
                "source_type": "window",
                "app": "Slack",
                "category": "communication",
                "category_confidence": "high",
                "category_rule": "app:slack",
                "title": "chat",
            },
        ],
        columns=WINDOW_COLUMNS,
    )
    web = pd.DataFrame(
        [
            {
                "bucket": "web",
                "user_id": "user1",
                "hostname": "host",
                "timestamp": pd.Timestamp("2026-01-01 09:40:00"),
                "event_end": pd.Timestamp("2026-01-01 09:45:00"),
                "date": "2026-01-01",
                "duration_seconds": 5 * 60,
                "source_type": "web",
                "domain": "github.com",
                "category": "development",
                "category_confidence": "high",
                "category_rule": "domain:github",
                "url": "https://github.com",
                "title": "repo",
                "audible": False,
                "tab_count": 3,
            },
        ],
        columns=WEB_COLUMNS,
    )
    input_events = pd.DataFrame(
        [
            {
                "bucket": "input",
                "user_id": "user1",
                "hostname": "host",
                "timestamp": pd.Timestamp("2026-01-01 09:35:00"),
                "event_end": pd.Timestamp("2026-01-01 09:35:30"),
                "date": "2026-01-01",
                "duration_seconds": 30,
                "source_type": "input",
                "presses": 10.0,
                "clicks": 2.0,
                "mouse_distance": 1000.0,
                "scroll_abs": 200.0,
            },
        ],
        columns=INPUT_COLUMNS,
    )
    return ParsedActivityWatchEvents(window=window, web=web, input=input_events)


def test_build_window_features_from_parsed_aw_tables() -> None:
    timestamps = pd.DataFrame(
        [{"user_id": "user1", "timestamp": pd.Timestamp("2026-01-01 10:00:00")}],
    )

    features = build_window_features(make_events(), timestamps)
    row = features.iloc[0]

    assert row["window_minutes"] == 60
    assert row["app_switch_count"] == 1.0
    assert row["total_active_minutes"] == 30.0
    assert row["time_personal_distraction"] == 0.0
    assert row["mean_focus_block_minutes"] == 15.0
    assert row["short_fragment_count_2m"] == 0.0
    assert row["switch_rate"] == pytest.approx(1 / 30)
    assert row["input_active_ratio"] == pytest.approx(0.5 / 60)
    assert row["input_load"] == 15.0
    assert row["ratio_communication"] == pytest.approx(1 / 3)


def test_overlapping_web_activity_replaces_window_activity() -> None:
    events = make_events()
    web = events.web.copy()
    web["category"] = "personal_distraction"
    events = events._replace(web=web)
    timestamps = pd.DataFrame(
        [{"user_id": "user1", "timestamp": pd.Timestamp("2026-01-01 10:00:00")}],
    )

    row = build_window_features(events, timestamps).iloc[0]

    assert row["total_active_minutes"] == 30.0
    assert row["time_personal_distraction"] == 5.0
    assert row["ratio_communication"] == pytest.approx(1 / 3)
    assert row["mean_focus_block_minutes"] == 7.5
