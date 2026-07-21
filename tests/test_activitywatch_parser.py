"""Test the live ActivityWatch inference parser"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from trustme_xai.feature_pipeline import parse_activitywatch_events


def test_parser_builds_the_three_inference_streams() -> None:
    buckets = {
        "aw-watcher-window_mac": {
            "type": "currentwindow",
            "hostname": "mac",
            "events": [
                {
                    "timestamp": "2026-01-01T09:00:00Z",
                    "duration": timedelta(seconds=60),
                    "data": {"app": "Code", "title": "main.py"},
                },
            ],
        },
        "aw-watcher-web-chrome_mac": {
            "type": "web.tab.current",
            "hostname": "mac",
            "events": [
                {
                    "timestamp": "2026-01-01T09:01:00Z",
                    "duration": 30,
                    "data": {
                        "url": "https://github.com/openai",
                        "title": "GitHub",
                        "audible": False,
                        "tabCount": 2,
                    },
                },
            ],
        },
        "aw-watcher-input_mac": {
            "type": "os.hid.input",
            "hostname": "mac",
            "events": [
                {
                    "timestamp": "2026-01-01T09:02:00Z",
                    "duration": 5,
                    "data": {
                        "presses": 3,
                        "clicks": 1,
                        "deltaX": 3,
                        "deltaY": 4,
                        "scrollX": 0,
                        "scrollY": -2,
                    },
                },
            ],
        },
        "aw-watcher-afk_mac": {
            "type": "afkstatus",
            "hostname": "mac",
            "events": [],
        },
    }

    events = parse_activitywatch_events(buckets, "participant-1")

    assert len(events.window) == 1
    assert len(events.web) == 1
    assert len(events.input) == 1
    assert events.window.loc[0, "category"] == "development"
    assert events.web.loc[0, "domain"] == "github.com"
    assert events.input.loc[0, "mouse_distance"] == 5.0
    assert events.input.loc[0, "scroll_abs"] == 2.0
    assert events.window.loc[0, "timestamp"] == pd.Timestamp(
        "2026-01-01 10:00:00",
    )


def test_parser_keeps_the_longest_heartbeat_snapshot() -> None:
    repeated = {
        "timestamp": "2026-01-01T09:00:00Z",
        "data": {"app": "Finder", "title": "Downloads"},
    }
    buckets = {
        "window": {
            "type": "currentwindow",
            "events": [
                {**repeated, "duration": 25},
                {**repeated, "duration": 173},
                {**repeated, "duration": 49},
            ],
        },
    }

    events = parse_activitywatch_events(buckets, "participant-1")

    assert len(events.window) == 1
    assert events.window.loc[0, "duration_seconds"] == 173


def test_parser_skips_invalid_and_locked_window_events() -> None:
    buckets = {
        "window": {
            "type": "currentwindow",
            "events": [
                {
                    "timestamp": "2026-01-01T09:00:00Z",
                    "duration": 0,
                    "data": {"app": "Code"},
                },
                {
                    "timestamp": "2026-01-01T09:01:00Z",
                    "duration": 10,
                    "data": {"app": "loginwindow"},
                },
            ],
        },
    }

    events = parse_activitywatch_events(buckets, "participant-1")

    assert events.window.empty
    assert events.web.empty
    assert events.input.empty
