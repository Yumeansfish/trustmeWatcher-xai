from __future__ import annotations

import pandas as pd
import pytest

from trustme_xai.contracts import ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.history_features import (
    add_rolling_history_features,
    build_context_features,
)
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
    build_production_features,
)
from trustme_xai.feature_pipeline.window_features import build_window_features

EXPECTED_PRODUCTION_25 = [
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


def inference_events() -> ParsedActivityWatchEvents:
    window = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 09:30:00"),
                "event_end": pd.Timestamp("2026-01-01 09:50:00"),
                "source_type": "window",
                "app": "Code",
                "category": "development",
                "category_confidence": "high",
            },
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 09:50:00"),
                "event_end": pd.Timestamp("2026-01-01 10:00:00"),
                "source_type": "window",
                "app": "Slack",
                "category": "communication",
                "category_confidence": "high",
            },
        ],
    )
    web = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 09:40:00"),
                "event_end": pd.Timestamp("2026-01-01 09:45:00"),
                "source_type": "web",
                "category": "development",
                "category_confidence": "high",
            },
        ],
    )
    input_events = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 09:35:00"),
                "event_end": pd.Timestamp("2026-01-01 09:35:30"),
                "duration_seconds": 30.0,
                "presses": 10.0,
                "clicks": 2.0,
                "mouse_distance": 1000.0,
                "scroll_abs": 200.0,
            },
        ],
    )
    return ParsedActivityWatchEvents(
        window=window,
        web=web,
        input=input_events,
    )


def events_with_prior_context() -> ParsedActivityWatchEvents:
    events = inference_events()
    prior = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 08:00:00"),
                "event_end": pd.Timestamp("2026-01-01 08:30:00"),
                "source_type": "window",
                "app": "Code",
                "category": "development",
                "category_confidence": "high",
            },
        ],
    )
    return events._replace(window=pd.concat([events.window, prior], ignore_index=True))


def test_context_features_use_only_time_before_current_window() -> None:
    timestamps = pd.DataFrame(
        [{"user_id": "user1", "timestamp": pd.Timestamp("2026-01-01 10:00:00")}],
    )
    events = events_with_prior_context()
    current = build_window_features(events, timestamps)
    row = build_context_features(events, current).iloc[0]

    assert row["total_active_minutes"] == 30.0
    assert row["context_24h_ratio_personal_distraction"] == 0.0
    assert row["context_7d_ratio_development"] == 1.0
    assert row["context_7d_ratio_media"] == 0.0
    assert row["context_7d_ratio_research"] == 0.0


def test_rolling_history_uses_prior_rows_only() -> None:
    table = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
                "input_load": 10.0,
                "switch_rate": 1.0,
            },
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 11:00:00"),
                "input_load": 20.0,
                "switch_rate": 2.0,
            },
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 12:30:00"),
                "input_load": 40.0,
                "switch_rate": 4.0,
            },
        ],
    )

    out = add_rolling_history_features(table, ["input_load", "switch_rate"])

    assert out.loc[0, "minutes_since_prev1_window"] == 0.0
    assert out.loc[1, "minutes_since_prev1_window"] == 60.0
    assert out.loc[2, "minutes_since_prev1_window"] == 90.0
    assert out.loc[0, "rolling_mean_3_input_load"] == 0.0
    assert out.loc[1, "rolling_mean_3_input_load"] == 10.0
    assert out.loc[2, "rolling_mean_3_input_load"] == 15.0
    assert out.loc[2, "current_minus_rolling_mean_3_switch_rate"] == 2.5


def test_production_25_feature_contract_and_golden_values() -> None:
    timestamps = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 09:00:00"),
            },
            {
                "user_id": "user1",
                "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
            },
        ],
    )

    features = build_production_features(events_with_prior_context(), timestamps)
    current = features.iloc[-1]

    assert PRODUCTION_FEATURE_COLUMNS == EXPECTED_PRODUCTION_25
    assert len(set(EXPECTED_PRODUCTION_25)) == 25
    assert list(features.columns) == [
        "user_id",
        "timestamp",
        *EXPECTED_PRODUCTION_25,
    ]
    assert current[EXPECTED_PRODUCTION_25].to_numpy(dtype=float) == pytest.approx(
        [
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            60.0,
            0.0,
            0.5 / 60,
            15.0,
            0.0,
            0.0,
            -15.0,
            2.7745493121200364,
            0.0,
            0.0,
            0.0,
            1.0,
            15.0,
            0.0,
            0.0,
            1 / 3,
            1 / 30,
            0.0,
        ],
        rel=1e-12,
        abs=1e-12,
    )
