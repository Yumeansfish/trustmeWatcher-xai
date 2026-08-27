"""Exercise the canonical runtime feature interface."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.action_allocations import (
    ACTION_ALLOCATION_COLUMNS,
)
from trustme_xai.feature_pipeline.production_features import (
    ACTIONABLE_CATEGORY_COLUMNS,
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.runtime_features import (
    RUNTIME_FEATURE_COLUMNS,
    build_runtime_feature_row,
)
from trustme_xai.feature_pipeline.self_report_features import all_history_columns


def _model(minimum_history_rows: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        targets=list(MODEL_TARGETS),
        model_version="runtime-feature-test-v1",
        minimum_complete_history_rows=minimum_history_rows,
    )


def _event(
    timestamp: str,
    duration_minutes: float,
    app: str,
    title: str,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "duration": duration_minutes * 60.0,
        "data": {"app": app, "title": title},
    }


def _buckets() -> dict[str, dict[str, object]]:
    return {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                _event("2026-07-30T09:30:00+02:00", 10, "Code", "main.py"),
                _event("2026-07-30T09:40:00+02:00", 10, "Zoom", "Meeting"),
                _event("2026-07-30T09:50:00+02:00", 5, "Steam", "Game"),
                _event("2026-07-30T09:55:00+02:00", 5, "ChatGPT", "Chat"),
            ],
        },
    }


def _complete_report() -> dict[str, object]:
    return {
        "timestamp": "2026-07-29T10:00:00+02:00",
        **{target: 3.0 for target in MODEL_TARGETS},
    }


def test_runtime_feature_row_has_one_canonical_schema() -> None:
    row = build_runtime_feature_row(
        model=_model(),
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
    )

    assert list(row.columns) == RUNTIME_FEATURE_COLUMNS
    assert RUNTIME_FEATURE_COLUMNS == [
        "user_id",
        "timestamp",
        *PRODUCTION_FEATURE_COLUMNS,
        *[
            column
            for column in ACTIONABLE_CATEGORY_COLUMNS
            if column not in PRODUCTION_FEATURE_COLUMNS
        ],
        *ACTION_ALLOCATION_COLUMNS,
        *all_history_columns(),
    ]
    assert row.loc[0, list(ACTION_ALLOCATION_COLUMNS)].to_dict() == {
        "action_minutes_development": 10.0,
        "action_minutes_writing": 0.0,
        "action_minutes_research": 0.0,
        "action_minutes_communication": 10.0,
        "action_minutes_media": 5.0,
        "action_minutes_personal_distraction": 0.0,
        "action_minutes_other": 5.0,
    }


def test_runtime_feature_interface_has_no_shape_flags() -> None:
    parameters = inspect.signature(build_runtime_feature_row).parameters

    assert "include_actionable_categories" not in parameters
    assert "include_action_classifier_features" not in parameters


def test_runtime_feature_row_enforces_model_history_requirement() -> None:
    with pytest.raises(
        ValueError,
        match="needs one complete earlier StreamDeck check-in",
    ):
        build_runtime_feature_row(
            model=_model(minimum_history_rows=1),
            activitywatch_buckets=_buckets(),
            user_id="user_e2e",
            as_of="2026-07-30T10:00:00+02:00",
        )

    row = build_runtime_feature_row(
        model=_model(minimum_history_rows=1),
        activitywatch_buckets=_buckets(),
        user_id="user_e2e",
        as_of="2026-07-30T10:00:00+02:00",
        past_self_reports=[_complete_report()],
    )

    assert row.loc[0, "history__productivity__last"] == 3.0
    assert row.loc[0, "timestamp"] == pd.Timestamp("2026-07-30 10:00:00")


def test_runtime_feature_row_requires_window_or_web_source() -> None:
    with pytest.raises(ValueError, match="window or web bucket is required"):
        build_runtime_feature_row(
            model=_model(),
            activitywatch_buckets={
                "input": {"type": "os.hid.input", "events": []},
            },
            user_id="user_e2e",
            as_of="2026-07-30T10:00:00+02:00",
        )


def test_runtime_feature_row_ignores_future_activity() -> None:
    future = {
        "window": {
            "type": "currentwindow",
            "hostname": "test-host",
            "events": [
                _event("2026-07-30T10:30:00+02:00", 10, "Code", "main.py"),
            ],
        },
    }

    with pytest.raises(ValueError, match="no window or web activity"):
        build_runtime_feature_row(
            model=_model(),
            activitywatch_buckets=future,
            user_id="user_e2e",
            as_of="2026-07-30T10:00:00+02:00",
        )
