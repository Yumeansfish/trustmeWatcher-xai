"""Build the canonical feature row consumed by runtime models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS, ParsedActivityWatchEvents
from trustme_xai.feature_pipeline.activitywatch_parser import (
    parse_activitywatch_events,
)
from trustme_xai.feature_pipeline.event_contract import (
    clip_events_at,
    normalize_timestamp,
    prepare_inference_events,
)
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_WINDOW_MINUTES,
    RUNTIME_ACTIVITY_FEATURE_COLUMNS,
    build_production_features,
)
from trustme_xai.feature_pipeline.self_report_features import (
    add_self_report_features,
    all_history_columns,
)

RUNTIME_FEATURE_COLUMNS = [
    "user_id",
    "timestamp",
    *RUNTIME_ACTIVITY_FEATURE_COLUMNS,
    *all_history_columns(),
]


class _RuntimeFeatureModel(Protocol):
    targets: list[str]
    model_version: str


def _has_current_activity(
    events: ParsedActivityWatchEvents,
    as_of: pd.Timestamp,
) -> bool:
    start = as_of - pd.Timedelta(minutes=PRODUCTION_WINDOW_MINUTES)
    return any(
        not table.empty
        and bool(
            ((table["timestamp"] < as_of) & (table["event_end"] > start)).any(),
        )
        for table in (events.window, events.web)
    )


def _request_times(
    user_id: str,
    as_of: pd.Timestamp,
    previous_questionnaire_times: Sequence[object],
) -> pd.DataFrame:
    normalized = {normalize_timestamp(value) for value in previous_questionnaire_times}
    previous = sorted(value for value in normalized if value < as_of)[-3:]
    return pd.DataFrame(
        {
            "user_id": [user_id] * (len(previous) + 1),
            "timestamp": [*previous, as_of],
        },
    )


def _history_features(
    user_id: str,
    as_of: pd.Timestamp,
    past_self_reports: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report in past_self_reports:
        if "timestamp" not in report:
            raise ValueError("past self-report needs a timestamp")
        timestamp = normalize_timestamp(report["timestamp"])
        if timestamp >= as_of:
            continue
        rows.append(
            {
                "user_id": user_id,
                "timestamp": timestamp,
                "answer_source": "streamdeck",
                **{target: report.get(target) for target in MODEL_TARGETS},
            },
        )

    rows.append(
        {
            "user_id": user_id,
            "timestamp": as_of,
            "answer_source": "prediction",
            **{target: None for target in MODEL_TARGETS},
        },
    )
    history = add_self_report_features(pd.DataFrame(rows))
    current = history.loc[history["timestamp"].eq(as_of)]
    if len(current) != 1:
        raise ValueError("self-report pipeline did not return one current row")
    return current[["user_id", "timestamp", *all_history_columns()]]


def _validate_bucket_sources(
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
) -> None:
    source_types = {
        str(bucket.get("type") or "") for bucket in activitywatch_buckets.values()
    }
    if not source_types.intersection({"currentwindow", "web.tab.current"}):
        raise ValueError("ActivityWatch window or web bucket is required")


def build_runtime_feature_row(
    model: _RuntimeFeatureModel,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] = (),
    past_self_reports: Sequence[Mapping[str, object]] = (),
) -> pd.DataFrame:
    """Build one canonical row for score or classifier inference.

    Args:
        model: runtime model declaring targets and history requirements
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        one row with the complete canonical runtime feature schema
    """
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be blank")
    current_time = normalize_timestamp(as_of)
    _validate_bucket_sources(activitywatch_buckets)

    parsed = parse_activitywatch_events(
        activitywatch_buckets,
        normalized_user_id,
    )
    events = clip_events_at(prepare_inference_events(parsed), current_time)
    if not _has_current_activity(events, current_time):
        raise ValueError("no window or web activity in the current 60-minute window")

    report_times = [
        report["timestamp"] for report in past_self_reports if "timestamp" in report
    ]
    requests = _request_times(
        normalized_user_id,
        current_time,
        [*previous_questionnaire_times, *report_times],
    )
    activity = build_production_features(events, requests)
    current = activity.loc[
        activity["user_id"].astype(str).eq(normalized_user_id)
        & activity["timestamp"].eq(current_time)
    ].copy()
    if len(current) != 1:
        raise ValueError("feature pipeline did not return one current row")

    history = _history_features(
        normalized_user_id,
        current_time,
        past_self_reports,
    )
    current = current.merge(
        history,
        on=["user_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if list(current.columns) != RUNTIME_FEATURE_COLUMNS:
        raise ValueError("runtime feature schema does not match the model")
    return current.reset_index(drop=True)
