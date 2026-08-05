"""Run production inference from raw ActivityWatch buckets"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
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
    PRODUCTION_FEATURE_COLUMNS,
    PRODUCTION_WINDOW_MINUTES,
    build_production_features,
)
from trustme_xai.feature_pipeline.self_report_features import (
    add_self_report_features,
    all_history_columns,
)
from trustme_xai.inference.model_runtime import ModelBundle
from trustme_xai.inference.prediction_report import build_prediction_report


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
    normalized = {
        normalize_timestamp(value)
        for value in previous_questionnaire_times
    }
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


def build_current_features(
    events: ParsedActivityWatchEvents,
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] = (),
    past_self_reports: Sequence[Mapping[str, object]] = (),
) -> pd.DataFrame:
    """Build one current production feature row

    Args:
        events: parsed ActivityWatch event tables
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        one pd.DataFrame row with activity and self-report features
    """
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be blank")
    current_time = normalize_timestamp(as_of)
    clipped = clip_events_at(prepare_inference_events(events), current_time)
    if not _has_current_activity(clipped, current_time):
        raise ValueError("no window or web activity in the current 60-minute window")

    report_times = [
        report["timestamp"]
        for report in past_self_reports
        if "timestamp" in report
    ]
    requests = _request_times(
        normalized_user_id,
        current_time,
        [*previous_questionnaire_times, *report_times],
    )
    features = build_production_features(clipped, requests)
    current = features.loc[
        features["user_id"].astype(str).eq(normalized_user_id)
        & features["timestamp"].eq(current_time)
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
    expected = [
        "user_id",
        "timestamp",
        *PRODUCTION_FEATURE_COLUMNS,
        *all_history_columns(),
    ]
    if list(current.columns) != expected:
        raise ValueError("runtime feature schema does not match the model")
    return current.reset_index(drop=True)


def predict_current(
    bundle: ModelBundle,
    current_features: pd.DataFrame,
) -> dict[str, float]:
    """Predict all saved targets for one feature row

    Args:
        bundle: fitted production model bundle
        current_features: one current feature row

    Returns:
        target names mapped to finite predictions
    """
    if len(current_features) != 1:
        raise ValueError("current_features must contain exactly one row")
    predictions = bundle.predict(current_features)
    values = predictions.iloc[0].to_dict()
    result = {target: float(values[target]) for target in bundle.targets}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("runtime predictions must be finite")
    return result


def _validate_bucket_sources(
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
) -> None:
    source_types = {
        str(bucket.get("type") or "")
        for bucket in activitywatch_buckets.values()
    }
    if not source_types.intersection({"currentwindow", "web.tab.current"}):
        raise ValueError("ActivityWatch window or web bucket is required")
    if "os.hid.input" not in source_types:
        raise ValueError("ActivityWatch input bucket is required")


def build_current_features_from_buckets(
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] = (),
    past_self_reports: Sequence[Mapping[str, object]] = (),
) -> pd.DataFrame:
    """Build one production row from raw ActivityWatch buckets

    Args:
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        one pd.DataFrame row with activity and self-report features
    """
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("user_id must not be blank")
    _validate_bucket_sources(activitywatch_buckets)
    current_time = normalize_timestamp(as_of)
    events = parse_activitywatch_events(
        activitywatch_buckets,
        normalized_user_id,
    )
    return build_current_features(
        events,
        normalized_user_id,
        current_time,
        previous_questionnaire_times,
        past_self_reports,
    )


def run_inference(
    bundle: ModelBundle,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] | None = None,
    past_self_reports: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Run the complete production inference path

    Args:
        bundle: fitted production model bundle
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        semantic seven-target prediction report
    """
    current_time = normalize_timestamp(as_of)
    current = build_current_features_from_buckets(
        activitywatch_buckets,
        user_id,
        current_time,
        previous_questionnaire_times or (),
        past_self_reports or (),
    )
    normalized_user_id = user_id.strip()
    return build_prediction_report(
        bundle,
        current,
        normalized_user_id,
        current_time,
    )
