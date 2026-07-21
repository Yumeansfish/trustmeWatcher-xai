from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from trustme_xai.contracts import TARGET_QUESTIONS, ParsedActivityWatchEvents
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
    build_production_features,
)
from trustme_xai.inference.model_runtime import ModelBundle
from trustme_xai.inference.prediction_report import (
    PredictionReport,
    build_prediction_report,
)


def build_current_features(
    events: ParsedActivityWatchEvents,
    user_id: str,
    timestamp: str | pd.Timestamp,
    previous_answer_times: list[str | pd.Timestamp],
) -> pd.DataFrame:
    """Build the current feature row

    Args:
        events: categorized activitywatch event tables
        user_id: participant identifier
        timestamp: prediction time
        previous_answer_times: earlier questionnaire recording times

    Returns:
        one model-ready feature row
    """

    if not user_id.strip():
        raise ValueError("user_id must not be blank")
    user_id = user_id.strip()
    current_time = normalize_timestamp(timestamp)

    history = sorted(
        {
            value
            for value in (normalize_timestamp(item) for item in previous_answer_times)
            if value < current_time
        }
    )[-3:]
    requests = pd.DataFrame({"timestamp": [*history, current_time]})
    requests.insert(0, "user_id", user_id)
    prepared_events = clip_events_at(
        prepare_inference_events(events),
        current_time,
    )
    current_start = current_time - pd.Timedelta(
        minutes=PRODUCTION_WINDOW_MINUTES,
    )
    has_recent_activity = any(
        bool(
            (
                (table["user_id"] == user_id)
                & (table["timestamp"] < current_time)
                & (table["event_end"] > current_start)
            ).any(),
        )
        for table in (prepared_events.window, prepared_events.web)
        if not table.empty
    )
    if not has_recent_activity:
        raise ValueError("no window or web activity found in the prediction window")

    features = build_production_features(prepared_events, requests)
    return features.tail(1).reset_index(drop=True)


def predict_current(
    bundle: ModelBundle, current_features: pd.DataFrame
) -> dict[str, float]:
    """Predict all questionnaire targets

    Args:
        bundle: loaded dashboard model
        current_features: one model-ready feature row

    Returns:
        mapping from question id to prediction
    """

    if len(current_features) != 1:
        raise ValueError("current_features must contain exactly one row")
    predictions = bundle.predict(current_features)
    row = predictions.iloc[0]
    return {
        TARGET_QUESTIONS.get(target, target): float(row[target])
        for target in bundle.targets
    }


def run_inference(
    *,
    bundle: ModelBundle,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: str | pd.Timestamp,
    previous_questionnaire_times: list[str | pd.Timestamp],
) -> PredictionReport:
    """Parse ActivityWatch data and build one dashboard report

    Args:
        bundle: preloaded dashboard model
        activitywatch_buckets: bucket metadata with fetched events
        user_id: participant identifier
        as_of: prediction time
        previous_questionnaire_times: earlier questionnaire recording times

    Returns:
        dashboard prediction report
    """

    events = parse_activitywatch_events(activitywatch_buckets, user_id)
    features = build_current_features(
        events,
        user_id,
        as_of,
        previous_questionnaire_times,
    )
    return build_prediction_report(bundle, features, user_id, as_of)
