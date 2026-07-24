"""Build the dashboard prediction report"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

import pandas as pd

from trustme_xai.contracts import QUESTION_TARGETS
from trustme_xai.feature_pipeline.event_contract import normalize_timestamp
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_SET,
    PRODUCTION_WINDOW_MINUTES,
)
from trustme_xai.inference.model_runtime import CURRENT_MODEL_SHA256, ModelBundle
# from trustme_xai.inference.shap_explainer import (
#     LocalShapExplanation,
#     explain_target_prediction,
# )


class QuestionPrediction(TypedDict):
    id: str
    prediction: float


class PredictionReport(TypedDict):
    generated_at: str
    participant_id: str
    as_of: str
    window_minutes: int
    feature_set: str
    artifact_fingerprint: str
    questions: list[QuestionPrediction]


def build_prediction_report(
    bundle: ModelBundle,
    current_features: pd.DataFrame,
    user_id: str,
    as_of: str | pd.Timestamp,
) -> PredictionReport:
    """Build predictions for q1-q9

    Args:
        bundle: fitted models for all answer targets
        current_features: one row of current model features
        user_id: participant linked to the feature row
        as_of: end time of the feature window

    Returns:
        PredictionReport for the dashboard
    """
    if len(current_features) != 1:
        raise ValueError("current_features must contain exactly one row")
    missing = [
        column
        for column in ("user_id", "timestamp")
        if column not in current_features.columns
    ]
    if missing:
        raise ValueError(f"current_features is missing report fields: {missing}")

    feature_user = str(current_features.iloc[0]["user_id"])
    if feature_user != user_id:
        raise ValueError("user_id does not match current_features")
    report_time = normalize_timestamp(as_of)
    feature_time = normalize_timestamp(current_features.iloc[0]["timestamp"])
    if feature_time != report_time:
        raise ValueError("as_of does not match current_features")

    for target in QUESTION_TARGETS.values():
        if target not in bundle.target_models:
            raise ValueError(f"bundle is missing target model: {target}")

    predictions = bundle.predict(current_features).iloc[0]
    questions: list[QuestionPrediction] = [
        {
            "id": question_id,
            "prediction": float(predictions[target]),
        }
        for question_id, target in QUESTION_TARGETS.items()
    ]

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "generated_at": generated_at,
        "participant_id": user_id,
        "as_of": report_time.isoformat(),
        "window_minutes": PRODUCTION_WINDOW_MINUTES,
        "feature_set": PRODUCTION_FEATURE_SET,
        "artifact_fingerprint": CURRENT_MODEL_SHA256,
        "questions": questions,
    }
