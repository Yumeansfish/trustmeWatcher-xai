"""Build the semantic production prediction report"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from trustme_xai.contracts import TARGET_TITLES
from trustme_xai.inference.model_runtime import ModelBundle

PREDICTION_REPORT_SCHEMA_VERSION = 3


def build_prediction_report(
    bundle: ModelBundle,
    current_features: pd.DataFrame,
    participant_id: str,
    as_of: str | pd.Timestamp,
) -> dict[str, Any]:
    """Build one report with seven positive target predictions

    Args:
        bundle: fitted production model bundle
        current_features: one current feature row
        participant_id: participant linked to the row
        as_of: end of the prediction window

    Returns:
        semantic production prediction report
    """
    if len(current_features) != 1:
        raise ValueError("current_features must contain exactly one row")
    if not {"user_id", "timestamp"}.issubset(current_features.columns):
        raise ValueError("current_features must contain user_id and timestamp")
    row = current_features.iloc[0]
    if str(row["user_id"]) != participant_id:
        raise ValueError("participant_id does not match the feature row")
    if pd.Timestamp(row["timestamp"]) != pd.Timestamp(as_of):
        raise ValueError("as_of does not match the feature row")

    prediction_table = bundle.predict(current_features)
    predictions = []
    for target in bundle.targets:
        model = bundle.target_models[target]
        predictions.append(
            {
                "target": target,
                "title": TARGET_TITLES[target],
                "prediction": float(prediction_table[target].iloc[0]),
                "model": model.model_name,
                "validation_mse": float(model.metrics["validation_mse"]),
                "test_mse": float(model.metrics["test_mse"]),
            },
        )

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": PREDICTION_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "participant_id": participant_id,
        "prediction_timestamp": pd.Timestamp(as_of).isoformat(),
        "window_minutes": 60,
        "model_version": bundle.metadata["model_version"],
        "normalization": bundle.metadata["normalization"],
        "predictions": predictions,
    }
