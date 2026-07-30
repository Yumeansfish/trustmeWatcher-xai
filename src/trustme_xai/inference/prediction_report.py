"""Live prediction report generation averaging across 5-block ensemble models"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from trustme_xai.inference.ensemble_bundle import EnsembleBundle, TargetMetric


def create_prediction_report(
    bundle: EnsembleBundle,
    feature_row: pd.DataFrame,
) -> dict[str, Any]:
    """Average live predictions across 5 block models and return prediction dictionary

    Args:
        bundle: loaded 5-block EnsembleBundle instance
        feature_row: single-row pd.DataFrame containing current model features

    Returns:
        dict containing participant_id, prediction_timestamp, and averaged predictions
    """
    if len(feature_row) != 1:
        raise ValueError(f"feature_row must contain exactly 1 row, got {len(feature_row)}")

    if "user_id" not in feature_row.columns or "timestamp" not in feature_row.columns:
        raise ValueError("feature_row must contain user_id and timestamp columns")

    row = feature_row.iloc[0]
    participant_id = str(row["user_id"])
    prediction_timestamp = pd.Timestamp(row["timestamp"]).isoformat()

    predictions: list[dict[str, Any]] = []

    for target in bundle.targets:
        target_model = bundle.target_models[target]
        # Average live predictions across the 5 block models: y_hat = 1/5 * sum(f_k(x))
        avg_score = target_model.predict(feature_row)

        predictions.append(
            {
                "target": str(target),
                "predicted_score": round(float(avg_score), 4),
                "model_family": str(target_model.family),
                "avg_val_mse": round(float(target_model.avg_val_mse), 4),
            },
        )

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    return {
        "participant_id": participant_id,
        "prediction_timestamp": prediction_timestamp,
        "generated_at": generated_at,
        "model_version": bundle.metadata.get("model_version", "5block_ensemble_v1"),
        "predictions": predictions,
    }
