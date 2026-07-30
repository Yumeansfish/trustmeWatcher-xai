"""Zero-Sum Swap search engine for actionable counterfactual recourse"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trustme_xai.inference.counterfactual_constraints import (
    ACTIONABLE_CATEGORIES,
    check_time_budget,
    validate_counterfactual,
)
from trustme_xai.inference.ensemble_bundle import EnsembleBundle, TargetMetric

DONOR_PRIORITY: tuple[str, ...] = (
    "minutes_media_social_media",
    "minutes_media_games",
    "minutes_media_video",
    "minutes_comms_im",
    "minutes_comms_email",
    "minutes_uncategorized",
    "time_personal_distraction",
    "time_media",
    "time_communication",
    "time_research",
    "time_writing",
    "time_development",
)

RECEIVER_PRIORITY: tuple[str, ...] = (
    "minutes_work_programming",
    "minutes_work_research_and_reading",
    "minutes_work_office",
    "minutes_work",
    "time_development",
    "time_writing",
    "time_research",
    "time_communication",
    "time_media",
    "time_personal_distraction",
)

STEP_SIZE_MINUTES: float = 5.0
MAX_SHIFTS: int = 3


def find_counterfactual(
    bundle: EnsembleBundle,
    feature_row: pd.DataFrame,
    target: str,
    desired_score: float,
) -> dict[str, Any]:
    """Run Zero-Sum Swap search evaluated against 5-block ensemble predictor

    Args:
        bundle: loaded 5-block EnsembleBundle instance
        feature_row: single-row pd.DataFrame containing baseline features
        target: target metric identifier string
        desired_score: target desired continuous score

    Returns:
        dict containing participant_id, prediction_timestamp, target, predicted_score,
        desired_score, success boolean, and minimal list of shifts
    """
    if len(feature_row) != 1:
        raise ValueError(f"feature_row must contain exactly 1 row, got {len(feature_row)}")
    if not {"user_id", "timestamp"}.issubset(feature_row.columns):
        raise ValueError("feature_row must contain user_id and timestamp columns")

    metric_key = TargetMetric(target)
    baseline_pred = float(bundle.predict_target(feature_row, metric_key))

    row_data = feature_row.iloc[0].to_dict()
    participant_id, ts_str = str(row_data["user_id"]), pd.Timestamp(row_data["timestamp"]).isoformat()
    is_neg = metric_key in (TargetMetric.STRESS, TargetMetric.FATIGUE)
    is_satisfied = baseline_pred <= desired_score if is_neg else baseline_pred >= desired_score

    if is_satisfied:
        return {
            "participant_id": participant_id,
            "prediction_timestamp": ts_str,
            "target": str(metric_key),
            "predicted_score": round(baseline_pred, 4),
            "desired_score": round(float(desired_score), 4),
            "success": True,
            "shifts": [],
        }

    current, deltas_acc = dict(row_data), {}
    donors = [c for c in DONOR_PRIORITY if float(current.get(c, 0.0)) >= STEP_SIZE_MINUTES] or [
        c for c in ACTIONABLE_CATEGORIES if float(current.get(c, 0.0)) >= STEP_SIZE_MINUTES
    ]
    best_pred, found_solution = baseline_pred, False

    for _ in range(MAX_SHIFTS):
        if found_solution:
            break
        improved = False

        for donor in donors:
            if float(current.get(donor, 0.0)) < STEP_SIZE_MINUTES:
                continue

            for receiver in RECEIVER_PRIORITY:
                if receiver == donor or not check_time_budget(current, {donor: -STEP_SIZE_MINUTES, receiver: STEP_SIZE_MINUTES}):
                    continue

                cand = dict(current)
                cand[donor] -= STEP_SIZE_MINUTES
                cand[receiver] = float(cand.get(receiver, 0.0)) + STEP_SIZE_MINUTES
                validate_counterfactual(row_data, cand)

                cand_pred = float(bundle.predict_target(pd.DataFrame([cand]), metric_key))
                is_better = cand_pred < best_pred if is_neg else cand_pred > best_pred

                if is_better:
                    best_pred, current = cand_pred, cand
                    deltas_acc[donor] = deltas_acc.get(donor, 0.0) - STEP_SIZE_MINUTES
                    deltas_acc[receiver] = deltas_acc.get(receiver, 0.0) + STEP_SIZE_MINUTES
                    improved = True
                    found_solution = cand_pred <= desired_score if is_neg else cand_pred >= desired_score
                    break
            if improved:
                break
        if not improved:
            break

    shifts = [
        {"category": cat, "time_spent": float(row_data.get(cat, 0.0)), "delta_minutes": delta}
        for cat, delta in deltas_acc.items()
        if abs(delta) > 1e-4
    ]

    return {
        "participant_id": participant_id,
        "prediction_timestamp": ts_str,
        "target": str(metric_key),
        "predicted_score": round(baseline_pred, 4),
        "desired_score": round(float(desired_score), 4),
        "success": found_solution or (best_pred <= desired_score if is_neg else best_pred >= desired_score),
        "shifts": shifts,
    }
