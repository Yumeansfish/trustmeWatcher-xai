"""Zero-Sum Swap search engine for actionable counterfactual recourse"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trustme_xai.inference.counterfactual_constraints import (
    ACTIONABLE_CATEGORIES,
    check_time_budget,
    validate_counterfactual,
)
from trustme_xai.inference.model_runtime import ModelBundle

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
MAX_SHIFTS: int = 12
DEFAULT_SENSITIVITY_GAMMA: float = 1.0




def find_counterfactual(
    bundle: ModelBundle,
    feature_row: pd.DataFrame,
    target: str,
    desired_score: float,
    sensitivity_gamma: float = DEFAULT_SENSITIVITY_GAMMA,
) -> dict[str, Any]:
    """Search for a zero-sum category swap

    Args:
        bundle: fitted production model bundle
        feature_row: single-row pd.DataFrame containing baseline features
        target: target metric identifier string
        desired_score: target desired continuous score
        sensitivity_gamma: sensitivity scaling multiplier (default 1.0)

    Returns:
        counterfactual result with the selected shifts
    """
    if len(feature_row) != 1:
        raise ValueError(
            f"feature_row must contain exactly 1 row, got {len(feature_row)}",
        )
    if not {"user_id", "timestamp"}.issubset(feature_row.columns):
        raise ValueError("feature_row must contain user_id and timestamp columns")

    if target not in bundle.target_models:
        raise ValueError(f"target not found in model bundle: {target}")
    baseline_pred = bundle.predict_target(feature_row, target)

    row_data = feature_row.iloc[0].to_dict()
    participant_id = str(row_data["user_id"])
    ts_str = pd.Timestamp(row_data["timestamp"]).isoformat()
    want_increase = desired_score > baseline_pred
    is_satisfied = (
        baseline_pred >= desired_score
        if want_increase
        else baseline_pred <= desired_score
    )

    if is_satisfied:
        return {
            "participant_id": participant_id,
            "prediction_timestamp": ts_str,
            "target": target,
            "predicted_score": round(baseline_pred, 4),
            "desired_score": round(float(desired_score), 4),
            "calibrated_projected_score": round(baseline_pred, 4),
            "success": True,
            "shifts": [],
        }

    current = dict(row_data)
    actionable_in_row = [
        category
        for category in ACTIONABLE_CATEGORIES
        if category in current
        and float(current.get(category, 0.0)) >= STEP_SIZE_MINUTES
    ]
    if not actionable_in_row:
        actionable_in_row = [
            category
            for category in current
            if category.startswith(("time_", "minutes_"))
        ]

    best_pred = baseline_pred
    deltas_acc: dict[str, float] = {}

    for _ in range(MAX_SHIFTS):
        improved = False
        best_step_pred = best_pred
        best_donor, best_receiver, best_step_size = None, None, 0.0

        for donor in actionable_in_row:
            avail = float(current.get(donor, 0.0))
            if avail < STEP_SIZE_MINUTES:
                continue

            for receiver in ACTIONABLE_CATEGORIES:
                deltas = {
                    donor: -STEP_SIZE_MINUTES,
                    receiver: STEP_SIZE_MINUTES,
                }
                if receiver == donor or not check_time_budget(current, deltas):
                    continue

                cand = dict(current)
                cand[donor] -= STEP_SIZE_MINUTES
                cand[receiver] = float(cand.get(receiver, 0.0)) + STEP_SIZE_MINUTES
                validate_counterfactual(row_data, cand)

                cand_pred = bundle.predict_target(pd.DataFrame([cand]), target)
                is_better = (
                    cand_pred > best_step_pred
                    if want_increase
                    else cand_pred < best_step_pred
                )

                if is_better:
                    best_step_pred = cand_pred
                    best_donor, best_receiver = donor, receiver
                    best_step_size = STEP_SIZE_MINUTES
                    improved = True

        if improved and best_donor and best_receiver:
            best_pred = best_step_pred
            current[best_donor] -= best_step_size
            current[best_receiver] = (
                float(current.get(best_receiver, 0.0)) + best_step_size
            )
            deltas_acc[best_donor] = deltas_acc.get(best_donor, 0.0) - best_step_size
            deltas_acc[best_receiver] = (
                deltas_acc.get(best_receiver, 0.0) + best_step_size
            )

            found_solution = (
                best_pred >= desired_score
                if want_increase
                else best_pred <= desired_score
            )
            if found_solution:
                break
        else:
            break

    shifts = [
        {
            "category": category,
            "time_spent": float(row_data.get(category, 0.0)),
            "delta_minutes": delta,
        }
        for category, delta in deltas_acc.items()
        if abs(delta) > 1e-4
    ]

    raw_delta = best_pred - baseline_pred
    calibrated_projected = baseline_pred + (raw_delta * sensitivity_gamma)

    is_satisfied_final = (
        calibrated_projected >= desired_score
        if want_increase
        else calibrated_projected <= desired_score
    )

    return {
        "participant_id": participant_id,
        "prediction_timestamp": ts_str,
        "target": target,
        "predicted_score": round(baseline_pred, 4),
        "desired_score": round(float(desired_score), 4),
        "calibrated_projected_score": round(calibrated_projected, 4),
        "success": is_satisfied_final,
        "shifts": shifts,
    }
