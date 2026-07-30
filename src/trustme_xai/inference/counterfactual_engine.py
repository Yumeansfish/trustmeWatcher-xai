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

# Priority order for donor (source) categories to cut time from
DONOR_PRIORITY: tuple[str, ...] = (
    "time_personal_distraction",
    "time_media",
    "time_communication",
    "time_research",
    "time_writing",
    "time_development",
)

# Priority order for receiver (target) categories to add time to
RECEIVER_PRIORITY: tuple[str, ...] = (
    "time_development",
    "time_writing",
    "time_research",
    "time_communication",
    "time_media",
    "time_personal_distraction",
)

STEP_SIZE_MINUTES: float = 15.0
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

    if "user_id" not in feature_row.columns or "timestamp" not in feature_row.columns:
        raise ValueError("feature_row must contain user_id and timestamp columns")

    metric_key = TargetMetric(target)
    baseline_pred = float(bundle.predict_target(feature_row, metric_key))

    row_data = feature_row.iloc[0].to_dict()
    participant_id = str(row_data["user_id"])
    prediction_timestamp = pd.Timestamp(row_data["timestamp"]).isoformat()

    # Targets like stress or fatigue are negative (lower is better)
    is_negative_target = metric_key in (TargetMetric.STRESS, TargetMetric.FATIGUE)
    is_satisfied = (
        baseline_pred <= desired_score if is_negative_target else baseline_pred >= desired_score
    )

    if is_satisfied:
        return {
            "participant_id": participant_id,
            "prediction_timestamp": prediction_timestamp,
            "target": str(metric_key),
            "predicted_score": round(baseline_pred, 4),
            "desired_score": round(float(desired_score), 4),
            "success": True,
            "shifts": [],
        }

    # Prepare working copy of feature row dictionary
    current_features = dict(row_data)
    shifts: list[dict[str, Any]] = []

    # Filter candidate donors with non-zero available minutes
    candidate_donors = [
        c for c in DONOR_PRIORITY if float(current_features.get(c, 0.0)) >= STEP_SIZE_MINUTES
    ]
    # Filter candidate receivers different from donor
    candidate_receivers = list(RECEIVER_PRIORITY)

    best_features = dict(current_features)
    best_pred = baseline_pred
    found_solution = False

    for step in range(MAX_SHIFTS):
        if found_solution:
            break

        step_improved = False
        for donor in candidate_donors:
            donor_val = float(current_features.get(donor, 0.0))
            if donor_val < STEP_SIZE_MINUTES:
                continue

            for receiver in candidate_receivers:
                if receiver == donor:
                    continue

                proposed_deltas = {donor: -STEP_SIZE_MINUTES, receiver: STEP_SIZE_MINUTES}
                if not check_time_budget(current_features, proposed_deltas):
                    continue

                # Apply proposed zero-sum swap
                candidate_features = dict(current_features)
                candidate_features[donor] = donor_val - STEP_SIZE_MINUTES
                candidate_features[receiver] = float(candidate_features.get(receiver, 0.0)) + STEP_SIZE_MINUTES

                # Recompute derived focus metrics
                total_active = sum(float(candidate_features.get(c, 0.0)) for c in ACTIONABLE_CATEGORIES)
                candidate_features["mean_focus_block_minutes"] = round(total_active / max(1, len(ACTIONABLE_CATEGORIES)), 2)

                # Validate physical invariants
                validate_counterfactual(row_data, candidate_features)

                # Evaluate candidate prediction against ensemble model
                cand_df = pd.DataFrame([candidate_features])
                cand_pred = float(bundle.predict_target(cand_df, metric_key))

                # Check if this swap moves prediction in the right direction
                is_better = (
                    cand_pred < best_pred if is_negative_target else cand_pred > best_pred
                )

                if is_better:
                    best_pred = cand_pred
                    best_features = candidate_features
                    current_features = candidate_features
                    shifts.append(
                        {
                            "category": donor,
                            "time_spent": float(row_data.get(donor, 0.0)),
                            "delta_minutes": -STEP_SIZE_MINUTES,
                        },
                    )
                    shifts.append(
                        {
                            "category": receiver,
                            "time_spent": float(row_data.get(receiver, 0.0)),
                            "delta_minutes": STEP_SIZE_MINUTES,
                        },
                    )
                    step_improved = True
                    is_achieved = (
                        cand_pred <= desired_score if is_negative_target else cand_pred >= desired_score
                    )
                    if is_achieved:
                        found_solution = True
                    break

            if step_improved:
                break

        if not step_improved:
            break

    # Format final merged shifts dictionary (combining deltas per category)
    merged_shifts_dict: dict[str, dict[str, Any]] = {}
    for shift in shifts:
        cat = shift["category"]
        if cat not in merged_shifts_dict:
            merged_shifts_dict[cat] = {
                "category": cat,
                "time_spent": float(row_data.get(cat, 0.0)),
                "delta_minutes": 0.0,
            }
        merged_shifts_dict[cat]["delta_minutes"] += shift["delta_minutes"]

    final_shifts = [
        val for val in merged_shifts_dict.values() if abs(val["delta_minutes"]) > 1e-4
    ]

    return {
        "participant_id": participant_id,
        "prediction_timestamp": prediction_timestamp,
        "target": str(metric_key),
        "predicted_score": round(baseline_pred, 4),
        "desired_score": round(float(desired_score), 4),
        "success": found_solution or (
            best_pred <= desired_score if is_negative_target else best_pred >= desired_score
        ),
        "shifts": final_shifts,
    }
