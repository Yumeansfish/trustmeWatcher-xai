"""End-to-end integration test validating predictions and Zero-Sum counterfactual recourse"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from scripts.train_ensemble_models import train_dashboard_models
from trustme_xai.data.composite_scores import combine_state_averages, normalize_answers
from trustme_xai.inference.counterfactual_service import run_counterfactual
from trustme_xai.inference.inference_service import run_inference


def _synthetic_e2e_dataset() -> pd.DataFrame:
    rows = []
    for day in range(1, 11):
        for hour in [9, 13, 17]:
            dev = float(day * 2.5)
            distraction = float(25.0 - dev * 0.4)
            rows.append(
                {
                    "user_id": "user_e2e",
                    "timestamp": pd.Timestamp(f"2026-07-{day:02d} {hour:02d}:00:00"),
                    "time_development": dev,
                    "time_writing": 10.0,
                    "time_communication": 5.0,
                    "time_personal_distraction": max(0.0, distraction),
                    "time_media": 0.0,
                    "time_research": 0.0,
                    "q1_feelings": float((day % 5) - 2),
                    "q2_intensity": float((day % 6)),
                    "q3_tiredness": float((day % 4) + 1),
                    "q4_enthusiasm": float((day % 5) + 1),
                    "q5_immersion": float((day % 5) + 1),
                    "q8_stress": float((day % 4) + 1),
                    "q9_productivity": float(1.0 + dev * 0.1),
                },
            )
    raw_df = pd.DataFrame(rows)
    norm_df = normalize_answers(raw_df)
    return combine_state_averages(norm_df)


def test_end_to_end_pipeline() -> None:
    """Full E2E validation: features -> 5-block ensemble training -> live report -> Zero-Sum counterfactual"""
    df = _synthetic_e2e_dataset()
    feature_cols = [
        "time_development",
        "time_writing",
        "time_communication",
        "time_personal_distraction",
        "time_media",
        "time_research",
    ]

    # 1. Train 5-block ensemble model bundle
    bundle = train_dashboard_models(df, feature_columns=feature_cols)

    # 2. Simulate raw ActivityWatch bucket payload
    mock_buckets = {
        "aw-watcher-window_e2e": {
            "type": "currentwindow",
            "events": [
                {
                    "timestamp": "2026-07-30T10:00:00Z",
                    "duration": 1200.0,
                    "data": {"category": "Work > Programming", "app": "PyCharm"},
                },
                {
                    "timestamp": "2026-07-30T10:20:00Z",
                    "duration": 900.0,
                    "data": {"category": "Media > Social Media", "app": "Chrome"},
                },
            ],
        },
    }

    # 3. Execute live inference report
    inference_report = run_inference(
        bundle=bundle,
        activitywatch_buckets=mock_buckets,
        user_id="user_e2e",
        as_of=datetime(2026, 7, 30, 10, 30),
    )

    assert inference_report["participant_id"] == "user_e2e"
    assert len(inference_report["predictions"]) == 7

    # 4. Execute Zero-Sum counterfactual recourse query
    counterfactual_report = run_counterfactual(
        bundle=bundle,
        buckets=mock_buckets,
        user_id="user_e2e",
        timestamp=datetime(2026, 7, 30, 10, 30),
        target="productivity",
        desired_score=4.5,
    )

    assert counterfactual_report["participant_id"] == "user_e2e"
    assert counterfactual_report["target"] == "productivity"
    assert isinstance(counterfactual_report["success"], bool)

    # 5. Assert physical invariants on counterfactual recourse
    for shift in counterfactual_report["shifts"]:
        assert "category" in shift
        assert "time_spent" in shift
        assert "delta_minutes" in shift
        assert isinstance(shift["delta_minutes"], float)

    total_delta = sum(s["delta_minutes"] for s in counterfactual_report["shifts"])
    assert abs(total_delta) < 1e-4
