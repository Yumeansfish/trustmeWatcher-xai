"""Tests for Zero-Sum Swap search engine and counterfactual production service"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from scripts.train_ensemble_models import train_dashboard_models
from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.counterfactual_service import run_counterfactual
from trustme_xai.inference.ensemble_bundle import TargetMetric


def _synthetic_training_data() -> pd.DataFrame:
    rows = []
    for day in range(1, 11):
        for hour in [10, 14]:
            dev = float(day * 3.0)
            distraction = float(30.0 - dev * 0.5)
            rows.append(
                {
                    "user_id": "u1",
                    "timestamp": pd.Timestamp(f"2026-07-{day:02d} {hour:02d}:30:00"),
                    "time_development": dev,
                    "time_writing": 5.0,
                    "time_communication": 5.0,
                    "time_personal_distraction": max(0.0, distraction),
                    "time_media": 0.0,
                    "time_research": 0.0,
                    "hour_of_day": hour + 0.5,
                    "is_morning": int(hour < 12),
                    "is_afternoon": int(hour >= 12),
                    "sleep_hours": 7.5,
                    "minutes_since_prev1_window": 60.0,
                    "stress": float((day % 4) + 1.0),
                    "fatigue": float((day % 3) + 1.0),
                    "valence": float((day % 5) + 1.0),
                    "arousal": float((day % 3) + 2.0),
                    "productivity": float(1.0 + dev * 0.1),
                    "engagement": float((day % 4) + 2.0),
                    "overall_wellbeing": float((day % 5) + 2.5),
                },
            )
    return pd.DataFrame(rows)


class TestCounterfactualEngine:
    def test_find_counterfactual_zero_sum_preservation(self) -> None:
        train_df = _synthetic_training_data()
        feature_cols = [
            "time_development",
            "time_writing",
            "time_communication",
            "time_personal_distraction",
            "time_media",
            "time_research",
        ]
        bundle = train_dashboard_models(train_df, feature_columns=feature_cols)

        # Baseline row with high distraction and low development
        base_row = pd.DataFrame(
            [
                {
                    "user_id": "u1",
                    "timestamp": pd.Timestamp("2026-07-01 10:30:00"),
                    "time_development": 5.0,
                    "time_writing": 5.0,
                    "time_communication": 5.0,
                    "time_personal_distraction": 25.0,
                    "time_media": 0.0,
                    "time_research": 0.0,
                    "hour_of_day": 10.5,
                    "is_morning": 1,
                    "is_afternoon": 0,
                    "sleep_hours": 7.5,
                    "minutes_since_prev1_window": 60.0,
                },
            ],
        )

        res = find_counterfactual(
            bundle=bundle,
            feature_row=base_row,
            target="productivity",
            desired_score=3.5,
        )

        assert res["participant_id"] == "u1"
        assert res["target"] == "productivity"
        assert isinstance(res["success"], bool)

        # Assert zero-sum conservation invariant on generated shifts
        total_shift_delta = sum(shift["delta_minutes"] for shift in res["shifts"])
        assert abs(total_shift_delta) < 1e-4

        # Assert maximum 3 shifts (sparsity constraint)
        assert len(res["shifts"]) <= 3

    def test_run_counterfactual_service(self) -> None:
        train_df = _synthetic_training_data()
        feature_cols = [
            "time_development",
            "time_writing",
            "time_communication",
            "time_personal_distraction",
            "time_media",
            "time_research",
        ]
        bundle = train_dashboard_models(train_df, feature_columns=feature_cols)

        buckets = {
            "aw-watcher-window_host": {
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2026-07-29T14:00:00Z",
                        "duration": 1500.0,
                        "data": {"category": "Social Media"},
                    },
                ],
            },
        }

        res = run_counterfactual(
            bundle=bundle,
            buckets=buckets,
            user_id="u1",
            timestamp=datetime(2026, 7, 29, 14, 30),
            target="productivity",
            desired_score=4.0,
        )

        assert res["participant_id"] == "u1"
        assert "productivity" in res["target"]
