"""Tests for live PredictionReport serving and run_inference service"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from scripts.train_ensemble_models import train_dashboard_models
from trustme_xai.inference.inference_service import run_inference
from trustme_xai.inference.prediction_report import create_prediction_report


def _sample_feature_df() -> pd.DataFrame:
    """Build synthetic dataset spanning 10 distinct dates for 5-block cross-validation"""
    rows = []
    for day in range(1, 11):
        for hour in [10, 14]:
            rows.append(
                {
                    "user_id": "u1",
                    "timestamp": pd.Timestamp(f"2026-07-{day:02d} {hour:02d}:30:00"),
                    "time_development": float(day * 2.0),
                    "time_writing": 15.0,
                    "time_communication": 10.0,
                    "time_personal_distraction": 5.0,
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
                    "productivity": float((day % 5) + 1.5),
                    "engagement": float((day % 4) + 2.0),
                    "overall_wellbeing": float((day % 5) + 2.5),
                },
            )
    return pd.DataFrame(rows)


def _single_feature_row() -> pd.DataFrame:
    return _sample_feature_df().iloc[0:1]


class TestCreatePredictionReport:
    def test_report_structure(self) -> None:
        feature_df = _sample_feature_df()
        bundle = train_dashboard_models(
            feature_df,
            feature_columns=[
                "time_development",
                "time_writing",
                "time_communication",
                "time_personal_distraction",
            ],
        )

        single_row = _single_feature_row()
        report = create_prediction_report(bundle, single_row)

        assert report["participant_id"] == "u1"
        assert "2026-07-01" in report["prediction_timestamp"]
        assert len(report["predictions"]) == len(bundle.targets)

        for item in report["predictions"]:
            assert "target" in item
            assert "predicted_score" in item
            assert isinstance(item["predicted_score"], float)

    def test_multi_row_raises(self) -> None:
        feature_df = _sample_feature_df()
        bundle = train_dashboard_models(feature_df, feature_columns=["time_development"])

        two_rows = _sample_feature_df().iloc[:2]
        with pytest.raises(ValueError, match="exactly 1 row"):
            create_prediction_report(bundle, two_rows)


class TestRunInferenceService:
    def test_run_inference_with_buckets(self) -> None:
        feature_df = _sample_feature_df()
        bundle = train_dashboard_models(
            feature_df,
            feature_columns=[
                "time_development",
                "time_writing",
                "time_communication",
                "time_personal_distraction",
                "time_media",
                "time_research",
            ],
        )

        buckets = {
            "aw-watcher-window_host": {
                "type": "currentwindow",
                "events": [
                    {
                        "timestamp": "2026-07-29T14:00:00Z",
                        "duration": 1800.0,
                        "data": {"category": "Work > Programming", "app": "VS Code"},
                    },
                ],
            },
        }

        report = run_inference(
            bundle=bundle,
            activitywatch_buckets=buckets,
            user_id="u1",
            as_of=datetime(2026, 7, 29, 14, 30),
        )

        assert report["participant_id"] == "u1"
        assert len(report["predictions"]) > 0
