"""Tests for physical simplex constraints and feature taxonomy rules"""

from __future__ import annotations

import math

import pytest

from trustme_xai.inference.counterfactual_constraints import (
    check_time_budget,
    is_immutable,
    validate_counterfactual,
)


class TestCounterfactualConstraints:
    def test_immutable_feature_identification(self) -> None:
        assert is_immutable("timestamp")
        assert is_immutable("hour_of_day")
        assert is_immutable("sleep_hours")
        assert not is_immutable("time_development")
        assert not is_immutable("time_personal_distraction")

    def test_check_time_budget_zero_sum_valid(self) -> None:
        row = {"time_personal_distraction": 30.0, "time_development": 10.0}
        deltas = {"time_personal_distraction": -15.0, "time_development": 15.0}
        assert check_time_budget(row, deltas)

    def test_check_time_budget_non_zero_sum_invalid(self) -> None:
        row = {"time_personal_distraction": 30.0, "time_development": 10.0}
        deltas = {"time_personal_distraction": -15.0, "time_development": 20.0}
        assert not check_time_budget(row, deltas)

    def test_check_time_budget_negative_duration_invalid(self) -> None:
        row = {"time_personal_distraction": 10.0, "time_development": 10.0}
        deltas = {"time_personal_distraction": -15.0, "time_development": 15.0}
        assert not check_time_budget(row, deltas)

    def test_validate_counterfactual_immutable_modified_raises(self) -> None:
        original = {
            "user_id": "u1",
            "timestamp": "2026-07-29T14:30:00",
            "hour_of_day": 14.5,
        }
        modified = {
            "user_id": "u1",
            "timestamp": "2026-07-29T14:30:00",
            "hour_of_day": 15.5,
        }
        with pytest.raises(ValueError, match="immutable feature modified"):
            validate_counterfactual(original, modified)

    def test_validate_counterfactual_zero_sum_violated_raises(self) -> None:
        original = {"time_personal_distraction": 20.0, "time_development": 10.0}
        modified = {"time_personal_distraction": 10.0, "time_development": 30.0}
        with pytest.raises(
            ValueError,
            match="zero-sum conservation invariant violated",
        ):
            validate_counterfactual(original, modified)

    def test_missing_immutable_value_is_not_treated_as_modified(self) -> None:
        original = {
            "user_id": "u1",
            "timestamp": "2026-07-29T14:30:00",
            "minutes_since_prev1_window": math.nan,
            "time_personal_distraction": 10.0,
            "time_development": 10.0,
        }
        modified = {
            **original,
            "minutes_since_prev1_window": math.nan,
            "time_personal_distraction": 5.0,
            "time_development": 15.0,
        }

        validate_counterfactual(original, modified)
