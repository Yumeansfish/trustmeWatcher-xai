"""Tests for composite score normalization and state averages"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.data.composite_scores import (
    COMPOSITE_TARGETS,
    MAX_SCORE,
    combine_state_averages,
    derive_model_target_values,
    normalize_answers,
)


def _raw_answers() -> pd.DataFrame:
    """Build a small test pd.DataFrame covering boundary values"""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            # q1_feelings: bipolar scale -3 to +3
            "q1_feelings": [-3.0, 0.0, 3.0, 1.5],
            # q2_intensity: 0 to 6
            "q2_intensity": [0.0, 3.0, 6.0, 4.0],
            # q3_tiredness: 0 to 6 (reversed for fatigue)
            "q3_tiredness": [6.0, 3.0, 0.0, 2.0],
            # q4_enthusiasm: 0 to 6
            "q4_enthusiasm": [0.0, 3.0, 6.0, 5.0],
            # q5_immersion: 0 to 6
            "q5_immersion": [0.0, 3.0, 6.0, 4.0],
            # q8_stress: 0 to 6 (reversed for stress)
            "q8_stress": [6.0, 3.0, 0.0, 1.0],
            # q9_productivity: 0 to 6
            "q9_productivity": [0.0, 3.0, 6.0, 5.0],
        },
    )


class TestNormalizeAnswers:
    def test_output_columns_present(self) -> None:
        result = normalize_answers(_raw_answers())
        expected = {
            "mood_valence",
            "arousal",
            "restfulness",
            "stress_management",
            "productivity",
        }
        assert expected.issubset(result.columns)

    def test_valence_shift(self) -> None:
        result = normalize_answers(_raw_answers())
        # -3 + 3 = 0, 0 + 3 = 3, 3 + 3 = 6
        np.testing.assert_array_almost_equal(
            result["mood_valence"].to_numpy(),
            [0.0, 3.0, 6.0, 4.5],
        )

    def test_stress_reversed(self) -> None:
        result = normalize_answers(_raw_answers())
        # 6 - 6 = 0, 6 - 3 = 3, 6 - 0 = 6, 6 - 1 = 5
        np.testing.assert_array_almost_equal(
            result["stress_management"].to_numpy(),
            [0.0, 3.0, 6.0, 5.0],
        )

    def test_fatigue_reversed(self) -> None:
        result = normalize_answers(_raw_answers())
        # 6 - 6 = 0, 6 - 3 = 3, 6 - 0 = 6, 6 - 2 = 4
        np.testing.assert_array_almost_equal(
            result["restfulness"].to_numpy(),
            [0.0, 3.0, 6.0, 4.0],
        )

    def test_all_targets_within_bounds(self) -> None:
        result = normalize_answers(_raw_answers())
        for col in [
            "mood_valence",
            "arousal",
            "restfulness",
            "stress_management",
            "productivity",
        ]:
            values = result[col].to_numpy()
            assert np.all(values >= 0.0), f"{col} has values below 0"
            assert np.all(values <= MAX_SCORE), f"{col} has values above {MAX_SCORE}"

    def test_preserves_original_columns(self) -> None:
        raw = _raw_answers()
        result = normalize_answers(raw)
        assert "user_id" in result.columns
        assert "timestamp" in result.columns
        assert "q1_feelings" in result.columns

    def test_missing_columns_raises(self) -> None:
        bad_df = pd.DataFrame({"q1_feelings": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            normalize_answers(bad_df)


class TestCombineStateAverages:
    def test_engagement_is_mean_of_q4_q5(self) -> None:
        normalized = normalize_answers(_raw_answers())
        result = combine_state_averages(normalized)
        expected = normalized[["q4_enthusiasm", "q5_immersion"]].mean(axis=1)
        np.testing.assert_array_almost_equal(
            result["engagement"].to_numpy(),
            expected.to_numpy(),
        )

    def test_wellbeing_is_mean_of_stress_engagement_valence(self) -> None:
        normalized = normalize_answers(_raw_answers())
        result = combine_state_averages(normalized)
        expected = result[
            ["stress_management", "engagement", "mood_valence"]
        ].mean(axis=1)
        np.testing.assert_array_almost_equal(
            result["overall_wellbeing"].to_numpy(),
            expected.to_numpy(),
        )

    def test_all_composite_targets_within_bounds(self) -> None:
        normalized = normalize_answers(_raw_answers())
        result = combine_state_averages(normalized)
        for col in COMPOSITE_TARGETS:
            values = result[col].to_numpy()
            assert np.all(values >= 0.0), f"{col} has values below 0"
            assert np.all(values <= MAX_SCORE), f"{col} has values above {MAX_SCORE}"

    def test_missing_engagement_columns_raises(self) -> None:
        df = pd.DataFrame(
            {"stress_management": [3.0], "mood_valence": [3.0]}
        )
        with pytest.raises(ValueError, match="missing engagement columns"):
            combine_state_averages(df)

    def test_boundary_values(self) -> None:
        """All-zero and all-max inputs produce scores at the boundaries"""
        normalized = normalize_answers(_raw_answers())
        result = combine_state_averages(normalized)
        # row 0 has all-zero normalized targets
        assert result["engagement"].iloc[0] == pytest.approx(0.0)
        assert result["overall_wellbeing"].iloc[0] == pytest.approx(0.0)
        # row 2 has all-max normalized targets
        assert result["engagement"].iloc[2] == pytest.approx(6.0)
        assert result["overall_wellbeing"].iloc[2] == pytest.approx(6.0)


def test_derive_one_runtime_target_mapping() -> None:
    values = derive_model_target_values(
        {
            "q1_feelings": -1,
            "q2_intensity": 4,
            "q3_tiredness": 5,
            "q4_enthusiasm": 2,
            "q5_immersion": 4,
            "q8_stress": 1,
            "q9_productivity": 5,
        }
    )

    assert values == {
        "mood_valence": 2.0,
        "arousal": 4.0,
        "restfulness": 1.0,
        "stress_management": 5.0,
        "productivity": 5.0,
        "engagement": 3.0,
        "overall_wellbeing": pytest.approx(10.0 / 3.0),
    }
