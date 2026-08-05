"""Tests for target binarization and per-user medians calculation"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.data.binarization import (
    binarize_targets,
    compute_user_medians,
)


def _sample_train_data() -> pd.DataFrame:
    """Build a synthetic training dataset with two users"""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "productivity": [1.0, 3.0, 5.0, 2.0, 4.0, 6.0],  # u1 med=3.0, u2 med=4.0
            "stress_management": [5.0, 4.0, 3.0, 1.0, 2.0, 3.0],
        },
    )


def _sample_val_data() -> pd.DataFrame:
    """Build a validation dataset containing known users and one new user"""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3_new"],
            "productivity": [2.5, 4.5, 1.0],
            "stress_management": [3.5, 2.5, 5.0],
        },
    )


class TestComputeUserMedians:
    def test_output_schema(self) -> None:
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)
        assert list(medians_df.columns) == ["user_id", "target", "median"]
        assert len(medians_df) == 4  # 2 users * 2 targets

    def test_calculated_values(self) -> None:
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)

        u1_prod = medians_df.loc[
            (medians_df["user_id"] == "u1") & (medians_df["target"] == "productivity"),
            "median",
        ].iloc[0]
        assert u1_prod == pytest.approx(3.0)

        u2_prod = medians_df.loc[
            (medians_df["user_id"] == "u2") & (medians_df["target"] == "productivity"),
            "median",
        ].iloc[0]
        assert u2_prod == pytest.approx(4.0)

    def test_missing_user_id_raises(self) -> None:
        bad_df = pd.DataFrame({"productivity": [1.0, 2.0]})
        with pytest.raises(ValueError, match="user_id"):
            compute_user_medians(bad_df)

    def test_missing_targets_raises(self) -> None:
        bad_df = pd.DataFrame({"user_id": ["u1"], "other_col": [10.0]})
        with pytest.raises(ValueError, match="target column"):
            compute_user_medians(bad_df)


class TestBinarizeTargets:
    def test_hard_threshold_binarization(self) -> None:
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)
        val_df = _sample_val_data()

        result = binarize_targets(val_df, medians_df)
        assert "productivity_bin_hard" in result.columns
        assert "stress_management_bin_hard" in result.columns

        # productivity >= 3.0: u1 (2.5 -> 0), u2 (4.5 -> 1), u3 (1.0 -> 0)
        np.testing.assert_array_equal(
            result["productivity_bin_hard"].to_numpy(),
            [0, 1, 0],
        )

    def test_median_threshold_binarization(self) -> None:
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)
        val_df = _sample_val_data()

        result = binarize_targets(val_df, medians_df)
        assert "productivity_bin_median" in result.columns
        assert "stress_management_bin_median" in result.columns

        # u1 prod=2.5 vs u1_med=3.0 -> 0
        # u2 prod=4.5 vs u2_med=4.0 -> 1
        np.testing.assert_array_equal(
            result["productivity_bin_median"].iloc[:2].to_numpy(),
            [0, 1],
        )

    def test_new_user_fallback(self) -> None:
        """Unseen users in test set fall back gracefully to global training medians"""
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)
        val_df = _sample_val_data()

        result = binarize_targets(val_df, medians_df)
        # u3_new has stress=5.0. Global median for stress is median(4.0, 2.0) = 3.0
        # 5.0 >= 3.0 -> 1
        assert result["stress_management_bin_median"].iloc[2] == 1

    def test_no_data_leakage_on_validation_splits(self) -> None:
        """Verifies thresholds are strictly derived from medians_df (training data)"""
        train_df = _sample_train_data()
        medians_df = compute_user_medians(train_df)

        # Validation set with high numbers that would change medians if recalculated
        val_df = pd.DataFrame(
            {
                "user_id": ["u1", "u1"],
                "productivity": [10.0, 20.0],
            },
        )
        result = binarize_targets(val_df, medians_df)
        # u1 medians_df productivity is 3.0. Both 10.0 and 20.0 >= 3.0 -> [1, 1]
        np.testing.assert_array_equal(
            result["productivity_bin_median"].to_numpy(),
            [1, 1],
        )
