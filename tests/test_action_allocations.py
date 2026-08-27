"""Exercise Action allocation invariants through its interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustme_xai.feature_pipeline.action_allocations import (
    ACTION_ALLOCATION_COLUMNS,
    ActionAllocation,
)


def _feature_row(**overrides: float) -> pd.DataFrame:
    values = {column: 0.0 for column in ACTION_ALLOCATION_COLUMNS}
    values.update(overrides)
    return pd.DataFrame([{"user_id": "u1", "context": 7.0, **values}])


def test_action_allocation_rejects_invalid_minutes() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        ActionAllocation.from_feature_row(
            _feature_row(action_minutes_development=-1.0).iloc[0],
        )
    with pytest.raises(ValueError, match="exceed the 60-minute window"):
        ActionAllocation.from_feature_row(
            _feature_row(
                action_minutes_development=40.0,
                action_minutes_writing=21.0,
            ).iloc[0],
        )
    with pytest.raises(ValueError, match="must be finite"):
        ActionAllocation.from_feature_row(
            _feature_row(action_minutes_development=np.inf).iloc[0],
        )


def test_swap_scenarios_are_zero_sum_and_deterministic() -> None:
    baseline_row = _feature_row(action_minutes_development=10.0)
    allocation = ActionAllocation.from_feature_row(baseline_row.iloc[0])

    first = allocation.swap_scenarios(baseline_row, step_minutes=5.0)
    second = allocation.swap_scenarios(baseline_row, step_minutes=5.0)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(ACTION_ALLOCATION_COLUMNS) - 1
    assert first.loc[0, "action_minutes_development"] == 5.0
    assert first.loc[0, "action_minutes_writing"] == 5.0
    assert first.loc[1, "action_minutes_research"] == 5.0
    assert first["context"].eq(7.0).all()
    values = first.loc[:, list(ACTION_ALLOCATION_COLUMNS)].to_numpy(dtype=float)
    assert (values >= 0.0).all()
    np.testing.assert_allclose(values.sum(axis=1), 10.0)


def test_action_allocation_describes_changes_in_canonical_order() -> None:
    baseline = ActionAllocation.from_feature_row(
        _feature_row(
            action_minutes_development=10.0,
            action_minutes_personal_distraction=20.0,
        ).iloc[0],
    )
    changed = ActionAllocation.from_feature_row(
        _feature_row(
            action_minutes_development=25.0,
            action_minutes_personal_distraction=5.0,
        ).iloc[0],
    )

    assert baseline.minutes == {
        "development": 10.0,
        "writing": 0.0,
        "research": 0.0,
        "communication": 0.0,
        "media": 0.0,
        "personal_distraction": 20.0,
        "other": 0.0,
    }
    assert [
        (
            change.category,
            change.baseline_minutes,
            change.counterfactual_minutes,
            change.delta_minutes,
        )
        for change in baseline.changes_to(changed)
    ] == [
        ("development", 10.0, 25.0, 15.0),
        ("personal_distraction", 20.0, 5.0, -15.0),
    ]
