"""Binarize continuous psychological targets using hard and participant median thresholds"""

from __future__ import annotations

import pandas as pd

HARD_THRESHOLD = 3.0

_TARGET_CANDIDATES = [
    "stress",
    "fatigue",
    "valence",
    "arousal",
    "productivity",
    "engagement",
    "overall_wellbeing",
]


def _detect_targets(df: pd.DataFrame) -> list[str]:
    """Find target columns present in the input table"""
    return [col for col in _TARGET_CANDIDATES if col in df.columns]


def compute_user_medians(train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate median score per user for each target metric on training data

    Args:
        train_df: training split pd.DataFrame with user_id and target columns

    Returns:
        pd.DataFrame with columns user_id, target, median
    """
    if "user_id" not in train_df.columns:
        raise ValueError("train_df must contain user_id column")

    targets = _detect_targets(train_df)
    if not targets:
        raise ValueError("train_df must contain at least one valid target column")

    records: list[dict[str, str | float]] = []
    for user_id, user_rows in train_df.groupby("user_id", sort=True):
        for target in targets:
            median_val = float(user_rows[target].median())
            records.append(
                {
                    "user_id": str(user_id),
                    "target": target,
                    "median": median_val,
                },
            )

    return pd.DataFrame(records, columns=["user_id", "target", "median"])


def binarize_targets(
    df: pd.DataFrame,
    medians_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary label columns using hard 3.0 and per-user median thresholds

    Adds target_bin_hard (1 if score >= 3.0, else 0) and target_bin_median
    (1 if score >= user median, else 0).

    Args:
        df: target pd.DataFrame containing user_id and target columns
        medians_df: baseline thresholds pd.DataFrame from compute_user_medians

    Returns:
        pd.DataFrame with binary target columns added
    """
    if "user_id" not in df.columns:
        raise ValueError("df must contain user_id column")

    required_medians_cols = {"user_id", "target", "median"}
    if not required_medians_cols.issubset(medians_df.columns):
        raise ValueError(
            f"medians_df must contain columns: {sorted(required_medians_cols)}",
        )

    targets = _detect_targets(df)
    if not targets:
        raise ValueError("df must contain at least one valid target column")

    result = df.copy()

    # Calculate global target medians as fallbacks for new users
    global_medians = {
        target: float(medians_df.loc[medians_df["target"] == target, "median"].median())
        if not medians_df.loc[medians_df["target"] == target].empty
        else HARD_THRESHOLD
        for target in targets
    }

    # Map per-user medians into lookup dictionary
    user_medians: dict[tuple[str, str], float] = {
        (str(row["user_id"]), str(row["target"])): float(row["median"])
        for _, row in medians_df.iterrows()
    }

    for target in targets:
        scores = result[target].to_numpy()

        # Hard threshold binarization (>= 3.0)
        result[f"{target}_bin_hard"] = (scores >= HARD_THRESHOLD).astype(int)

        # Participant median threshold binarization
        medians = [
            user_medians.get((str(uid), target), global_medians[target])
            for uid in result["user_id"]
        ]

        result[f"{target}_bin_median"] = (scores >= medians).astype(int)

    return result
