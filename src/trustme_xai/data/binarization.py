"""Binarize continuous targets with fixed and participant thresholds"""

from __future__ import annotations

import pandas as pd

HARD_THRESHOLD = 3.0

_TARGET_CANDIDATES = (
    "stress",
    "fatigue",
    "valence",
    "arousal",
    "productivity",
    "engagement",
    "overall_wellbeing",
)


def _detect_targets(df: pd.DataFrame) -> list[str]:
    """Find target columns present in input table

    Args:
        df: input pd.DataFrame

    Returns:
        list of target column names
    """
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

    melted = train_df.melt(
        id_vars=["user_id"],
        value_vars=targets,
        var_name="target",
        value_name="score",
    )
    res = (
        melted.groupby(["user_id", "target"], as_index=False)["score"]
        .median()
        .rename(columns={"score": "median"})
    )
    res["user_id"] = res["user_id"].astype(str)
    return res


def binarize_targets(
    df: pd.DataFrame,
    medians_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary label columns using hard 3.0 and per-user median thresholds

    Args:
        df: target pd.DataFrame containing user_id and target columns
        medians_df: baseline thresholds pd.DataFrame from compute_user_medians

    Returns:
        pd.DataFrame with binary target columns added
    """
    if "user_id" not in df.columns:
        raise ValueError("df must contain user_id column")

    required = {"user_id", "target", "median"}
    if not required.issubset(medians_df.columns):
        raise ValueError(f"medians_df must contain columns: {sorted(required)}")

    targets = _detect_targets(df)
    if not targets:
        raise ValueError("df must contain at least one valid target column")

    res = df.copy()
    user_map = {
        (str(row["user_id"]), str(row["target"])): float(row["median"])
        for _, row in medians_df.iterrows()
    }
    glob_map = {
        t: float(medians_df.loc[medians_df["target"] == t, "median"].median())
        if not medians_df.loc[medians_df["target"] == t].empty
        else HARD_THRESHOLD
        for t in targets
    }

    for t in targets:
        scores = res[t].to_numpy()
        res[f"{t}_bin_hard"] = (scores >= HARD_THRESHOLD).astype(int)
        meds = [user_map.get((str(u), t), glob_map[t]) for u in res["user_id"]]
        res[f"{t}_bin_median"] = (scores >= meds).astype(int)

    return res
