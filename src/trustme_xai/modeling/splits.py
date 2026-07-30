"""Split questionnaire rows without crossing users or days"""

from __future__ import annotations

from collections.abc import Iterator
import math
from typing import Literal, cast

import numpy as np
import pandas as pd

SplitName = Literal["train", "validation", "test", "excluded", "gap"]
NUM_BLOCKS = 5
PURGE_BUFFER_DAYS = 1


def _split_counts(total: int, train_ratio: float) -> tuple[int, int, int]:
    if total < 3:
        return 0, 0, 0
    train = max(1, int(math.floor(total * train_ratio)))
    if total - train < 2:
        train = total - 2
    remaining = total - train
    validation = max(1, remaining // 2)
    test = remaining - validation
    return train, validation, test


def within_user_split(
    table: pd.DataFrame,
    train_ratio: float = 0.8,
) -> pd.Series:
    """Split each participant in chronological day blocks

    Args:
        table: rows containing user_id and timestamp
        train_ratio: share of participant days used for training

    Returns:
        pd.Series with one split name per row
    """
    if not {"user_id", "timestamp"}.issubset(table.columns):
        raise ValueError("table must contain user_id and timestamp")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")

    timestamps = pd.to_datetime(table["timestamp"], errors="raise")
    work = table[["user_id"]].copy()
    work["timestamp"] = timestamps
    split = pd.Series("excluded", index=table.index, dtype="object")

    for _, rows in work.groupby("user_id", sort=True):
        ordered = rows.sort_values("timestamp", kind="stable")
        dates = list(ordered["timestamp"].dt.date.drop_duplicates())
        train_count, val_count, _ = _split_counts(len(dates), train_ratio)
        if train_count == 0:
            continue

        val_end = train_count + val_count
        by_date: dict[object, SplitName] = {d: "train" for d in dates[:train_count]}
        by_date.update({d: "validation" for d in dates[train_count:val_end]})
        by_date.update({d: "test" for d in dates[val_end:]})
        split.loc[ordered.index] = ordered["timestamp"].dt.date.map(by_date)

    return cast(pd.Series, split)


def day_purged_within_user_split(
    table: pd.DataFrame,
    train_ratio: float = 0.5,
    validation_ratio: float = 0.2,
    gap_days: int = 1,
) -> pd.Series:
    """Split participant days into chronological train, validation, and test with purging gaps

    Args:
        table: rows containing user_id and timestamp
        train_ratio: ratio of participant days used for training
        validation_ratio: ratio of participant days used for validation
        gap_days: number of gap days between splits for purging

    Returns:
        pd.Series with split name for each row
    """
    if not {"user_id", "timestamp"}.issubset(table.columns):
        raise ValueError("table must contain user_id and timestamp")

    timestamps = pd.to_datetime(table["timestamp"], errors="raise")
    work = table[["user_id"]].copy()
    work["timestamp"] = timestamps
    split = pd.Series("excluded", index=table.index, dtype="object")

    for _, rows in work.groupby("user_id", sort=True):
        ordered = rows.sort_values("timestamp", kind="stable")
        dates = list(ordered["timestamp"].dt.date.drop_duplicates())
        n_dates = len(dates)
        if n_dates == 0:
            continue

        tr_cnt = max(1, int(math.floor(n_dates * train_ratio)))
        val_cnt = max(1, int(math.floor(n_dates * validation_ratio)))

        by_date: dict[object, SplitName] = {}
        tr_end = min(tr_cnt, n_dates)
        for d in dates[:tr_end]:
            by_date[d] = "train"

        gap1_end = min(tr_end + gap_days, n_dates)
        for d in dates[tr_end:gap1_end]:
            by_date[d] = "gap"

        val_end = min(gap1_end + val_cnt, n_dates)
        for d in dates[gap1_end:val_end]:
            by_date[d] = "validation"

        gap2_end = min(val_end + gap_days, n_dates)
        for d in dates[val_end:gap2_end]:
            by_date[d] = "gap"

        for d in dates[gap2_end:]:
            by_date[d] = "test"

        split.loc[ordered.index] = ordered["timestamp"].dt.date.map(by_date)

    return cast(pd.Series, split)


def split_purged_blocks(
    df: pd.DataFrame,
    num_blocks: int = NUM_BLOCKS,
    purge_buffer_days: int = PURGE_BUFFER_DAYS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train_idx and val_idx array tuples for 5 blocks with temporal purging buffers

    Args:
        df: pd.DataFrame containing user_id and timestamp columns
        num_blocks: number of cross-validation blocks
        purge_buffer_days: number of days to purge around validation blocks

    Yields:
        tuples of (train_indices, val_indices) numpy arrays
    """
    if not {"user_id", "timestamp"}.issubset(df.columns):
        raise ValueError("df must contain user_id and timestamp columns")

    timestamps = pd.to_datetime(df["timestamp"])
    dates = timestamps.dt.date.to_numpy()
    all_indices = np.arange(len(df))
    block_assignments = np.full(len(df), -1, dtype=int)

    for _, user_rows in df.groupby("user_id", sort=True):
        u_indices = user_rows.index.to_numpy()
        u_dates = np.sort(timestamps.iloc[u_indices].dt.date.unique())
        if len(u_dates) == 0:
            continue

        d_to_block = {
            d: b_id
            for b_id, d_block in enumerate(np.array_split(u_dates, num_blocks))
            for d in d_block
        }
        for idx in u_indices:
            block_assignments[idx] = d_to_block.get(dates[idx], -1)

    for val_block in range(num_blocks):
        val_mask = block_assignments == val_block
        val_indices = all_indices[val_mask]
        if len(val_indices) == 0:
            continue

        val_dates = dates[val_indices]
        p_start = min(val_dates) - pd.Timedelta(days=purge_buffer_days)
        p_end = max(val_dates) + pd.Timedelta(days=purge_buffer_days)

        train_mask = (block_assignments != val_block) & (
            (dates < p_start) | (dates > p_end)
        )
        yield all_indices[train_mask], val_indices


def subject_out_split(
    table: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> pd.Series:
    """Split whole participants into disjoint groups

    Args:
        table: rows containing user_id
        train_ratio: share of participants used for training
        random_state: seed used to order participants

    Returns:
        pd.Series with one split name per row
    """
    if "user_id" not in table.columns:
        raise ValueError("table must contain user_id")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")

    users = np.asarray(sorted(table["user_id"].astype(str).unique()))
    tr_cnt, val_cnt, _ = _split_counts(len(users), train_ratio)
    if tr_cnt == 0:
        raise ValueError("subject-out split needs at least three participants")

    rng = np.random.default_rng(random_state)
    ordered = users[rng.permutation(len(users))]
    val_end = tr_cnt + val_cnt
    by_user: dict[str, SplitName] = {u: "train" for u in ordered[:tr_cnt]}
    by_user.update({u: "validation" for u in ordered[tr_cnt:val_end]})
    by_user.update({u: "test" for u in ordered[val_end:]})
    return cast(pd.Series, table["user_id"].astype(str).map(by_user))


def validate_split(
    table: pd.DataFrame,
    row_split: pd.Series,
    strategy: str,
) -> None:
    """Check that one split has no forbidden overlap

    Args:
        table: rows containing user_id and timestamp
        row_split: split name for each row
        strategy: within-user or subject-out
    """
    if not table.index.equals(row_split.index):
        raise ValueError("row_split index must match the feature table")

    allowed = {"train", "validation", "test", "excluded", "gap"}
    unknown = sorted(set(row_split.astype(str)) - allowed)
    if unknown:
        raise ValueError(f"split contains unknown names: {unknown}")

    active = row_split.isin(["train", "validation", "test"])
    if set(row_split.loc[active].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("split must contain train validation and test rows")

    check = table.loc[active, ["user_id", "timestamp"]].copy()
    check["split"] = row_split.loc[active]

    if strategy == "within-user":
        check["date"] = pd.to_datetime(check["timestamp"], errors="raise").dt.date
        if (check.groupby(["user_id", "date"])["split"].nunique() > 1).any():
            raise ValueError("one participant day appears in multiple splits")

        for user_id, rows in check.groupby("user_id", sort=False):
            if set(rows["split"].astype(str)) != {"train", "validation", "test"}:
                raise ValueError(f"{user_id} does not appear in every split")
            dates = {
                s: rows.loc[rows["split"] == s, "date"]
                for s in ("train", "validation", "test")
            }
            if not (max(dates["train"]) < min(dates["validation"]) < min(dates["test"])):
                raise ValueError(f"{user_id} split order is not chronological")
        return

    if strategy == "subject-out":
        if (check.groupby("user_id")["split"].nunique() > 1).any():
            raise ValueError("one participant appears in multiple splits")
        return

    raise ValueError(f"unknown split strategy: {strategy}")
