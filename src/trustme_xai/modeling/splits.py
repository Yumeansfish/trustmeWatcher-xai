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
    required = {"user_id", "timestamp"}
    if not required.issubset(table.columns):
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
        train_count, validation_count, _ = _split_counts(
            len(dates),
            train_ratio,
        )
        if train_count == 0:
            continue

        validation_end = train_count + validation_count
        by_date: dict[object, SplitName] = {
            date: "train" for date in dates[:train_count]
        }
        by_date.update(
            {
                date: "validation"
                for date in dates[train_count:validation_end]
            },
        )
        by_date.update(
            {date: "test" for date in dates[validation_end:]},
        )
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
    required = {"user_id", "timestamp"}
    if not required.issubset(table.columns):
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

        train_count = max(1, int(math.floor(n_dates * train_ratio)))
        val_count = max(1, int(math.floor(n_dates * validation_ratio)))

        by_date: dict[object, SplitName] = {}

        # Train dates
        train_end = min(train_count, n_dates)
        for d in dates[:train_end]:
            by_date[d] = "train"

        # Gap after train
        gap1_end = min(train_end + gap_days, n_dates)
        for d in dates[train_end:gap1_end]:
            by_date[d] = "gap"

        # Validation dates
        val_end = min(gap1_end + val_count, n_dates)
        for d in dates[gap1_end:val_end]:
            by_date[d] = "validation"

        # Gap after validation
        gap2_end = min(val_end + gap_days, n_dates)
        for d in dates[val_end:gap2_end]:
            by_date[d] = "gap"

        # Test dates
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
        num_blocks: number of cross-validation blocks (defaults to 5)
        purge_buffer_days: number of days to purge around validation blocks

    Yields:
        tuples of (train_indices, val_indices) numpy arrays
    """
    required = {"user_id", "timestamp"}
    if not required.issubset(df.columns):
        raise ValueError("df must contain user_id and timestamp columns")

    timestamps = pd.to_datetime(df["timestamp"])
    dates = timestamps.dt.date.to_numpy()
    all_indices = np.arange(len(df))

    block_assignments = np.full(len(df), -1, dtype=int)

    for _, user_rows in df.groupby("user_id", sort=True):
        user_indices = user_rows.index.to_numpy()
        user_dates = np.sort(timestamps.iloc[user_indices].dt.date.unique())

        if len(user_dates) == 0:
            continue

        date_blocks = np.array_split(user_dates, num_blocks)
        date_to_block = {}
        for block_id, d_block in enumerate(date_blocks):
            for d in d_block:
                date_to_block[d] = block_id

        for idx in user_indices:
            row_date = dates[idx]
            block_assignments[idx] = date_to_block.get(row_date, -1)

    for val_block in range(num_blocks):
        val_mask = block_assignments == val_block
        val_indices = all_indices[val_mask]

        if len(val_indices) == 0:
            continue

        val_dates = dates[val_indices]
        min_val_date = min(val_dates)
        max_val_date = max(val_dates)

        purge_start = min_val_date - pd.Timedelta(days=purge_buffer_days)
        purge_end = max_val_date + pd.Timedelta(days=purge_buffer_days)

        train_mask = (block_assignments != val_block) & (
            (dates < purge_start) | (dates > purge_end)
        )
        train_indices = all_indices[train_mask]

        yield train_indices, val_indices


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
    train_count, validation_count, _ = _split_counts(
        len(users),
        train_ratio,
    )
    if train_count == 0:
        raise ValueError("subject-out split needs at least three participants")

    rng = np.random.default_rng(random_state)
    ordered = users[rng.permutation(len(users))]
    validation_end = train_count + validation_count
    by_user: dict[str, SplitName] = {
        user: "train" for user in ordered[:train_count]
    }
    by_user.update(
        {
            user: "validation"
            for user in ordered[train_count:validation_end]
        },
    )
    by_user.update(
        {user: "test" for user in ordered[validation_end:]},
    )
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

    Returns:
        None
    """
    if not table.index.equals(row_split.index):
        raise ValueError("row_split index must match the feature table")
    labels = set(row_split.astype(str))
    allowed = {"train", "validation", "test", "excluded", "gap"}
    unknown = sorted(labels - allowed)
    if unknown:
        raise ValueError(f"split contains unknown names: {unknown}")
    active = row_split.isin(["train", "validation", "test"])
    names = set(row_split.loc[active].astype(str))
    if names != {"train", "validation", "test"}:
        raise ValueError("split must contain train validation and test rows")

    check = table.loc[active, ["user_id", "timestamp"]].copy()
    check["split"] = row_split.loc[active]
    if strategy == "within-user":
        check["date"] = pd.to_datetime(
            check["timestamp"],
            errors="raise",
        ).dt.date
        overlap = check.groupby(["user_id", "date"])["split"].nunique()
        if (overlap > 1).any():
            raise ValueError("one participant day appears in multiple splits")
        for user_id, rows in check.groupby("user_id", sort=False):
            if set(rows["split"].astype(str)) != {
                "train",
                "validation",
                "test",
            }:
                raise ValueError(f"{user_id} does not appear in every split")
            dates = {
                split: rows.loc[rows["split"] == split, "date"]
                for split in ("train", "validation", "test")
            }
            if not (
                max(dates["train"]) < min(dates["validation"])
                and max(dates["validation"]) < min(dates["test"])
            ):
                raise ValueError(f"{user_id} split order is not chronological")
        return
    if strategy == "subject-out":
        overlap = check.groupby("user_id")["split"].nunique()
        if (overlap > 1).any():
            raise ValueError("one participant appears in multiple splits")
        return
    raise ValueError(f"unknown split strategy: {strategy}")
