"""Split questionnaire rows without crossing users or days"""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import pandas as pd

SplitName = Literal["train", "validation", "test", "excluded", "gap"]


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


def production_day_purged_within_user_split(
    table: pd.DataFrame,
) -> pd.Series:
    """Build the locked chronological production split

    The split targets 80 percent training rows. It targets 10 percent
    validation rows. One whole participant day is purged at each boundary.

    Args:
        table: rows with user_id and timestamp

    Returns:
        pd.Series with one split name per row
    """
    if not {"user_id", "timestamp"}.issubset(table.columns):
        raise ValueError("table must contain user_id and timestamp")

    work = table[["user_id"]].copy()
    work["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    split = pd.Series("excluded", index=table.index, dtype="object")
    target_shares = np.asarray([0.8, 0.1, 0.1], dtype=float)

    for user_id, rows in work.groupby("user_id", sort=True):
        ordered = rows.sort_values("timestamp", kind="stable")
        dates = list(ordered["timestamp"].dt.date.drop_duplicates())
        if len(dates) < 5:
            raise ValueError(f"{user_id} needs at least five questionnaire days")
        rows_per_day = ordered["timestamp"].dt.date.value_counts()
        choices: list[tuple[float, int, int]] = []
        for train_end in range(1, len(dates) - 3):
            for second_gap in range(train_end + 2, len(dates) - 1):
                counts = np.asarray(
                    [
                        sum(rows_per_day[day] for day in dates[:train_end]),
                        sum(
                            rows_per_day[day]
                            for day in dates[train_end + 1 : second_gap]
                        ),
                        sum(rows_per_day[day] for day in dates[second_gap + 1 :]),
                    ],
                    dtype=float,
                )
                shares = counts / counts.sum()
                score = float(np.square(shares - target_shares).sum())
                choices.append((score, train_end, second_gap))

        _, train_end, second_gap = min(choices)
        names: dict[object, SplitName] = {
            **{day: "train" for day in dates[:train_end]},
            dates[train_end]: "gap",
            **{
                day: "validation"
                for day in dates[train_end + 1 : second_gap]
            },
            dates[second_gap]: "gap",
            **{day: "test" for day in dates[second_gap + 1 :]},
        }
        split.loc[ordered.index] = ordered["timestamp"].dt.date.map(names)

    result = cast(pd.Series, split)
    validate_split(table, result, "within-user")
    gap_days = (
        work.assign(split=result, date=work["timestamp"].dt.date)
        .loc[lambda frame: frame["split"] == "gap"]
        .groupby("user_id")["date"]
        .nunique()
    )
    if not bool(gap_days.eq(2).all()):
        raise ValueError("each participant must have two purge days")
    return result


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
            ordered = (
                max(dates["train"])
                < min(dates["validation"])
                < min(dates["test"])
            )
            if not ordered:
                raise ValueError(f"{user_id} split order is not chronological")
        return

    if strategy == "subject-out":
        if (check.groupby("user_id")["split"].nunique() > 1).any():
            raise ValueError("one participant appears in multiple splits")
        return

    raise ValueError(f"unknown split strategy: {strategy}")
