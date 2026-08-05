"""Build hourly behavior state features"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from typing import Any, cast

import numpy as np
import pandas as pd

from trustme_xai.contracts import ParsedActivityWatchEvents

STATE_CATEGORIES = [
    "development",
    "writing",
    "communication",
    "research",
    "media",
    "personal_distraction",
    "meetings",
    "ai_assistant",
    "browser_uncategorized",
    "file_management",
    "other",
]

BEHAVIOR_STATE_SCHEMA = "trustme_xai.behavior_state_model"
PRODUCTION_BEHAVIOR_STATE_RESOURCE = "behavior_state_model.json"
PRODUCTION_STATE_NAMES = [
    "media-heavy",
    "mixed high-activity work",
    "development-heavy",
    "mixed browser/writing/research",
    "file-management-heavy",
    "uncategorized/other-heavy",
]


@dataclass(frozen=True)
class BehaviorStateModel:
    """Store the saved behavior state model"""

    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray
    state_names: list[str]

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).copy()
        scale = np.asarray(self.scale, dtype=float).copy()
        centers = np.asarray(self.centers, dtype=float).copy()
        feature_count = len(self.feature_columns)
        if mean.shape != (feature_count,) or scale.shape != (feature_count,):
            raise ValueError("behavior state scaler shape does not match its features")
        if centers.ndim != 2 or centers.shape[1] != feature_count:
            raise ValueError("behavior state center shape does not match its features")
        if len(self.state_names) != centers.shape[0]:
            raise ValueError("behavior state name count does not match its centers")

        mean.setflags(write=False)
        scale.setflags(write=False)
        centers.setflags(write=False)
        object.__setattr__(self, "feature_columns", list(self.feature_columns))
        object.__setattr__(self, "state_names", list(self.state_names))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "centers", centers)

    @property
    def n_states(self) -> int:
        """Return the number of behavior states

        Returns:
            number of behavior states
        """

        return int(self.centers.shape[0])

    def scale_values(self, values: np.ndarray) -> np.ndarray:
        """Scale hourly feature values

        Args:
            values: hourly feature values

        Returns:
            scaled np.ndarray
        """

        numeric = np.asarray(values, dtype=float)
        if numeric.ndim != 2 or numeric.shape[1] != len(self.feature_columns):
            raise ValueError("behavior state values have the wrong shape")
        return (numeric - self.mean) / self.scale

    def distances(self, values: np.ndarray) -> np.ndarray:
        """Calculate distance from each value to every state

        Args:
            values: hourly feature values

        Returns:
            np.ndarray containing state distances
        """

        scaled = self.scale_values(values)
        deltas = scaled[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        return np.sqrt(np.sum(np.square(deltas), axis=2))

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Find the nearest state for each value

        Args:
            values: hourly feature values

        Returns:
            np.ndarray containing state ids
        """

        return np.argmin(self.distances(values), axis=1)


def _model_from_payload(payload: Any) -> BehaviorStateModel:
    if payload["schema"] != BEHAVIOR_STATE_SCHEMA:
        raise ValueError("behavior state artifact schema is not supported")

    model = BehaviorStateModel(
        feature_columns=payload["feature_columns"],
        mean=payload["scaler"]["mean"],
        scale=payload["scaler"]["scale"],
        centers=payload["cluster_centers"],
        state_names=payload["state_names"],
    )
    if model.n_states != payload["n_states"]:
        raise ValueError("behavior state artifact n_states does not match its centers")
    return model


def load_production_behavior_state_model() -> BehaviorStateModel:
    """Load the saved production behavior-state model.

    Returns:
        saved production behavior state model
    """

    artifact = resources.files("trustme_xai.feature_pipeline").joinpath(
        PRODUCTION_BEHAVIOR_STATE_RESOURCE,
    )
    with artifact.open(encoding="utf-8") as handle:
        model = _model_from_payload(json.load(handle))

    if model.feature_columns != state_feature_columns():
        raise ValueError("bundled behavior state feature order is not supported")
    if model.state_names != PRODUCTION_STATE_NAMES:
        raise ValueError("bundled behavior state names are not supported")
    return model


def clipped_hour_records(events: pd.DataFrame) -> pd.DataFrame:
    """Split events across hourly boundaries

    Args:
        events: parsed window or web events

    Returns:
        pd.DataFrame with one row per event hour
    """

    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(
            columns=["user_id", "hour", "category", "minutes", "event_count"]
        )

    for row in events[["user_id", "timestamp", "event_end", "category"]].itertuples(
        index=False
    ):
        start = cast(pd.Timestamp, getattr(row, "timestamp"))
        end = cast(pd.Timestamp, getattr(row, "event_end"))
        hour = start.floor("h")
        while hour < end:
            next_hour = hour + pd.Timedelta(hours=1)
            clip_start = max(start, hour)
            clip_end = min(end, next_hour)
            minutes = (clip_end - clip_start).total_seconds() / 60.0
            if minutes > 0:
                rows.append(
                    {
                        "user_id": str(getattr(row, "user_id")),
                        "hour": hour,
                        "category": str(getattr(row, "category")),
                        "minutes": minutes,
                        "event_count": 1.0,
                    },
                )
            hour = next_hour
    return pd.DataFrame(rows)


def build_hourly_behavior_table(events: ParsedActivityWatchEvents) -> pd.DataFrame:
    """Build hourly category features

    Args:
        events: parsed activitywatch tables

    Returns:
        pd.DataFrame with one row per user hour
    """

    records = pd.concat(
        [clipped_hour_records(events.window), clipped_hour_records(events.web)],
        ignore_index=True,
    )
    if records.empty:
        return pd.DataFrame(columns=["user_id", "hour", *STATE_CATEGORIES])

    grouped = records.groupby(["user_id", "hour", "category"], as_index=False).agg(
        minutes=("minutes", "sum"), event_count=("event_count", "sum")
    )
    pivot = grouped.pivot_table(
        index=["user_id", "hour"],
        columns="category",
        values="minutes",
        aggfunc="sum",
        fill_value=0.0,
    )
    for category in STATE_CATEGORIES:
        if category not in pivot.columns:
            pivot[category] = 0.0
    hourly = pivot[STATE_CATEGORIES].reset_index()
    event_counts = cast(
        pd.DataFrame,
        grouped.groupby(["user_id", "hour"], as_index=False)["event_count"].sum(),
    )
    hourly = hourly.merge(event_counts, on=["user_id", "hour"], how="left")

    category_values = hourly[STATE_CATEGORIES].to_numpy(float)
    active_minutes = np.minimum(category_values.sum(axis=1), 60.0)
    hourly["active_minutes"] = active_minutes
    hourly["active_ratio"] = active_minutes / 60.0
    hourly["category_count"] = (category_values > 0).sum(axis=1)
    totals = np.maximum(category_values.sum(axis=1, keepdims=True), 1e-9)
    probabilities = category_values / totals
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log2(probabilities[positive])
    hourly["category_entropy"] = -np.sum(probabilities * log_probabilities, axis=1)
    hour_of_day = pd.to_datetime(hourly["hour"]).dt.hour.to_numpy(float)
    hourly["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    hourly["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)
    totals_flat = np.maximum(category_values.sum(axis=1), 1e-9)
    for category in STATE_CATEGORIES:
        hourly[f"ratio_{category}"] = hourly[category] / totals_flat
    return hourly.sort_values(["user_id", "hour"]).reset_index(drop=True)


def state_feature_columns() -> list[str]:
    """Return ordered state model columns

    Returns:
        list of state model columns
    """

    return [
        "active_minutes",
        "active_ratio",
        "event_count",
        "category_count",
        "category_entropy",
        "hour_sin",
        "hour_cos",
        *[f"ratio_{category}" for category in STATE_CATEGORIES],
    ]


def label_hourly_states(
    hourly: pd.DataFrame, model: BehaviorStateModel
) -> pd.DataFrame:
    """Assign each hour to its nearest state

    Args:
        hourly: hourly behavior table
        model: behavior state model

    Returns:
        pd.DataFrame with state labels and distances
    """

    if hourly.empty:
        return pd.DataFrame(
            columns=["user_id", "hour", "active_minutes", "state", "state_distance_min"]
        )

    values = hourly[model.feature_columns].astype(float).fillna(0.0).to_numpy()
    distances = model.distances(values)
    labeled = hourly[["user_id", "hour", "active_minutes"]].copy()
    labeled["state"] = np.argmin(distances, axis=1)
    labeled["state_distance_min"] = distances.min(axis=1)
    return labeled


def interval_state_features(
    user_hours: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prefix: str,
    n_states: int,
) -> dict[str, float]:
    """Summarize behavior states over one interval

    Args:
        user_hours: hourly state rows for one user
        start: interval start time
        end: interval end time
        prefix: output column prefix
        n_states: number of behavior states

    Returns:
        dictionary containing interval state features
    """

    hours = user_hours[
        (user_hours["hour"] < end) & (user_hours["hour"] >= start.floor("h"))
    ]
    active = hours[hours["active_minutes"] > 0].copy()
    out = {
        f"{prefix}_state_active_hours": float(len(active)),
        f"{prefix}_state_distance_median": 0.0,
        f"{prefix}_state_dominant_share": 0.0,
        f"{prefix}_state_entropy": 0.0,
        f"{prefix}_state_transition_count": 0.0,
    }
    for state in range(n_states):
        out[f"{prefix}_state_share_{state}"] = 0.0
    if active.empty:
        return out

    counts = active["state"].value_counts().sort_index()
    shares = counts / float(counts.sum())
    state_numbers = shares.index.to_numpy(dtype=int)
    for state, share in zip(state_numbers, shares.to_numpy(dtype=float)):
        out[f"{prefix}_state_share_{state}"] = float(share)
    share_values = shares.to_numpy(float)
    sequence = active.sort_values("hour")["state"].to_numpy()
    out[f"{prefix}_state_dominant_share"] = float(shares.max())
    out[f"{prefix}_state_entropy"] = float(
        -sum(value * math.log2(value) for value in share_values if value > 0)
    )
    out[f"{prefix}_state_distance_median"] = float(
        cast(Any, active["state_distance_min"].median())
    )
    out[f"{prefix}_state_transition_count"] = float(
        np.sum(sequence[1:] != sequence[:-1])
    )
    return out


def add_behavior_state_features(
    feature_table: pd.DataFrame,
    hourly: pd.DataFrame,
    model: BehaviorStateModel,
    current_window_minutes: int = 60,
    events: ParsedActivityWatchEvents | None = None,
) -> pd.DataFrame:
    """Add current and prior state features to answer rows

    Args:
        feature_table: answer feature rows
        hourly: hourly behavior table
        model: behavior state model
        current_window_minutes: minutes in the current answer window
        events: optional events used to clip the current hour

    Returns:
        pd.DataFrame with added behavior state features
    """

    table = feature_table.copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    labeled_hourly = label_hourly_states(hourly, model)
    by_user = (
        dict(tuple(labeled_hourly.groupby("user_id")))
        if not labeled_hourly.empty
        else {}
    )
    empty = pd.DataFrame(columns=labeled_hourly.columns)
    event_groups: dict[str, dict[str, pd.DataFrame]] = {}
    if events is not None:
        event_groups = {
            name: (
                dict(tuple(table.groupby("user_id")))
                if not table.empty
                else {}
            )
            for name, table in (("window", events.window), ("web", events.web))
        }
    rows: list[dict[str, float]] = []

    for item in table.itertuples(index=False):
        user_id = str(getattr(item, "user_id"))
        raw_answer_time = pd.Timestamp(getattr(item, "timestamp"))
        if pd.isna(raw_answer_time):
            raise ValueError("feature_table contains an invalid timestamp")
        answer_time = raw_answer_time
        current_start = answer_time - pd.Timedelta(minutes=current_window_minutes)
        user_hours = by_user.get(user_id, empty)
        if events is not None:
            hour_start = answer_time.floor("h")

            def clip_current(name: str) -> pd.DataFrame:
                source = event_groups[name].get(user_id)
                if source is None or source.empty:
                    return pd.DataFrame(columns=getattr(events, name).columns)
                selected = source.loc[
                    (source["timestamp"] < answer_time)
                    & (source["event_end"] > hour_start)
                ].copy()
                selected["event_end"] = selected["event_end"].clip(
                    upper=answer_time,
                )
                return selected.loc[
                    selected["event_end"] > selected["timestamp"]
                ]

            partial_events = ParsedActivityWatchEvents(
                window=clip_current("window"),
                web=clip_current("web"),
                input=pd.DataFrame(),
            )
            partial = label_hourly_states(
                build_hourly_behavior_table(partial_events),
                model,
            )
            user_hours = pd.concat(
                [
                    user_hours.loc[user_hours["hour"] != hour_start],
                    partial,
                ],
                ignore_index=True,
            )
        row: dict[str, float] = {}
        row.update(
            interval_state_features(
                user_hours, current_start, answer_time, "current60", model.n_states
            )
        )
        row.update(
            interval_state_features(
                user_hours,
                current_start - pd.Timedelta(hours=24),
                current_start,
                "prior24h",
                model.n_states,
            ),
        )
        row.update(
            interval_state_features(
                user_hours,
                current_start - pd.Timedelta(days=7),
                current_start,
                "prior7d",
                model.n_states,
            ),
        )
        rows.append(row)

    return pd.concat([table, pd.DataFrame(rows, index=table.index)], axis=1)
