"""Tests for the seven-target production training pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import (
    add_self_report_features,
    all_history_columns,
)
from trustme_xai.inference.model_runtime import load_model_bundle, save_model_bundle
from trustme_xai.modeling.models import model_specs
from trustme_xai.modeling.splits import production_day_purged_within_user_split
from trustme_xai.modeling.train import train_models


def _synthetic_feature_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user_offset, user_id in enumerate(("u1", "u2")):
        for day in range(1, 13):
            for hour in (10, 14):
                base = float((day + hour + user_offset) % 7)
                row: dict[str, object] = {
                    "user_id": user_id,
                    "timestamp": pd.Timestamp(
                        f"2026-01-{day:02d} {hour:02d}:00:00"
                    ),
                    "answer_source": "streamdeck",
                }
                row.update(
                    {
                        feature: float(day + hour / 100.0 + index / 1000.0)
                        for index, feature in enumerate(PRODUCTION_FEATURE_COLUMNS)
                    }
                )
                row.update(
                    {
                        target: float((base + index / 10.0) % 6.0)
                        for index, target in enumerate(MODEL_TARGETS)
                    }
                )
                rows.append(row)
    return add_self_report_features(pd.DataFrame(rows))


def test_production_split_is_chronological_and_purged() -> None:
    table = _synthetic_feature_table()

    split = production_day_purged_within_user_split(table)

    assert set(split) == {"train", "validation", "gap", "test"}
    for user_id, rows in table.assign(split=split).groupby("user_id"):
        assert set(rows["split"]) == {"train", "validation", "gap", "test"}
        gap_days = rows.loc[rows["split"] == "gap", "timestamp"].dt.date.nunique()
        assert gap_days == 2, user_id


def test_train_models_selects_all_targets_and_round_trips(tmp_path: Path) -> None:
    table = _synthetic_feature_table()
    split = production_day_purged_within_user_split(table)

    result = train_models(
        table,
        split,
        normalization="global",
        model_specs=model_specs(["ridge_1"]),
    )

    assert result.bundle.targets == MODEL_TARGETS
    assert set(result.bundle.target_models) == set(MODEL_TARGETS)
    assert set(result.summary["selected_model"]) == {"ridge_1"}
    assert set(result.test_predictions["target"]) == set(MODEL_TARGETS)

    path = tmp_path / "current.joblib"
    save_model_bundle(result.bundle, path)
    loaded = load_model_bundle(path)
    assert loaded.targets == MODEL_TARGETS
    assert loaded.feature_columns == [
        *PRODUCTION_FEATURE_COLUMNS,
        *all_history_columns(),
    ]
