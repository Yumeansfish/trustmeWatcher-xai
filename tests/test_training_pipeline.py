"""Tests for purged 5-block model training pipeline and EnsembleBundle artifact"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_ensemble_models import (
    CandidateEvaluation,
    create_candidate_model,
    evaluate_candidate,
    fit_and_score_block,
    train_dashboard_models,
)
from trustme_xai.inference.ensemble_bundle import (
    EnsembleBundle,
    EnsembleTargetModel,
    ModelFamily,
    NUM_BLOCKS,
    TargetMetric,
)
from trustme_xai.inference.model_runtime import load_model_bundle, save_model_bundle
from trustme_xai.modeling.splits import split_purged_blocks


def _synthetic_feature_table() -> pd.DataFrame:
    """Build a synthetic feature table spanning 10 days for two users"""
    rows = []
    for uid in ["u1", "u2"]:
        for day in range(1, 11):
            for hour in [10, 14]:
                rows.append(
                    {
                        "user_id": uid,
                        "timestamp": pd.Timestamp(f"2026-01-{day:02d} {hour:02d}:00:00"),
                        "feat_a": float(day * 0.5 + (1.0 if uid == "u1" else 2.0)),
                        "feat_b": float(hour / 10.0),
                        "stress": float((day % 5) + 1.0),
                        "fatigue": float((day % 4) + 1.0),
                        "valence": float((day % 6) + 0.5),
                        "arousal": float((day % 3) + 2.0),
                        "productivity": float((day % 5) + 1.5),
                        "engagement": float((day % 4) + 2.0),
                        "overall_wellbeing": float((day % 5) + 2.5),
                    },
                )
    return pd.DataFrame(rows)


class TestSplitPurgedBlocks:
    def test_yields_five_blocks(self) -> None:
        df = _synthetic_feature_table()
        blocks = list(split_purged_blocks(df))
        assert len(blocks) == NUM_BLOCKS

    def test_indices_are_disjoint_per_block(self) -> None:
        df = _synthetic_feature_table()
        for train_idx, val_idx in split_purged_blocks(df):
            overlap = set(train_idx).intersection(set(val_idx))
            assert len(overlap) == 0, f"Found overlap between train and val indices: {overlap}"

    def test_missing_required_columns_raises(self) -> None:
        bad_df = pd.DataFrame({"user_id": ["u1"]})
        with pytest.raises(ValueError, match="user_id and timestamp"):
            list(split_purged_blocks(bad_df))


class TestModelFactoryAndScoring:
    def test_create_candidate_model_valid_families(self) -> None:
        for family in ModelFamily:
            model = create_candidate_model(family)
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")

    def test_fit_and_score_block(self) -> None:
        df = _synthetic_feature_table()
        train_block = df.iloc[:20]
        val_block = df.iloc[20:]
        feature_cols = ["feat_a", "feat_b"]

        model, mse = fit_and_score_block(
            ModelFamily.RIDGE,
            train_block,
            val_block,
            TargetMetric.STRESS,
            feature_cols,
        )
        assert hasattr(model, "predict")
        assert mse >= 0.0

    def test_evaluate_candidate(self) -> None:
        df = _synthetic_feature_table()
        feature_cols = ["feat_a", "feat_b"]

        eval_res = evaluate_candidate(
            ModelFamily.RIDGE,
            df,
            TargetMetric.PRODUCTIVITY,
            feature_cols,
        )
        assert eval_res.family == ModelFamily.RIDGE
        assert len(eval_res.block_models) == NUM_BLOCKS
        assert eval_res.avg_val_mse >= 0.0


class TestTrainDashboardModels:
    def test_trains_all_targets(self, tmp_path: Path) -> None:
        df = _synthetic_feature_table()
        bundle = train_dashboard_models(df, feature_columns=["feat_a", "feat_b"])

        assert isinstance(bundle, EnsembleBundle)
        assert len(bundle.targets) == len(TargetMetric)

        for target in TargetMetric:
            target_model = bundle.target_models[target]
            assert isinstance(target_model, EnsembleTargetModel)
            assert len(target_model.block_models) == NUM_BLOCKS
            assert target_model.target == target

        # Save and load roundtrip verification
        save_path = tmp_path / "current.joblib"
        save_model_bundle(bundle, save_path)
        assert save_path.exists()

        loaded_bundle = load_model_bundle(save_path)
        assert loaded_bundle.schema_version == bundle.schema_version
        assert len(loaded_bundle.targets) == len(bundle.targets)

    def test_prediction_averages_across_blocks(self) -> None:
        df = _synthetic_feature_table()
        bundle = train_dashboard_models(df, feature_columns=["feat_a", "feat_b"])

        single_row = df.iloc[0:1]
        prod_score = bundle.predict_target(single_row, TargetMetric.PRODUCTIVITY)
        assert isinstance(prod_score, float)
        assert 0.0 <= prod_score <= 10.0

        all_preds = bundle.predict(single_row)
        assert len(all_preds) == len(TargetMetric)
        assert "productivity" in all_preds
