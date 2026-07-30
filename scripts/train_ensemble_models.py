# PYTHON_ARGCOMPLETE_OK
"""Train 5-block ensemble models across all target metrics and save production current.joblib"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import argcomplete
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error

from trustme_xai.data.composite_scores import combine_state_averages, normalize_answers
from trustme_xai.inference.ensemble_bundle import (
    EnsembleBundle,
    EnsembleTargetModel,
    ModelFamily,
    NUM_BLOCKS,
    TargetMetric,
)
from trustme_xai.inference.model_runtime import save_model_bundle
from trustme_xai.modeling.splits import split_purged_blocks


@dataclass
class CandidateEvaluation:
    """Store candidate evaluation results across 5 blocks for one model family"""

    family: ModelFamily
    avg_val_mse: float
    block_models: list[RegressorMixin]


def create_candidate_model(family: ModelFamily) -> RegressorMixin:
    """Create a new model instance for a given model family

    Args:
        family: model family enum

    Returns:
        unfitted sklearn RegressorMixin instance
    """
    if family == ModelFamily.RIDGE:
        return RidgeCV(alphas=[0.1, 1.0, 10.0])
    if family == ModelFamily.LASSO:
        return LassoCV(alphas=[0.01, 0.1, 1.0])
    if family == ModelFamily.GRADIENT_BOOSTING:
        return GradientBoostingRegressor(max_depth=3, n_estimators=50, random_state=42)
    if family == ModelFamily.RANDOM_FOREST:
        return RandomForestRegressor(max_depth=5, n_estimators=50, random_state=42)
    raise ValueError(f"unsupported model family: {family}")


def fit_and_score_block(
    family: ModelFamily,
    train_block: pd.DataFrame,
    val_block: pd.DataFrame,
    target: TargetMetric,
    feature_columns: list[str],
) -> tuple[RegressorMixin, float]:
    """Train 1 candidate model on train_block and return fitted model and validation MSE

    Args:
        family: model family enum
        train_block: training rows
        val_block: validation rows
        target: target metric enum
        feature_columns: list of feature names

    Returns:
        tuple of (fitted RegressorMixin, float val_mse)
    """
    model = create_candidate_model(family)
    X_train = train_block[feature_columns].to_numpy()
    y_train = train_block[target].to_numpy()

    X_val = val_block[feature_columns].to_numpy()
    y_val = val_block[target].to_numpy()

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, val_preds))
    return model, val_mse


def evaluate_candidate(
    family: ModelFamily,
    df: pd.DataFrame,
    target: TargetMetric,
    feature_columns: list[str],
) -> CandidateEvaluation:
    """Evaluate 1 model family across all 5 purged blocks

    Args:
        family: model family enum
        df: feature table containing user_id, timestamp, features, and targets
        target: target metric enum
        feature_columns: list of feature names

    Returns:
        CandidateEvaluation containing family, avg_val_mse, and 5 block models
    """
    clean_df = df.dropna(subset=[target, *feature_columns]).copy().reset_index(drop=True)
    block_models: list[RegressorMixin] = []
    val_mses: list[float] = []

    for train_idx, val_idx in split_purged_blocks(clean_df):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        train_block = clean_df.iloc[train_idx]
        val_block = clean_df.iloc[val_idx]

        fitted_model, val_mse = fit_and_score_block(
            family,
            train_block,
            val_block,
            target,
            feature_columns,
        )
        block_models.append(fitted_model)
        val_mses.append(val_mse)

    avg_mse = float(np.mean(val_mses)) if val_mses else float("inf")
    return CandidateEvaluation(
        family=family,
        avg_val_mse=avg_mse,
        block_models=block_models,
    )


def train_dashboard_models(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> EnsembleBundle:
    """Benchmark candidate model families and package 5-block production EnsembleBundle

    Args:
        df: preprocessed feature DataFrame with normalized and composite targets
        feature_columns: optional explicit list of feature names to use

    Returns:
        fitted EnsembleBundle instance
    """
    if feature_columns is None:
        # Default feature columns excluding identifiers and targets
        non_feature_cols = {
            "user_id",
            "timestamp",
            "date",
            *[t.value for t in TargetMetric],
            "q1_feelings",
            "q2_intensity",
            "q3_tiredness",
            "q4_enthusiasm",
            "q5_immersion",
            "q6_comfort",
            "q7_social",
            "q8_stress",
            "q9_productivity",
        }
        feature_columns = [col for col in df.columns if col not in non_feature_cols]

    if not feature_columns:
        raise ValueError("df contains no valid feature columns for training")

    candidate_families = [
        ModelFamily.RIDGE,
        ModelFamily.LASSO,
        ModelFamily.GRADIENT_BOOSTING,
        ModelFamily.RANDOM_FOREST,
    ]
    target_models: dict[TargetMetric, EnsembleTargetModel] = {}

    for target in TargetMetric:
        evaluations = [
            evaluate_candidate(family, df, target, feature_columns)
            for family in candidate_families
        ]
        # Functional selection of best candidate by minimum validation MSE
        winner = min(evaluations, key=lambda e: e.avg_val_mse)

        # Compute per-user means and global fallback mean for baseline modeling
        clean_target_df = df.dropna(subset=[target])
        user_means = {
            str(uid): float(rows[target].mean())
            for uid, rows in clean_target_df.groupby("user_id", sort=True)
        }
        global_mean = float(clean_target_df[target].mean())

        target_models[target] = EnsembleTargetModel(
            target=target,
            family=winner.family,
            block_models=winner.block_models,
            user_means=user_means,
            global_mean=global_mean,
            feature_columns=feature_columns,
            avg_val_mse=winner.avg_val_mse,
        )

    return EnsembleBundle(
        targets=list(TargetMetric),
        target_models=target_models,
        feature_columns=feature_columns,
        metadata={"model_version": "5block_ensemble_v1"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="train 5-block ensemble models and save production current.joblib",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="optional input CSV containing pre-calculated features and raw q1-q9 answers",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("src/trustme_xai/current.joblib"),
        help="path to save fitted joblib bundle",
    )
    argcomplete.autocomplete(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv is not None and args.input_csv.exists():
        raw_df = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
    else:
        # Build synthetic features DataFrame for demonstration / quick test
        from trustme_xai.data import parse_answers, parse_aw_data
        from trustme_xai.features import build_features

        data_root = Path("raw_data")
        answers = parse_answers(data_root)
        events = parse_aw_data(data_root)
        raw_df = build_features(events, answers)

    # Normalize targets and compute state averages
    norm_df = normalize_answers(raw_df)
    full_df = combine_state_averages(norm_df)

    print(f"Training 5-block ensemble models on {len(full_df)} rows...")
    bundle = train_dashboard_models(full_df)
    save_model_bundle(bundle, args.output_path)
    print(f"Successfully saved 5-block EnsembleBundle to {args.output_path}")


if __name__ == "__main__":
    main()
