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


HAS_LIGHTGBM = False
HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None



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
    if family == ModelFamily.LIGHTGBM and HAS_LIGHTGBM and LGBMRegressor is not None:
        return LGBMRegressor(n_estimators=100, learning_rate=0.03, num_leaves=15, min_child_samples=10, random_state=42, verbosity=-1)
    if family == ModelFamily.XGBOOST and HAS_XGBOOST and XGBRegressor is not None:
        return XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, min_child_weight=5, random_state=42, n_jobs=1)
    raise ValueError(f"unsupported or unavailable model family: {family}")




from sklearn.preprocessing import StandardScaler


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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model.fit(X_train_scaled, y_train)
    val_preds = model.predict(X_val_scaled)
    val_mse = float(mean_squared_error(y_val, val_preds))
    return model, val_mse



from trustme_xai.modeling.splits import day_purged_within_user_split, split_purged_blocks


def evaluate_candidate(
    family: ModelFamily,
    df: pd.DataFrame,
    target: TargetMetric,
    feature_columns: list[str],
    split_strategy: str = "single_purged_split",
) -> CandidateEvaluation:
    """Evaluate 1 model family across single purged split or purged blocks

    Args:
        family: model family enum
        df: feature table containing user_id, timestamp, features, and targets
        target: target metric enum
        feature_columns: list of feature names
        split_strategy: "single_purged_split" (default) or "purged_5block"


    Returns:
        CandidateEvaluation containing family, avg_val_mse, and block models
    """
    clean_df = df.dropna(subset=[target, *feature_columns]).copy().reset_index(drop=True)
    block_models: list[RegressorMixin] = []
    val_mses: list[float] = []

    if split_strategy == "single_purged_split":
        row_splits = day_purged_within_user_split(clean_df)
        train_idx = clean_df.index[row_splits == "train"].to_numpy()
        val_idx = clean_df.index[row_splits == "validation"].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            fitted_model, val_mse = fit_and_score_block(
                family,
                clean_df.iloc[train_idx],
                clean_df.iloc[val_idx],
                target,
                feature_columns,
            )
            block_models.append(fitted_model)
            val_mses.append(val_mse)
    else:
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
    split_strategy: str = "single_purged_split",
) -> EnsembleBundle:
    """Benchmark candidate model families and package production EnsembleBundle

    Args:
        df: preprocessed feature DataFrame with normalized and composite targets
        feature_columns: optional explicit list of feature names to use
        split_strategy: "single_purged_split" (default) or "purged_5block"


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
    if HAS_LIGHTGBM:
        candidate_families.append(ModelFamily.LIGHTGBM)
    if HAS_XGBOOST:
        candidate_families.append(ModelFamily.XGBOOST)

    target_models: dict[TargetMetric, EnsembleTargetModel] = {}

    for target in TargetMetric:
        evaluations = [
            evaluate_candidate(family, df, target, feature_columns, split_strategy=split_strategy)
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
        metadata={"model_version": "production_purged_v1"},

    )


def parse_args() -> argparse.Namespace:
    default_csv = Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai/output/features_60m.csv")
    parser = argparse.ArgumentParser(
        description="train 5-block ensemble models and save production current.joblib",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_csv if default_csv.exists() else None,
        help="optional input CSV containing pre-calculated features and raw q1-q9 answers",
    )
    parser.add_argument(
        "--feature-set",
        choices=["features_60m", "production_25_v1", "multi", "multi_focus", "focus", "recent_trends", "combined", "categories"],
        default=None,
        help="optional feature set name to dynamically extract from raw_data",
    )

    parser.add_argument(
        "--min-active-minutes",
        type=float,
        default=0.0,
        help="minimum active computer minutes required in 60m window (default 0.0 retains all rows)",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["purged_5block", "single_purged_split"],
        default="single_purged_split",
        help="split strategy: purged_5block (5-fold CV) or single_purged_split (single train/val/test split)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai/raw_data"),
        help="path to raw_data directory containing participant subfolders",
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
    if args.feature_set is None and args.input_csv is not None and args.input_csv.exists():
        print(f"Loading pre-calculated features CSV from {args.input_csv}...")
        raw_df = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
    else:
        import importlib.util
        import sys

        usi_src = Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai/src")
        data_root = args.data_root if args.data_root.exists() else Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai/raw_data")
        feature_set = args.feature_set or "features_60m"
        print(f"Extracting '{feature_set}' feature set from raw data at {data_root}...")

        # Clear trustme_xai from sys.modules so usi_src submodules can load
        for k in list(sys.modules.keys()):
            if k == "trustme_xai" or k.startswith("trustme_xai."):
                del sys.modules[k]

        if str(usi_src) not in sys.path:
            sys.path.insert(0, str(usi_src))

        # Import parsers using spec
        spec_ans = importlib.util.spec_from_file_location("usi_data_answers", usi_src / "trustme_xai" / "data" / "answers.py")
        mod_ans = importlib.util.module_from_spec(spec_ans) # type: ignore
        spec_ans.loader.exec_module(mod_ans) # type: ignore

        spec_aw = importlib.util.spec_from_file_location("usi_data_aw", usi_src / "trustme_xai" / "data" / "aw_parser.py")
        mod_aw = importlib.util.module_from_spec(spec_aw) # type: ignore
        spec_aw.loader.exec_module(mod_aw) # type: ignore

        answers = mod_ans.parse_answers(data_root)
        events = mod_aw.parse_aw_data(data_root)

        # Clear usi_src from sys.modules so TrustmeWatcher modules load
        if str(usi_src) in sys.path:
            sys.path.remove(str(usi_src))
        for k in list(sys.modules.keys()):
            if k == "trustme_xai" or k.startswith("trustme_xai."):
                del sys.modules[k]


        if feature_set == "production_25_v1":
            if str(usi_src) in sys.path:
                sys.path.remove(str(usi_src))
            from trustme_xai.feature_pipeline.production_features import build_production_features
            raw_df = build_production_features(events, answers)
        else:
            spec_feat = importlib.util.spec_from_file_location("usi_features_pipeline", usi_src / "trustme_xai" / "features" / "pipeline.py")
            mod_feat = importlib.util.module_from_spec(spec_feat) # type: ignore
            spec_feat.loader.exec_module(mod_feat) # type: ignore
            raw_df = mod_feat.build_features(events, answers, feature_sets=[feature_set])
            if str(usi_src) in sys.path:
                sys.path.remove(str(usi_src))








    # Feature engineering for start-of-day and time-of-day context
    time_cols = [c for c in raw_df.columns if c.startswith("time_")]
    if time_cols:
        raw_df["total_active_minutes"] = raw_df[time_cols].sum(axis=1)
        raw_df["is_zero_activity"] = (raw_df["total_active_minutes"] == 0).astype(float)
    if "timestamp" in raw_df.columns:
        raw_df["hour_of_day"] = pd.to_datetime(raw_df["timestamp"]).dt.hour.astype(float)

    if time_cols and args.min_active_minutes > 0:
        before_count = len(raw_df)
        raw_df = raw_df[raw_df["total_active_minutes"] >= args.min_active_minutes].copy()
        print(f"Filtered out {before_count - len(raw_df)} AFK/idle survey rows (< {args.min_active_minutes}m active time).")


    # Normalize targets and compute state averages
    norm_df = normalize_answers(raw_df)
    full_df = combine_state_averages(norm_df)

    print(f"Training ensemble models ({args.split_strategy}) on {len(full_df)} active rows...")

    bundle = train_dashboard_models(full_df, split_strategy=args.split_strategy)


    mses = [m.avg_val_mse for m in bundle.target_models.values()]
    rmses = [np.sqrt(m) for m in mses if m < float("inf")]

    print("\n" + "=" * 70)
    print(f"{'Target Metric':<22} | {'Winning Model':<18} | {'Val MSE':<10} | {'Val RMSE':<10}")
    print("=" * 70)
    for target, model_entry in bundle.target_models.items():
        mse = model_entry.avg_val_mse
        rmse = float(np.sqrt(mse)) if mse < float("inf") else float("nan")
        print(f"{str(target):<22} | {str(model_entry.family):<18} | {mse:<10.4f} | {rmse:<10.4f}")
    print("-" * 70)
    print(f"{'AVERAGE (All Targets)':<22} | {'--':<18} | {np.mean(mses):<10.4f} | {np.mean(rmses):<10.4f}")
    print("=" * 70 + "\n")


    save_model_bundle(bundle, args.output_path)
    print(f"Successfully saved 5-block EnsembleBundle to {args.output_path}")


if __name__ == "__main__":
    main()

