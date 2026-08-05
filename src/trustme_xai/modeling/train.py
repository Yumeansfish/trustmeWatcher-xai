"""Select and refit one regressor for each positive target"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import (
    all_history_columns,
    history_column,
    target_feature_sets,
)
from trustme_xai.inference.model_runtime import (
    ModelBundle,
    TargetModel,
    clip_predictions,
    make_preprocessor,
)
from trustme_xai.modeling.metrics import regression_metrics
from trustme_xai.modeling.models import ModelSpec, default_model_specs
from trustme_xai.modeling.splits import validate_split

METRIC_NAMES = ("n", "mse", "rmse", "exact", "within_1", "within_1_5")
PREDICTION_FRAMES = ("absolute", "personal_residual")


@dataclass
class TargetTrainingResult:
    """Store training output for one target"""

    model: TargetModel
    candidates: pd.DataFrame
    test_predictions: pd.DataFrame


@dataclass
class TrainingResult:
    """Store one complete training run"""

    bundle: ModelBundle
    summary: pd.DataFrame
    candidates: pd.DataFrame
    test_predictions: pd.DataFrame


def _flatten_metrics(
    split: str,
    metrics: dict[str, int | float],
) -> dict[str, int | float]:
    return {f"{split}_{name}": metrics[name] for name in METRIC_NAMES}


def _target_rows(
    table: pd.DataFrame,
    row_split: pd.Series,
    split_name: str,
    target: str,
) -> pd.DataFrame:
    return table.loc[
        (row_split == split_name)
        & table[target].notna()
        & table[history_column(target, "mean")].notna()
    ].copy()


def _training_values(
    rows: pd.DataFrame,
    target: str,
    prediction_frame: str,
) -> np.ndarray:
    values = rows[target].to_numpy(dtype=float).copy()
    if prediction_frame == "personal_residual":
        values -= rows[history_column(target, "mean")].to_numpy(dtype=float)
    return values


def _predictions(
    estimator: Any,
    rows: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    prediction_frame: str,
) -> np.ndarray:
    values = np.asarray(
        estimator.predict(rows[feature_columns]),
        dtype=float,
    ).reshape(-1)
    if prediction_frame == "personal_residual":
        values += rows[history_column(target, "mean")].to_numpy(dtype=float)
    return clip_predictions(values)


def _metrics(
    estimator: Any,
    rows: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    prediction_frame: str,
) -> dict[str, int | float]:
    predictions = _predictions(
        estimator,
        rows,
        target,
        feature_columns,
        prediction_frame,
    )
    return regression_metrics(rows[target].to_numpy(dtype=float), predictions)


def train_target(
    table: pd.DataFrame,
    row_split: pd.Series,
    target: str,
    normalization: str,
    model_specs: Sequence[ModelSpec],
) -> TargetTrainingResult:
    """Select one target model then refit it on development rows

    Args:
        table: complete feature and target table
        row_split: locked row split
        target: positive target name
        normalization: global or per_user
        model_specs: candidate model specifications

    Returns:
        selected model with evaluation output
    """
    development_mask = row_split.isin(["train", "validation"])
    development = table.loc[development_mask].copy()
    development_split = row_split.loc[development.index]

    selection_preprocessor = make_preprocessor(
        normalization,
        list(PRODUCTION_FEATURE_COLUMNS),
    ).fit(development, development_split)
    normalized_development = selection_preprocessor.transform(development)
    train = _target_rows(
        normalized_development,
        development_split,
        "train",
        target,
    )
    validation = _target_rows(
        normalized_development,
        development_split,
        "validation",
        target,
    )
    if train.empty or validation.empty:
        raise ValueError(f"{target} needs train and validation history")

    candidates: list[dict[str, int | float | str]] = []
    for feature_set, feature_columns in target_feature_sets(target).items():
        for prediction_frame in PREDICTION_FRAMES:
            for spec in model_specs:
                estimator = spec.factory()
                estimator.fit(
                    train[feature_columns],
                    _training_values(train, target, prediction_frame),
                )
                train_metrics = _metrics(
                    estimator,
                    train,
                    target,
                    feature_columns,
                    prediction_frame,
                )
                validation_metrics = _metrics(
                    estimator,
                    validation,
                    target,
                    feature_columns,
                    prediction_frame,
                )
                candidates.append(
                    {
                        "target": target,
                        "feature_set": feature_set,
                        "feature_count": len(feature_columns),
                        "prediction_frame": prediction_frame,
                        "model": spec.name,
                        **_flatten_metrics("train", train_metrics),
                        **_flatten_metrics("validation", validation_metrics),
                    },
                )

    finite = [
        row
        for row in candidates
        if np.isfinite(float(row["validation_mse"]))
    ]
    if not finite:
        raise ValueError(f"{target} has no finite validation MSE")
    winner = min(finite, key=lambda row: float(row["validation_mse"]))
    winner_spec = next(
        spec for spec in model_specs if spec.name == winner["model"]
    )
    feature_set = str(winner["feature_set"])
    feature_columns = target_feature_sets(target)[feature_set]
    prediction_frame = str(winner["prediction_frame"])

    final_split = pd.Series("train", index=development.index, dtype="object")
    final_preprocessor = make_preprocessor(
        normalization,
        list(PRODUCTION_FEATURE_COLUMNS),
    ).fit(development, final_split)
    normalized_final = final_preprocessor.transform(development)
    final_rows = normalized_final.loc[
        normalized_final[target].notna()
        & normalized_final[history_column(target, "mean")].notna()
    ].copy()
    estimator = winner_spec.factory()
    estimator.fit(
        final_rows[feature_columns],
        _training_values(final_rows, target, prediction_frame),
    )

    raw_test = _target_rows(table, row_split, "test", target)
    if raw_test.empty:
        raise ValueError(f"{target} needs test history")
    test = final_preprocessor.transform(raw_test)
    test_values = _predictions(
        estimator,
        test,
        target,
        feature_columns,
        prediction_frame,
    )
    test_metrics = regression_metrics(
        test[target].to_numpy(dtype=float),
        test_values,
    )
    metrics = {
        **{
            key: value
            for key, value in winner.items()
            if key.startswith(("train_", "validation_"))
        },
        **_flatten_metrics("test", test_metrics),
    }
    model = TargetModel(
        target=target,
        model_name=str(winner["model"]),
        feature_set=feature_set,
        feature_columns=list(feature_columns),
        prediction_frame=prediction_frame,
        estimator=estimator,
        preprocessor=final_preprocessor,
        metrics=metrics,
        fallback_score=float(final_rows[target].mean()),
        score_range=(0.0, 6.0),
    )
    predictions = raw_test[["user_id", "timestamp", target]].copy()
    predictions = predictions.rename(columns={target: "truth"})
    predictions["target"] = target
    predictions["prediction"] = test_values
    predictions["model"] = model.model_name
    predictions["feature_set"] = feature_set
    predictions["prediction_frame"] = prediction_frame
    predictions["normalization"] = normalization
    return TargetTrainingResult(
        model=model,
        candidates=pd.DataFrame(candidates),
        test_predictions=predictions,
    )


def train_models(
    table: pd.DataFrame,
    row_split: pd.Series,
    normalization: str,
    model_specs: Sequence[ModelSpec] | None = None,
) -> TrainingResult:
    """Train all seven production targets

    Args:
        table: complete feature and target table
        row_split: locked row split
        normalization: global or per_user
        model_specs: optional candidate model specifications

    Returns:
        complete training result
    """
    if not table.index.equals(row_split.index):
        raise ValueError("row_split index must match the feature table")
    required = [
        *MODEL_TARGETS,
        *PRODUCTION_FEATURE_COLUMNS,
        *all_history_columns(),
    ]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"feature table is missing columns: {missing}")
    validate_split(table, row_split, "within-user")
    specs = tuple(default_model_specs() if model_specs is None else model_specs)
    if not specs:
        raise ValueError("at least one model must be selected")

    results = [
        train_target(
            table,
            row_split,
            target,
            normalization,
            specs,
        )
        for target in MODEL_TARGETS
    ]
    target_models = {result.model.target: result.model for result in results}
    trained_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    split_counts = {
        name: int((row_split == name).sum())
        for name in ("train", "validation", "gap", "test")
    }
    bundle = ModelBundle(
        feature_set="production_25_history",
        feature_columns=[*PRODUCTION_FEATURE_COLUMNS, *all_history_columns()],
        targets=list(MODEL_TARGETS),
        target_models=target_models,
        metadata={
            "model_version": f"production_25_history_{normalization}_v1",
            "trained_at": trained_at,
            "normalization": normalization,
            "preprocessor_fit_scope": "train_plus_validation",
            "model_selection": "lowest validation MSE",
            "prediction_sensitivity_gamma": 1.0,
            "history_policy": "strictly earlier StreamDeck answers only",
            "split_strategy": "chronological_day_purged_within_user_80_10_10",
            "source_rows": int(len(table)),
            "split_counts": split_counts,
            "target_formulas": {
                "mood_valence": "q1_feelings + 3",
                "arousal": "q2_intensity",
                "restfulness": "6 - q3_tiredness",
                "stress_management": "6 - q8_stress",
                "productivity": "q9_productivity",
                "engagement": "mean(q4_enthusiasm, q5_immersion)",
                "overall_wellbeing": (
                    "mean(mood_valence, engagement, stress_management)"
                ),
            },
            "excluded_questionnaire_items": ["q6_comfort", "q7_social"],
        },
    )
    summary = pd.DataFrame(
        [
            {
                "target": target,
                "selected_model": target_models[target].model_name,
                "feature_set": target_models[target].feature_set,
                "feature_count": len(target_models[target].feature_columns),
                "prediction_frame": target_models[target].prediction_frame,
                **target_models[target].metrics,
            }
            for target in MODEL_TARGETS
        ],
    )
    return TrainingResult(
        bundle=bundle,
        summary=summary,
        candidates=pd.concat(
            [result.candidates for result in results],
            ignore_index=True,
        ),
        test_predictions=pd.concat(
            [result.test_predictions for result in results],
            ignore_index=True,
        ),
    )
