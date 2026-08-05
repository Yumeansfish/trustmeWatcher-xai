# PYTHON_ARGCOMPLETE_OK
"""Train the seven-target production_25 model"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    import argcomplete
except ImportError:
    argcomplete = None  # type: ignore[assignment]

from trustme_xai.contracts import MODEL_TARGETS, ParsedActivityWatchEvents
from trustme_xai.data.composite_scores import derive_model_targets
from trustme_xai.feature_pipeline.activity_categories import (
    categorize_web_event,
    categorize_window_event,
)
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
    build_production_features,
)
from trustme_xai.inference.model_runtime import save_model_bundle
from trustme_xai.modeling.models import MODEL_NAMES, model_specs
from trustme_xai.modeling.splits import production_day_purged_within_user_split
from trustme_xai.modeling.train import TrainingResult, train_models

SOURCE_ROOT = Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai")
DEFAULT_ANSWERS = SOURCE_ROOT / "output" / "answers.csv"
DEFAULT_EVENT_ROOT = SOURCE_ROOT / "output" / "parsed_activitywatch"
DEFAULT_OUTPUT = Path("src/trustme_xai/current.joblib")
DEFAULT_REPORT_DIR = Path("/tmp/trustme_xai_seven_target_training")


def build_parser() -> argparse.ArgumentParser:
    """Build the training command parser

    Returns:
        configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="train seven production_25 regressors",
    )
    parser.add_argument("--answers-csv", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=None,
        help="cached production_25 rows built by the canonical feature pipeline",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="rebuild all feature rows from parsed ActivityWatch events",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--deploy-normalization",
        choices=("global", "per_user"),
        default="per_user",
        help="locked deployment policy independent of test metrics",
    )
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    return parser


def _load_event_table(
    path: Path,
    columns: list[str],
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["timestamp", "event_end"],
    )
    table["user_id"] = table["user_id"].astype(str)
    table = table.merge(bounds, on="user_id", how="inner")
    keep = (table["event_end"] > table["first_needed"]) & (
        table["timestamp"] < table["last_needed"]
    )
    return table.loc[keep, columns].reset_index(drop=True)


def rebuild_features(
    answers: pd.DataFrame,
    event_root: Path,
) -> pd.DataFrame:
    """Build production features from parsed ActivityWatch events

    Args:
        answers: questionnaire rows
        event_root: directory with parsed event CSV files

    Returns:
        pd.DataFrame with exact production features
    """
    bounds = (
        answers.groupby("user_id", as_index=False)["timestamp"]
        .agg(first_needed="min", last_needed="max")
    )
    bounds["first_needed"] -= pd.Timedelta(days=8)
    bounds["last_needed"] += pd.Timedelta(seconds=1)
    window_columns = [
        "user_id",
        "timestamp",
        "event_end",
        "app",
        "title",
        "category",
        "source_type",
        "category_confidence",
    ]
    web_columns = [
        "user_id",
        "timestamp",
        "event_end",
        "url",
        "title",
        "category",
        "source_type",
        "category_confidence",
    ]
    input_columns = [
        "user_id",
        "timestamp",
        "event_end",
        "duration_seconds",
        "presses",
        "clicks",
        "mouse_distance",
        "scroll_abs",
    ]
    window = _load_event_table(
        event_root / "window_events.csv",
        window_columns,
        bounds,
    )
    window[["app", "title"]] = window[["app", "title"]].fillna("")
    unique_window = window[["app", "title"]].drop_duplicates().copy()
    window_categories = [
        categorize_window_event(app, title, "")
        for app, title in unique_window.itertuples(index=False, name=None)
    ]
    unique_window["category"] = [item.category for item in window_categories]
    unique_window["category_confidence"] = [
        item.confidence for item in window_categories
    ]
    window = window.drop(
        columns=["category", "category_confidence"],
    ).merge(unique_window, on=["app", "title"], how="left")

    web = _load_event_table(
        event_root / "web_events.csv",
        web_columns,
        bounds,
    )
    web[["url", "title"]] = web[["url", "title"]].fillna("")
    unique_web = web[["url", "title"]].drop_duplicates().copy()
    web_categories = [
        categorize_web_event(url, title)
        for url, title in unique_web.itertuples(index=False, name=None)
    ]
    unique_web["category"] = [item.category for item in web_categories]
    unique_web["category_confidence"] = [
        item.confidence for item in web_categories
    ]
    web = web.drop(
        columns=["category", "category_confidence"],
    ).merge(unique_web, on=["url", "title"], how="left")

    events = ParsedActivityWatchEvents(
        window=window,
        web=web,
        input=_load_event_table(
            event_root / "input_events.csv",
            input_columns,
            bounds,
        ),
    )
    return build_production_features(
        events,
        answers[["user_id", "timestamp"]],
    )


def load_training_table(
    answers_path: Path,
    event_root: Path,
    feature_cache: Path | None,
    rebuild: bool,
) -> pd.DataFrame:
    """Load answers and exact production feature rows

    Args:
        answers_path: questionnaire answer CSV
        event_root: parsed ActivityWatch CSV directory
        feature_cache: cached production feature CSV
        rebuild: whether to ignore the cache

    Returns:
        pd.DataFrame with raw answers and derived targets
    """
    answers = pd.read_csv(answers_path, parse_dates=["timestamp"])
    answers["user_id"] = answers["user_id"].astype(str)
    if answers.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("answer rows must have unique user timestamps")

    if rebuild:
        features = rebuild_features(answers, event_root)
    else:
        if feature_cache is None:
            raise ValueError("features_csv is required when rebuild is false")
        cached = pd.read_csv(feature_cache, parse_dates=["timestamp"])
        missing = [
            column
            for column in PRODUCTION_FEATURE_COLUMNS
            if column not in cached.columns
        ]
        if missing:
            raise ValueError(f"feature cache is missing columns: {missing}")
        features = cached[
            ["user_id", "timestamp", *PRODUCTION_FEATURE_COLUMNS]
        ].copy()
        features["user_id"] = features["user_id"].astype(str)

    expected_columns = ["user_id", "timestamp", *PRODUCTION_FEATURE_COLUMNS]
    if list(features.columns) != expected_columns:
        raise ValueError("training feature schema does not match production_25")
    if features.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("feature rows must have unique user timestamps")
    values = features[PRODUCTION_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("production features must be finite")

    table = features.merge(
        answers,
        on=["user_id", "timestamp"],
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    if len(table) != len(answers) or len(table) != len(features):
        raise ValueError("answers and production features must match one-to-one")
    return derive_model_targets(table)


def _result_table(
    normalization: str,
    result: TrainingResult,
) -> pd.DataFrame:
    table = result.summary.copy()
    table.insert(0, "normalization", normalization)
    return table


def _macro(table: pd.DataFrame) -> dict[str, float]:
    return {
        "validation_mse": float(table["validation_mse"].mean()),
        "test_mse": float(table["test_mse"].mean()),
        "test_rmse": float(table["test_rmse"].mean()),
        "test_exact": float(table["test_exact"].mean()),
        "test_within_1": float(table["test_within_1"].mean()),
        "test_within_1_5": float(table["test_within_1_5"].mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    args = build_parser().parse_args()
    table = load_training_table(
        args.answers_csv,
        args.event_root,
        args.features_csv,
        args.rebuild_features or args.features_csv is None,
    )
    split = production_day_purged_within_user_split(table)
    specs = model_specs(args.models)
    results = {
        normalization: train_models(
            table,
            split,
            list(PRODUCTION_FEATURE_COLUMNS),
            normalization,
            specs,
        )
        for normalization in ("global", "per_user")
    }

    deployed = results[args.deploy_normalization]
    deployed.bundle.metadata.update(
        {
            "deployment_policy": (
                f"normalization locked to {args.deploy_normalization} before test"
            ),
            "training_environment": {
                "python": platform.python_version(),
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    save_model_bundle(deployed.bundle, args.output_path)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "training_table.csv", index=False)
    metrics = pd.concat(
        [
            _result_table(normalization, result)
            for normalization, result in results.items()
        ],
        ignore_index=True,
    )
    candidates = pd.concat(
        [
            result.candidates.assign(normalization=normalization)
            for normalization, result in results.items()
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [result.test_predictions for result in results.values()],
        ignore_index=True,
    )
    metrics.to_csv(report_dir / "metrics.csv", index=False)
    candidates.to_csv(report_dir / "validation_candidates.csv", index=False)
    predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "user_id": table["user_id"],
            "timestamp": table["timestamp"],
            "split": split,
        },
    ).to_csv(report_dir / "split.csv", index=False)

    comparison = {
        normalization: _macro(
            metrics.loc[metrics["normalization"] == normalization],
        )
        for normalization in ("global", "per_user")
    }
    payload = {
        "source_rows": len(table),
        "targets": MODEL_TARGETS,
        "feature_set": "production_25",
        "feature_columns": PRODUCTION_FEATURE_COLUMNS,
        "split_counts": {
            name: int((split == name).sum())
            for name in ("train", "validation", "gap", "test")
        },
        "normalization_comparison": comparison,
        "deployment_normalization": args.deploy_normalization,
        "preprocessor_fit_scope": "train_plus_validation",
        "artifact_path": str(Path(args.output_path).resolve()),
        "artifact_sha256": _sha256(args.output_path),
        "training_environment": deployed.bundle.metadata["training_environment"],
    }
    (report_dir / "report.json").write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(payload), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
