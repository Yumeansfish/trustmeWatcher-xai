# PYTHON_ARGCOMPLETE_OK
"""Train the fixed compact_90_v1 production bundle."""

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

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import (
    add_self_report_features,
)
from trustme_xai.inference.model_runtime import save_model_bundle
from trustme_xai.modeling.fixed_recipes import COMPACT_MODEL_VERSION
from trustme_xai.modeling.splits import production_day_purged_within_user_split
from trustme_xai.modeling.train import train_compact_models

SOURCE_ROOT = Path("/Users/chrono/trustme-proj/github_repo/usi-trust-me-xai")
DEFAULT_TABLE = (
    SOURCE_ROOT / "tmp" / "exact_kmeans_ablation" / "refit_training_table.csv"
)
DEFAULT_ANSWER_SOURCES = SOURCE_ROOT / "tmp" / "answer_sources.csv"
DEFAULT_OUTPUT = Path("src/trustme_xai/current.joblib")
DEFAULT_BEHAVIOR_MODEL = Path(
    "src/trustme_xai/feature_pipeline/behavior_state_model.json",
)
DEFAULT_REPORT_DIR = Path("/tmp/trustme_xai_compact_90_v1_training")
EXPECTED_ROWS = 1326
EXPECTED_STATE_FIT_HOURS = 4497
MAX_VALIDATION_MSE = 0.86
MAX_TEST_MSE = 0.79


def build_parser() -> argparse.ArgumentParser:
    """Build the compact training command parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="train the fixed compact_90_v1 production bundle",
    )
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--answer-sources-csv",
        type=Path,
        default=DEFAULT_ANSWER_SOURCES,
    )
    parser.add_argument(
        "--behavior-model",
        type=Path,
        default=DEFAULT_BEHAVIOR_MODEL,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_table(
    table_path: Path,
    answer_sources_path: Path,
) -> pd.DataFrame:
    """Load the frozen refit table and causal history.

    Args:
        table_path: Frozen train-validation-test feature table.
        answer_sources_path: Source labels for questionnaire rows.

    Returns:
        Training table with strictly earlier real self-report history.
    """
    table = pd.read_csv(table_path, parse_dates=["timestamp"])
    table["user_id"] = table["user_id"].astype(str)
    if len(table) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} frozen rows")
    required = ["user_id", "timestamp", *PRODUCTION_FEATURE_COLUMNS, *MODEL_TARGETS]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"frozen table is missing columns: {missing}")
    if table.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("frozen table rows must have unique user timestamps")
    values = table[PRODUCTION_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("frozen production features must be finite")

    sources = pd.read_csv(answer_sources_path, parse_dates=["timestamp"])
    sources["user_id"] = sources["user_id"].astype(str)
    sources = sources[["user_id", "timestamp", "answer_source"]]
    if sources.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("answer sources must have unique user timestamps")
    table = table.merge(
        sources,
        on=["user_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    if table["answer_source"].isna().any():
        raise ValueError("every frozen row needs an answer source")
    return add_self_report_features(table)


def behavior_model_provenance(path: Path) -> dict[str, object]:
    """Validate the packaged train-only behavior model.

    Args:
        path: Behavior-state JSON artifact.

    Returns:
        Validated provenance values.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("behavior model is missing provenance")
    expected = {
        "fit_hour_rows": EXPECTED_STATE_FIT_HOURS,
        "n_clusters": 6,
        "n_init": 20,
        "random_state": 42,
        "validation_or_test_activity_used": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"behavior model provenance is invalid: {key}")
    return provenance


def _macro(summary: pd.DataFrame) -> dict[str, float]:
    return {
        "validation_mse": float(summary["validation_mse"].mean()),
        "test_mse": float(summary["test_mse"].mean()),
        "test_rmse": float(summary["test_rmse"].mean()),
        "test_exact": float(summary["test_exact"].mean()),
        "test_within_1": float(summary["test_within_1"].mean()),
        "test_within_1_5": float(summary["test_within_1_5"].mean()),
    }


def main() -> None:
    args = build_parser().parse_args()
    table = load_frozen_table(args.table_csv, args.answer_sources_csv)
    split = production_day_purged_within_user_split(table)
    result = train_compact_models(table, split)
    metrics = _macro(result.summary)
    if metrics["validation_mse"] > MAX_VALIDATION_MSE:
        raise RuntimeError("compact validation MSE exceeds 0.86")
    if metrics["test_mse"] > MAX_TEST_MSE:
        raise RuntimeError("compact test MSE exceeds 0.79")

    behavior_provenance = behavior_model_provenance(args.behavior_model)
    result.bundle.metadata.update(
        {
            "behavior_state_model_sha256": _sha256(args.behavior_model),
            "behavior_state_provenance": behavior_provenance,
            "frozen_training_table_sha256": _sha256(args.table_csv),
            "training_environment": {
                "python": platform.python_version(),
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
    )
    save_model_bundle(result.bundle, args.output_path)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(report_dir / "metrics.csv", index=False)
    result.candidates.to_csv(report_dir / "fixed_recipes.csv", index=False)
    result.test_predictions.to_csv(
        report_dir / "test_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "user_id": table["user_id"],
            "timestamp": table["timestamp"],
            "split": split,
        },
    ).to_csv(report_dir / "split.csv", index=False)
    payload = {
        "model_version": COMPACT_MODEL_VERSION,
        "metrics": metrics,
        "split_counts": {
            name: int((split == name).sum())
            for name in ("train", "validation", "gap", "test")
        },
        "artifact_path": str(args.output_path.resolve()),
        "artifact_sha256": _sha256(args.output_path),
        "behavior_state_model_sha256": _sha256(args.behavior_model),
        "counterfactual_targets": [],
    }
    (report_dir / "report.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
