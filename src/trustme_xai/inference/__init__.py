"""Expose the seven-target production inference runtime"""

from trustme_xai.inference.inference_service import (
    build_current_features,
    build_current_features_from_buckets,
    predict_current,
    run_inference,
)
from trustme_xai.inference.model_runtime import (
    GlobalStandardizer,
    ModelBundle,
    PerUserStandardizer,
    TargetModel,
    load_model_bundle,
    save_model_bundle,
    validate_model_bundle,
)
from trustme_xai.inference.prediction_report import (
    build_prediction_report,
)

__all__ = [
    "GlobalStandardizer",
    "ModelBundle",
    "PerUserStandardizer",
    "TargetModel",
    "build_current_features",
    "build_current_features_from_buckets",
    "build_prediction_report",
    "load_model_bundle",
    "predict_current",
    "run_inference",
    "save_model_bundle",
    "validate_model_bundle",
]
