"""Expose the seven-target production inference runtime"""

from trustme_xai.feature_pipeline.runtime_features import (
    build_runtime_feature_row,
)
from trustme_xai.inference.inference_service import predict_current, run_inference
from trustme_xai.inference.model_runtime import (
    ModelBundle,
    TargetModel,
    load_model_bundle,
    validate_model_bundle,
)
from trustme_xai.inference.prediction_report import (
    build_prediction_report,
)

__all__ = [
    "ModelBundle",
    "TargetModel",
    "build_prediction_report",
    "build_runtime_feature_row",
    "load_model_bundle",
    "predict_current",
    "run_inference",
    "validate_model_bundle",
]
