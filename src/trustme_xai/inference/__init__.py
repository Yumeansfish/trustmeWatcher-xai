"""Inference package for 5-block ensemble models and actionable counterfactuals"""

from trustme_xai.inference.ensemble_bundle import (
    EnsembleBundle,
    EnsembleTargetModel,
    ModelFamily,
    NUM_BLOCKS,
    TargetMetric,
)
from trustme_xai.inference.inference_service import run_inference
from trustme_xai.inference.model_runtime import (
    load_model_bundle,
    save_model_bundle,
    validate_ensemble_bundle,
)
from trustme_xai.inference.prediction_report import create_prediction_report

__all__ = [
    "EnsembleBundle",
    "EnsembleTargetModel",
    "ModelFamily",
    "NUM_BLOCKS",
    "TargetMetric",
    "create_prediction_report",
    "load_model_bundle",
    "run_inference",
    "save_model_bundle",
    "validate_ensemble_bundle",
]
