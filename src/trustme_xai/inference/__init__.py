"""Inference package for 5-block ensemble models and actionable counterfactuals"""

from trustme_xai.inference.counterfactual_constraints import (
    ACTIONABLE_CATEGORIES,
    IMMUTABLE_FEATURES,
    WINDOW_TIME_BUDGET,
    check_time_budget,
    is_immutable,
    validate_counterfactual,
)
from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.counterfactual_service import run_counterfactual
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
    "ACTIONABLE_CATEGORIES",
    "EnsembleBundle",
    "EnsembleTargetModel",
    "IMMUTABLE_FEATURES",
    "ModelFamily",
    "NUM_BLOCKS",
    "TargetMetric",
    "WINDOW_TIME_BUDGET",
    "check_time_budget",
    "create_prediction_report",
    "find_counterfactual",
    "is_immutable",
    "load_model_bundle",
    "run_counterfactual",
    "run_inference",
    "save_model_bundle",
    "validate_ensemble_bundle",
    "validate_counterfactual",
]
