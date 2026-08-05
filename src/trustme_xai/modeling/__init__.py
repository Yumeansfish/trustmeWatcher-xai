"""Expose production model training contracts"""

from trustme_xai.inference.model_runtime import (
    GlobalStandardizer,
    ModelBundle,
    PerUserStandardizer,
    TargetModel,
    load_model_bundle,
    save_model_bundle,
)
from trustme_xai.modeling.models import (
    MODEL_NAMES,
    ModelSpec,
    default_model_specs,
    model_specs,
)
from trustme_xai.modeling.splits import (
    production_day_purged_within_user_split,
)
from trustme_xai.modeling.train import TrainingResult, train_models

__all__ = [
    "GlobalStandardizer",
    "MODEL_NAMES",
    "ModelBundle",
    "ModelSpec",
    "PerUserStandardizer",
    "TargetModel",
    "TrainingResult",
    "default_model_specs",
    "load_model_bundle",
    "model_specs",
    "production_day_purged_within_user_split",
    "save_model_bundle",
    "train_models",
]
