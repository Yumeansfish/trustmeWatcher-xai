"""Keep the import path required by legacy joblib artifacts"""

from trustme_xai.inference.model_runtime import (
    CURRENT_MODEL_SHA256,
    ModelBundle,
    TargetModel,
    _load_model_bundle,
    load_model_bundle,
)

__all__ = [
    "CURRENT_MODEL_SHA256",
    "ModelBundle",
    "TargetModel",
    "_load_model_bundle",
    "load_model_bundle",
]
