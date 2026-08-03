"""Keep the import path required by legacy joblib artifacts"""

from trustme_xai.inference.model_runtime import load_model_bundle

_load_model_bundle = load_model_bundle


__all__ = [
    "_load_model_bundle",
    "load_model_bundle",
]

