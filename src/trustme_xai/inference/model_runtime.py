"""Serialization and validation runtime for EnsembleBundle artifacts"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from trustme_xai.inference.ensemble_bundle import (
    ENSEMBLE_BUNDLE_SCHEMA_VERSION,
    EnsembleBundle,
    NUM_BLOCKS,
)



def save_model_bundle(bundle: EnsembleBundle, path: str | Path) -> None:
    """Save an EnsembleBundle instance to disk using joblib

    Args:
        bundle: EnsembleBundle instance
        path: target filepath
    """
    validate_ensemble_bundle(bundle)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)


def load_model_bundle(path: str | Path) -> EnsembleBundle:
    """Load an EnsembleBundle instance from disk using joblib

    Args:
        path: joblib filepath

    Returns:
        loaded EnsembleBundle instance
    """
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, EnsembleBundle):
        raise TypeError(f"loaded object is not an EnsembleBundle: {type(loaded)}")
    validate_ensemble_bundle(loaded)
    return loaded


_load_model_bundle = load_model_bundle


def validate_ensemble_bundle(bundle: EnsembleBundle) -> None:
    """Validate structural constraints of an EnsembleBundle

    Args:
        bundle: EnsembleBundle instance to validate
    """
    if bundle.schema_version != ENSEMBLE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {bundle.schema_version}")

    if not bundle.targets:
        raise ValueError("bundle.targets must not be empty")

    for target in bundle.targets:
        if target not in bundle.target_models:
            raise ValueError(f"missing target model for {target}")

        model_entry = bundle.target_models[target]
        if len(model_entry.block_models) not in (1, NUM_BLOCKS):
            raise ValueError(
                f"{target} must contain 1 or {NUM_BLOCKS} block models, got {len(model_entry.block_models)}",
            )

