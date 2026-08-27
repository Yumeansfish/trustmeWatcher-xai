"""Load the immutable contract for the packaged compact model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any

import numpy as np

from trustme_xai.contracts import MODEL_TARGETS
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
)
from trustme_xai.feature_pipeline.self_report_features import (
    all_history_columns,
    history_column,
)

COMPACT_MODEL_VERSION = "compact_90_v1"
COMPACT_CONTRACT_SCHEMA = "trustme_xai.compact_artifact.v1"
COMPACT_CONTRACT_RESOURCE = "compact_90_v1.json"


@dataclass(frozen=True)
class CompactTargetContract:
    """Describe the estimator inputs stored for one target."""

    model_name: str
    prediction_frame: str
    gamma: float
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class CompactArtifactContract:
    """Own every runtime invariant of the packaged compact score artifact."""

    model_version: str
    feature_set: str
    normalization: str
    minimum_complete_history_rows: int
    counterfactual_targets: tuple[str, ...]
    preprocessor_fit_scope: str
    score_range: tuple[float, float]
    targets: Mapping[str, CompactTargetContract]

    def validate_bundle(self, bundle: Any) -> None:
        """Validate a loaded score bundle against this artifact contract."""
        if getattr(bundle, "feature_set", None) != self.feature_set:
            raise ValueError("compact bundle feature set does not match")
        if getattr(bundle, "targets", None) != MODEL_TARGETS:
            raise ValueError("compact bundle targets do not match the contract")
        expected_features = [*PRODUCTION_FEATURE_COLUMNS, *all_history_columns()]
        if getattr(bundle, "feature_columns", None) != expected_features:
            raise ValueError("compact bundle features do not match the contract")

        metadata = getattr(bundle, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("compact bundle metadata must be an object")
        expected_metadata = {
            "model_version": self.model_version,
            "normalization": self.normalization,
            "minimum_complete_history_rows": self.minimum_complete_history_rows,
            "counterfactual_targets": list(self.counterfactual_targets),
            "preprocessor_fit_scope": self.preprocessor_fit_scope,
            "recipe_manifest": COMPACT_CONTRACT_RESOURCE,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"compact bundle {key} does not match")

        target_models = getattr(bundle, "target_models", {})
        for target, contract in self.targets.items():
            model = target_models[target]
            if getattr(model, "feature_set", None) != self.feature_set:
                raise ValueError(f"{target} compact feature set does not match")
            if getattr(model, "model_name", None) != contract.model_name:
                raise ValueError(f"{target} compact model does not match")
            if getattr(model, "prediction_frame", None) != contract.prediction_frame:
                raise ValueError(f"{target} compact frame does not match")
            if getattr(model, "feature_columns", None) != list(
                contract.feature_columns,
            ):
                raise ValueError(f"{target} compact features do not match")
            if getattr(model, "blend_gamma", None) != contract.gamma:
                raise ValueError(f"{target} compact gamma does not match")
            if getattr(model, "score_range", None) != self.score_range:
                raise ValueError(f"{target} compact score range does not match")


def _target_contract(target: str, payload: Any) -> CompactTargetContract:
    if not isinstance(payload, dict):
        raise ValueError(f"{target} compact contract must be an object")
    features = tuple(str(value) for value in payload.get("features", []))
    contract = CompactTargetContract(
        model_name=str(payload.get("model", "")),
        prediction_frame=str(payload.get("prediction_frame", "")),
        gamma=float(payload.get("gamma", float("nan"))),
        feature_columns=features,
    )
    if not contract.model_name:
        raise ValueError(f"{target} compact model name is invalid")
    if contract.prediction_frame not in {"absolute", "personal_residual"}:
        raise ValueError(f"{target} compact prediction frame is invalid")
    if not np.isfinite(contract.gamma) or not 0.0 <= contract.gamma <= 1.0:
        raise ValueError(f"{target} compact gamma is invalid")
    if not features:
        raise ValueError(f"{target} compact features must not be empty")
    if len(set(features)) != len(features):
        raise ValueError(f"{target} compact features must be unique")
    allowed = {*PRODUCTION_FEATURE_COLUMNS, *all_history_columns()}
    unknown = sorted(set(features) - allowed)
    if unknown:
        raise ValueError(f"{target} compact features are unknown: {unknown}")
    if history_column(target, "mean") not in features:
        raise ValueError(f"{target} compact contract needs its history mean")
    return contract


def load_compact_contract() -> CompactArtifactContract:
    """Load and validate the packaged compact-model contract.

    Returns:
        validated compact artifact contract
    """
    artifact = resources.files("trustme_xai.inference").joinpath(
        COMPACT_CONTRACT_RESOURCE,
    )
    with artifact.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != COMPACT_CONTRACT_SCHEMA:
        raise ValueError("compact contract schema is not supported")
    if payload.get("model_version") != COMPACT_MODEL_VERSION:
        raise ValueError("compact contract model version is not supported")
    target_payloads = payload.get("targets")
    if not isinstance(target_payloads, dict):
        raise ValueError("compact contract needs target objects")
    if list(target_payloads) != MODEL_TARGETS:
        raise ValueError("compact contract targets do not match the runtime")
    targets = {
        target: _target_contract(target, target_payloads[target])
        for target in MODEL_TARGETS
    }
    feature_set = payload.get("feature_set")
    normalization = payload.get("normalization")
    minimum_history = payload.get("minimum_complete_history_rows")
    counterfactual_targets = payload.get("counterfactual_targets")
    preprocessor_fit_scope = payload.get("preprocessor_fit_scope")
    score_range = payload.get("score_range")
    if feature_set != COMPACT_MODEL_VERSION:
        raise ValueError("compact contract feature set is not supported")
    if normalization != "per_user":
        raise ValueError("compact contract normalization is not supported")
    if minimum_history != 1:
        raise ValueError("compact contract history requirement is not supported")
    if counterfactual_targets != []:
        raise ValueError("compact contract must disable counterfactuals")
    if preprocessor_fit_scope != "train_plus_validation":
        raise ValueError("compact contract preprocessor scope is not supported")
    if score_range != [0.0, 6.0]:
        raise ValueError("compact contract score range is not supported")
    return CompactArtifactContract(
        model_version=COMPACT_MODEL_VERSION,
        feature_set=feature_set,
        normalization=normalization,
        minimum_complete_history_rows=minimum_history,
        counterfactual_targets=tuple(counterfactual_targets),
        preprocessor_fit_scope=preprocessor_fit_scope,
        score_range=(float(score_range[0]), float(score_range[1])),
        targets=MappingProxyType(targets),
    )
