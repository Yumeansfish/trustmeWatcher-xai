"""Run prediction and one-target recourse with the action classifier."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Protocol

import joblib
import numpy as np
import pandas as pd

from trustme_xai.feature_pipeline.action_allocations import (
    ActionAllocation,
    ActionChange,
)
from trustme_xai.feature_pipeline.runtime_features import (
    build_runtime_feature_row,
)
from trustme_xai.inference.action_classifier_contract import (
    ACTION_CLASSIFIER_SCHEMA_VERSION,
    ActionClassifierArtifactFacts,
    validate_action_classifier_artifact,
)

STEP_MINUTES = 5.0
MAX_STEPS = 12
MIN_SLIGHT_DELTA = 0.10
MIN_GREAT_DELTA = 0.25
EPSILON = 1e-12

PredictedState = Literal["high", "not_high"]
SuggestionStrength = Literal[
    "Keep",
    "Improve slightly",
    "Greatly improve",
]


@dataclass
class ActionClassifierTargetModel:
    """Store one binary target classifier loaded from an artifact."""

    target: str
    model_name: str
    feature_columns: list[str]
    estimator: Any
    preprocessor: Any
    metrics: dict[str, float] = field(default_factory=dict)

    def probability_high(self, table: pd.DataFrame) -> np.ndarray:
        """Estimate high-state probability for each row.

        Args:
            table: classifier feature rows

        Returns:
            finite high-state probabilities
        """
        missing = sorted(set(self.feature_columns) - set(table.columns))
        if missing:
            raise ValueError(f"classifier features are missing: {missing}")
        transformed = self.preprocessor.transform(table)
        probabilities = np.asarray(
            self.estimator.predict_proba(transformed[self.feature_columns]),
            dtype=float,
        )
        classes = np.asarray(self.estimator.classes_, dtype=int)
        high_positions = np.flatnonzero(classes == 1)
        if len(high_positions) != 1:
            raise ValueError("classifier estimator must expose the high class")
        if probabilities.ndim != 2 or probabilities.shape[0] != len(table):
            raise ValueError("classifier returned an invalid probability table")
        result = probabilities[:, int(high_positions[0])]
        if not np.isfinite(result).all() or bool(
            ((result < 0.0) | (result > 1.0)).any(),
        ):
            raise ValueError("classifier probabilities must be between zero and one")
        return result


@dataclass
class ActionClassifierBundle:
    """Store all seven action-compatible target classifiers."""

    targets: list[str]
    target_models: dict[str, ActionClassifierTargetModel]
    metadata: dict[str, Any]
    schema_version: int = ACTION_CLASSIFIER_SCHEMA_VERSION

    @property
    def model_version(self) -> str:
        """Return the artifact model version."""
        return str(self.metadata["model_version"])

    @property
    def minimum_complete_history_rows(self) -> int:
        """Return the number of complete prior reports required by inference."""
        return int(self.metadata.get("minimum_complete_history_rows", 0))

    def predict_high(self, table: pd.DataFrame) -> dict[str, float]:
        """Estimate every target's high-state probability for one row.

        Args:
            table: one classifier feature row

        Returns:
            target names mapped to high-state probability
        """
        if len(table) != 1:
            raise ValueError("classifier input must contain exactly one row")
        return {
            target: float(self.target_models[target].probability_high(table)[0])
            for target in self.targets
        }

    def probability_high(self, target: str, table: pd.DataFrame) -> np.ndarray:
        """Estimate one target's high-state probability for every row."""
        if target not in self.target_models:
            raise ValueError(f"target not found in action classifier bundle: {target}")
        return self.target_models[target].probability_high(table)


class _ClassifierModels(Protocol):
    targets: list[str]
    model_version: str
    minimum_complete_history_rows: int

    def predict_high(self, table: pd.DataFrame) -> dict[str, float]: ...

    def probability_high(self, target: str, table: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class TargetProbability:
    """Describe one target's baseline classifier output."""

    target: str
    probability_high: float
    state: PredictedState


@dataclass(frozen=True)
class PreparedClassifierSnapshot:
    """Store reusable model input and baseline target probabilities."""

    user_id: str
    prediction_timestamp: str
    model_version: str
    snapshot_id: str
    predictions: tuple[TargetProbability, ...]
    _feature_columns: tuple[str, ...] = field(repr=False)
    _feature_values: tuple[Any, ...] = field(repr=False)

    @property
    def action_allocation(self) -> Mapping[str, float]:
        """Return an immutable action allocation by domain name."""
        values = dict(zip(self._feature_columns, self._feature_values, strict=True))
        return ActionAllocation.from_feature_row(values).minutes

    def _feature_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [dict(zip(self._feature_columns, self._feature_values, strict=True))],
        )


@dataclass(frozen=True)
class CounterfactualSuggestion:
    """Describe one target's selected action-allocation change."""

    user_id: str
    prediction_timestamp: str
    snapshot_id: str
    target: str
    baseline_probability_high: float
    projected_probability_high: float
    delta_probability_high: float
    strength: SuggestionStrength
    changes: tuple[ActionChange, ...]


class ActionClassifierRuntime:
    """Own one loaded classifier artifact and its inference lifecycle."""

    __slots__ = ("_contract", "_models")

    def __init__(
        self,
        *,
        _models: _ClassifierModels,
        _contract: ActionClassifierArtifactFacts | None,
    ) -> None:
        self._models = _models
        self._contract = _contract

    @classmethod
    def _from_bundle(
        cls,
        bundle: ActionClassifierBundle,
    ) -> ActionClassifierRuntime:
        return cls(
            _models=bundle,
            _contract=validate_action_classifier_artifact(bundle),
        )

    @classmethod
    def _from_models(
        cls,
        models: _ClassifierModels,
    ) -> ActionClassifierRuntime:
        return cls(_models=models, _contract=None)

    @property
    def model_version(self) -> str:
        """Return the loaded classifier version."""
        if self._contract is not None:
            return self._contract.model_version
        return self._models.model_version

    def prepare_snapshot(
        self,
        activitywatch_buckets: Mapping[str, Mapping[str, object]],
        user_id: str,
        as_of: datetime | str | pd.Timestamp,
        previous_questionnaire_times: Sequence[object] = (),
        past_self_reports: Sequence[Mapping[str, object]] = (),
    ) -> PreparedClassifierSnapshot:
        """Calculate canonical features and every baseline probability once."""
        return _prepare_classifier_snapshot(
            models=self._models,
            activitywatch_buckets=activitywatch_buckets,
            user_id=user_id,
            as_of=as_of,
            previous_questionnaire_times=previous_questionnaire_times,
            past_self_reports=past_self_reports,
        )

    def generate_counterfactual(
        self,
        snapshot: PreparedClassifierSnapshot,
        target: str,
    ) -> CounterfactualSuggestion:
        """Generate exactly one suggestion for one selected target."""
        return _generate_target_counterfactual(
            models=self._models,
            snapshot=snapshot,
            target=target,
        )


def _load_action_classifier_bundle(path: str | Path) -> ActionClassifierBundle:
    """Load and validate one classifier artifact.

    Args:
        path: saved joblib path

    Returns:
        loaded classifier bundle
    """
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, ActionClassifierBundle):
        raise TypeError(
            f"loaded object is not an ActionClassifierBundle: {type(loaded)}",
        )
    return loaded


def load_action_classifier_runtime(path: str | Path) -> ActionClassifierRuntime:
    """Load one classifier artifact behind the runtime interface.

    Args:
        path: saved joblib path

    Returns:
        loaded action-classifier runtime
    """
    return ActionClassifierRuntime._from_bundle(_load_action_classifier_bundle(path))


def load_packaged_action_classifier() -> ActionClassifierRuntime:
    """Load the action-classifier runtime distributed with the package.

    Returns:
        packaged action-classifier runtime
    """
    artifact = resources.files("trustme_xai").joinpath(
        "action_classifier.joblib",
    )
    with resources.as_file(artifact) as path:
        return load_action_classifier_runtime(path)


def _prepare_classifier_snapshot(
    models: _ClassifierModels,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] = (),
    past_self_reports: Sequence[Mapping[str, object]] = (),
) -> PreparedClassifierSnapshot:
    """Calculate reusable features and all baseline probabilities.

    Args:
        models: loaded classifier models
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        one prepared classifier snapshot
    """
    features = build_runtime_feature_row(
        model=models,
        activitywatch_buckets=activitywatch_buckets,
        user_id=user_id,
        as_of=as_of,
        previous_questionnaire_times=previous_questionnaire_times,
        past_self_reports=past_self_reports,
    )
    probabilities = models.predict_high(features)
    predictions = tuple(
        TargetProbability(
            target=target,
            probability_high=probabilities[target],
            state="high" if probabilities[target] >= 0.5 else "not_high",
        )
        for target in models.targets
    )
    row = features.iloc[0]
    return PreparedClassifierSnapshot(
        user_id=str(row["user_id"]),
        prediction_timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
        model_version=models.model_version,
        snapshot_id=_snapshot_id(models.model_version, features),
        predictions=predictions,
        _feature_columns=tuple(features.columns),
        _feature_values=tuple(row[column] for column in features.columns),
    )


def _snapshot_id(model_version: str, features: pd.DataFrame) -> str:
    row = features.iloc[0]
    values: list[object] = []
    for column in features.columns:
        value = row[column]
        if bool(pd.isna(value)):
            values.append(None)
        elif isinstance(value, pd.Timestamp):
            values.append(value.isoformat())
        elif isinstance(value, np.generic):
            values.append(value.item())
        else:
            values.append(value)
    payload = json.dumps(
        {
            "model_version": model_version,
            "columns": list(features.columns),
            "values": values,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _baseline_probability(
    snapshot: PreparedClassifierSnapshot,
    target: str,
) -> float:
    matches = [
        prediction.probability_high
        for prediction in snapshot.predictions
        if prediction.target == target
    ]
    if len(matches) != 1:
        raise ValueError(f"snapshot does not contain target: {target}")
    return float(matches[0])


def _suggestion_strength(
    deltas: np.ndarray,
) -> tuple[SuggestionStrength, int]:
    great = np.flatnonzero(deltas >= MIN_GREAT_DELTA - EPSILON)
    if len(great):
        return "Greatly improve", int(great[0])
    slight = np.flatnonzero(deltas >= MIN_SLIGHT_DELTA - EPSILON)
    if len(slight):
        return "Improve slightly", int(slight[0])
    return "Keep", 0


def _generate_target_counterfactual(
    models: _ClassifierModels,
    snapshot: PreparedClassifierSnapshot,
    target: str,
) -> CounterfactualSuggestion:
    """Generate one counterfactual for one requested target.

    Args:
        models: loaded classifier models
        snapshot: prepared baseline snapshot
        target: target selected by the user

    Returns:
        one counterfactual suggestion
    """
    if target not in models.targets:
        raise ValueError(f"target not found in action classifier bundle: {target}")
    if snapshot.model_version != models.model_version:
        raise ValueError("snapshot and action classifier versions do not match")

    baseline = _baseline_probability(snapshot, target)
    feature_row = snapshot._feature_frame()
    original = ActionAllocation.from_feature_row(feature_row.iloc[0])
    probabilities = [baseline]
    allocations = [original]
    current = original

    if baseline < 0.5:
        for _ in range(MAX_STEPS):
            scenarios = current.swap_scenarios(feature_row, STEP_MINUTES)
            if scenarios.empty:
                break
            candidate_probabilities = models.probability_high(target, scenarios)
            position = int(np.argmax(candidate_probabilities))
            best = float(candidate_probabilities[position])
            if best <= probabilities[-1] + EPSILON:
                break
            current = ActionAllocation.from_feature_row(scenarios.iloc[position])
            probabilities.append(best)
            allocations.append(current)

    probability_values = np.asarray(probabilities, dtype=float)
    strength, selected_index = _suggestion_strength(
        probability_values - baseline,
    )
    selected = allocations[selected_index]
    projected = float(probability_values[selected_index])
    changes = original.changes_to(selected)
    return CounterfactualSuggestion(
        user_id=snapshot.user_id,
        prediction_timestamp=snapshot.prediction_timestamp,
        snapshot_id=snapshot.snapshot_id,
        target=target,
        baseline_probability_high=baseline,
        projected_probability_high=projected,
        delta_probability_high=projected - baseline,
        strength=strength,
        changes=changes,
    )
