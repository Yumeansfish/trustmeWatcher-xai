"""Lock the packaged action-classifier artifact contract."""

from __future__ import annotations

from importlib import resources

import joblib
import pytest

from trustme_xai.inference.action_classifier_contract import (
    validate_action_classifier_artifact,
)


def _packaged_bundle() -> object:
    artifact = resources.files("trustme_xai").joinpath(
        "action_classifier.joblib",
    )
    with resources.as_file(artifact) as path:
        return joblib.load(path)


def test_packaged_classifier_contract_exposes_runtime_facts() -> None:
    facts = validate_action_classifier_artifact(_packaged_bundle())

    assert facts.model_version == "action_classifier_v1"
    assert facts.minimum_complete_history_rows == 1
    assert facts.label_rule == "high_if_score_gt_3_else_not_high"
    assert facts.probability_calibration == "none"
    assert facts.test_macro_metrics == pytest.approx(
        {
            "accuracy": 0.7355356277855608,
            "balanced_accuracy": 0.699078079668029,
            "macro_f1": 0.696203373286336,
            "roc_auc": 0.8023012907569259,
        },
    )


def test_classifier_contract_rejects_a_different_label_rule() -> None:
    bundle = _packaged_bundle()
    bundle.metadata = {**bundle.metadata, "label_rule": "score_gte_3"}

    with pytest.raises(ValueError, match="label rule"):
        validate_action_classifier_artifact(bundle)
