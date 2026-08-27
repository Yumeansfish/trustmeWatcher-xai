"""Guard the inference distribution against training code."""

from __future__ import annotations

from pathlib import Path

from trustme_xai.inference import action_classifier, model_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "trustme_xai"


def test_source_tree_contains_only_runtime_model_code() -> None:
    assert not (PACKAGE_ROOT / "modeling").exists()
    assert not (PACKAGE_ROOT / "data" / "binarization.py").exists()
    assert not list((REPOSITORY_ROOT / "scripts").glob("train_*.py"))


def test_runtime_model_types_cannot_fit_or_save_artifacts() -> None:
    assert not hasattr(model_runtime.GlobalStandardizer, "fit")
    assert not hasattr(model_runtime.PerUserStandardizer, "fit")
    assert not hasattr(model_runtime, "make_preprocessor")
    assert not hasattr(model_runtime, "save_model_bundle")
    assert not hasattr(action_classifier, "save_action_classifier_bundle")
