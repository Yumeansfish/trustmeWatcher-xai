"""Keep compatibility with the parent desired-score counterfactual contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from trustme_xai.feature_pipeline.runtime_features import (
    build_runtime_feature_row,
)
from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.model_runtime import ModelBundle


def run_counterfactual(
    bundle: ModelBundle,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    target: str,
    desired_score: float,
    previous_questionnaire_times: Sequence[object] = (),
    past_self_reports: Sequence[Mapping[str, object]] = (),
) -> dict[str, Any]:
    """Build one legacy desired-score counterfactual report.

    Args:
        bundle: loaded production model bundle
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        target: target name
        desired_score: requested score
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        counterfactual report
    """
    if not user_id.strip():
        raise ValueError("user_id must not be blank")

    feature_row = build_runtime_feature_row(
        model=bundle,
        activitywatch_buckets=activitywatch_buckets,
        user_id=user_id,
        as_of=as_of,
        previous_questionnaire_times=previous_questionnaire_times,
        past_self_reports=past_self_reports,
    )

    return find_counterfactual(
        bundle=bundle,
        feature_row=feature_row,
        target=target,
        desired_score=desired_score,
    )
