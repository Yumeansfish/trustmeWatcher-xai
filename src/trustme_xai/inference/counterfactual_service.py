"""Build counterfactual reports from ActivityWatch buckets"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.inference_service import (
    build_current_features_from_buckets,
)
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
    """Build one zero-sum counterfactual report

    Args:
        bundle: fitted production model bundle
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

    ts = pd.Timestamp(as_of)
    if pd.isna(ts):
        raise ValueError("as_of must be a valid timestamp")

    feature_row = build_current_features_from_buckets(
        activitywatch_buckets=activitywatch_buckets,
        user_id=user_id,
        as_of=ts,
        previous_questionnaire_times=previous_questionnaire_times,
        past_self_reports=past_self_reports,
        include_actionable_categories=True,
    )

    return find_counterfactual(
        bundle=bundle,
        feature_row=feature_row,
        target=target,
        desired_score=desired_score,
    )
