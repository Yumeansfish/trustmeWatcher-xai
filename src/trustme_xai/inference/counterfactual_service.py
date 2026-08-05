"""Build counterfactual reports from ActivityWatch buckets"""

from __future__ import annotations

from collections.abc import Mapping
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
    buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    timestamp: datetime | str | pd.Timestamp,
    target: str,
    desired_score: float,
) -> dict[str, Any]:
    """Build one zero-sum counterfactual report

    Args:
        bundle: fitted production model bundle
        buckets: raw ActivityWatch buckets
        user_id: participant identifier
        timestamp: prediction timestamp
        target: target name
        desired_score: requested score

    Returns:
        counterfactual report
    """
    if not user_id.strip():
        raise ValueError("user_id must not be blank")

    ts = pd.Timestamp(timestamp)
    if pd.isna(ts):
        raise ValueError("timestamp must be a valid timestamp")

    feature_row = build_current_features_from_buckets(
        activitywatch_buckets=buckets,
        user_id=user_id,
        as_of=ts,
    )

    return find_counterfactual(
        bundle=bundle,
        feature_row=feature_row,
        target=target,
        desired_score=desired_score,
    )
