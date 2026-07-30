"""Production counterfactual service bridging raw ActivityWatch buckets to Zero-Sum engine"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from trustme_xai.inference.counterfactual_engine import find_counterfactual
from trustme_xai.inference.ensemble_bundle import EnsembleBundle
from trustme_xai.inference.inference_service import _build_feature_row_from_buckets


def run_counterfactual(
    bundle: EnsembleBundle,
    buckets: dict[str, dict[str, Any]],
    user_id: str,
    timestamp: datetime | str | pd.Timestamp,
    target: str,
    desired_score: float,
) -> dict[str, Any]:
    """Parse ActivityWatch buckets, run Zero-Sum Swap engine, and return counterfactual report

    Args:
        bundle: loaded 5-block EnsembleBundle instance
        buckets: raw ActivityWatch bucket dictionary
        user_id: user identifier string
        timestamp: prediction timestamp
        target: target metric identifier string (e.g. "productivity")
        desired_score: desired target continuous score (e.g. 4.5)

    Returns:
        counterfactual report dictionary
    """
    if not user_id.strip():
        raise ValueError("user_id must not be blank")

    ts = pd.Timestamp(timestamp)
    if pd.isna(ts):
        raise ValueError("timestamp must be a valid timestamp")

    # Build 1-row feature DataFrame from raw buckets
    feature_row = _build_feature_row_from_buckets(
        bundle=bundle,
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
