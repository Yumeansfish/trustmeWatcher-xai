"""Run production inference from raw ActivityWatch buckets"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from trustme_xai.feature_pipeline.runtime_features import (
    build_runtime_feature_row,
)
from trustme_xai.inference.model_runtime import ModelBundle
from trustme_xai.inference.prediction_report import build_prediction_report


def predict_current(
    bundle: ModelBundle,
    current_features: pd.DataFrame,
) -> dict[str, float]:
    """Predict all saved targets for one feature row

    Args:
        bundle: loaded production model bundle
        current_features: one current feature row

    Returns:
        target names mapped to finite predictions
    """
    if len(current_features) != 1:
        raise ValueError("current_features must contain exactly one row")
    predictions = bundle.predict(current_features)
    values = predictions.iloc[0].to_dict()
    result = {target: float(values[target]) for target in bundle.targets}
    if not np.isfinite(list(result.values())).all():
        raise ValueError("runtime predictions must be finite")
    return result


def run_inference(
    bundle: ModelBundle,
    activitywatch_buckets: Mapping[str, Mapping[str, object]],
    user_id: str,
    as_of: datetime | str | pd.Timestamp,
    previous_questionnaire_times: Sequence[object] | None = None,
    past_self_reports: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Run the complete production inference path

    Args:
        bundle: loaded production model bundle
        activitywatch_buckets: raw ActivityWatch buckets
        user_id: participant identifier
        as_of: prediction timestamp
        previous_questionnaire_times: prior questionnaire times
        past_self_reports: earlier derived self-report scores

    Returns:
        semantic seven-target prediction report
    """
    history = past_self_reports or ()
    current = build_runtime_feature_row(
        model=bundle,
        activitywatch_buckets=activitywatch_buckets,
        user_id=user_id,
        as_of=as_of,
        previous_questionnaire_times=previous_questionnaire_times or (),
        past_self_reports=history,
    )
    row = current.iloc[0]
    return build_prediction_report(
        bundle,
        current,
        str(row["user_id"]),
        pd.Timestamp(row["timestamp"]),
    )
