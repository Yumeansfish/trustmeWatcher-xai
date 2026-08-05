"""Calculate metrics for positive questionnaire targets"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from trustme_xai.inference.model_runtime import clip_predictions


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, int | float]:
    """Calculate continuous and display-scale metrics

    Args:
        y_true: observed positive target values
        y_pred: continuous target predictions

    Returns:
        metric names and values
    """
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = clip_predictions(y_pred)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have the same shape")
    if not len(truth):
        return {
            "n": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "exact": float("nan"),
            "within_1": float("nan"),
            "within_1_5": float("nan"),
        }
    if not np.isfinite(truth).all():
        raise ValueError("target values must be finite")
    if ((truth < 0.0) | (truth > 6.0)).any():
        raise ValueError("target values must be inside 0..6")

    errors = np.abs(truth - prediction)
    mse = float(np.mean(np.square(errors)))
    return {
        "n": int(len(truth)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "exact": float(np.mean(np.rint(truth) == np.rint(prediction))),
        "within_1": float(np.mean(errors <= 1.0)),
        "within_1_5": float(np.mean(errors <= 1.5)),
    }
