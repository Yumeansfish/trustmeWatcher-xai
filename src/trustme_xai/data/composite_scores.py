"""Derive positive model targets from questionnaire answers"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from trustme_xai.contracts import MODEL_TARGETS

COMPOSITE_TARGETS = list(MODEL_TARGETS)
MAX_SCORE = 6.0

_RAW_COLUMNS = [
    "q1_feelings",
    "q2_intensity",
    "q3_tiredness",
    "q4_enthusiasm",
    "q5_immersion",
    "q8_stress",
    "q9_productivity",
]
_ENGAGEMENT_COLUMNS = ["q4_enthusiasm", "q5_immersion"]
_WELLBEING_COLUMNS = [
    "mood_valence",
    "engagement",
    "stress_management",
]

_RAW_RANGES = {
    "q1_feelings": (-3.0, 3.0),
    "q2_intensity": (0.0, 6.0),
    "q3_tiredness": (0.0, 6.0),
    "q4_enthusiasm": (0.0, 6.0),
    "q5_immersion": (0.0, 6.0),
    "q8_stress": (0.0, 6.0),
    "q9_productivity": (0.0, 6.0),
}


def _validated_answers(answers: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_RAW_COLUMNS) - set(answers.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    result = answers.copy()
    for column in _RAW_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise")
        finite = result[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"{column} contains an infinite value")
        minimum, maximum = _RAW_RANGES[column]
        if ((finite < minimum) | (finite > maximum)).any():
            raise ValueError(
                f"{column} contains values outside {minimum}..{maximum}",
            )
    return result


def normalize_answers(answers: pd.DataFrame) -> pd.DataFrame:
    """Derive the five direct positive targets

    Args:
        answers: pd.DataFrame with raw questionnaire answers

    Returns:
        pd.DataFrame with five positive targets
    """
    result = _validated_answers(answers)
    result["mood_valence"] = result["q1_feelings"] + 3.0
    result["arousal"] = result["q2_intensity"]
    result["restfulness"] = MAX_SCORE - result["q3_tiredness"]
    result["stress_management"] = MAX_SCORE - result["q8_stress"]
    result["productivity"] = result["q9_productivity"]
    return result


def combine_state_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engagement and overall wellbeing

    Args:
        df: pd.DataFrame with direct positive targets

    Returns:
        pd.DataFrame with both mean targets
    """
    missing_eng = sorted(set(_ENGAGEMENT_COLUMNS) - set(df.columns))
    if missing_eng:
        raise ValueError(f"missing engagement columns: {missing_eng}")

    result = df.copy()
    result["engagement"] = result[_ENGAGEMENT_COLUMNS].mean(
        axis=1,
        skipna=False,
    )
    missing_wb = sorted(set(_WELLBEING_COLUMNS) - set(result.columns))
    if missing_wb:
        raise ValueError(f"missing wellbeing columns: {missing_wb}")
    result["overall_wellbeing"] = result[_WELLBEING_COLUMNS].mean(
        axis=1,
        skipna=False,
    )
    return result


def derive_model_targets(answers: pd.DataFrame) -> pd.DataFrame:
    """Derive all seven model targets

    Args:
        answers: pd.DataFrame with raw questionnaire answers

    Returns:
        pd.DataFrame with the seven positive targets
    """
    result = combine_state_averages(normalize_answers(answers))
    unexpected = [target for target in MODEL_TARGETS if target not in result]
    if unexpected:
        raise RuntimeError(f"failed to derive model targets: {unexpected}")
    return result


def derive_model_target_values(
    answers: Mapping[str, object],
) -> dict[str, float]:
    """Derive one runtime-ready target mapping from raw q1--q9 answers."""
    result = derive_model_targets(pd.DataFrame([dict(answers)]))
    row = result.iloc[0]
    return {target: float(row[target]) for target in MODEL_TARGETS}
