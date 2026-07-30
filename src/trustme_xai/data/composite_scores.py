"""Normalize survey answers and compute composite psychological scores"""

from __future__ import annotations

import pandas as pd

COMPOSITE_TARGETS = [
    "stress",
    "fatigue",
    "valence",
    "arousal",
    "productivity",
    "engagement",
    "overall_wellbeing",
]

MAX_SCORE = 6.0

# columns required by normalize_answers
_RAW_COLUMNS = [
    "q1_feelings",
    "q2_intensity",
    "q3_tiredness",
    "q4_enthusiasm",
    "q5_immersion",
    "q8_stress",
    "q9_productivity",
]

# columns required by combine_state_averages
_ENGAGEMENT_COLUMNS = ["q4_enthusiasm", "q5_immersion"]
_WELLBEING_COLUMNS = ["stress", "engagement", "valence"]


def normalize_answers(answers: pd.DataFrame) -> pd.DataFrame:
    """Convert raw q1-q9 survey items to a standardized 0-6 scale

    Shifts the bipolar q1_feelings scale from [-3, +3] to [0, 6].
    Reverses q8_stress and q3_tiredness so high values mean low
    stress and low fatigue respectively.

    Args:
        answers: raw survey pd.DataFrame with q1-q9 columns

    Returns:
        pd.DataFrame with five normalized target columns added
    """
    missing = sorted(set(_RAW_COLUMNS) - set(answers.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = answers.copy()

    # reverse negative states so high score = good state
    df["stress"] = MAX_SCORE - df["q8_stress"].to_numpy()
    df["fatigue"] = MAX_SCORE - df["q3_tiredness"].to_numpy()

    # shift bipolar scale (-3 to +3) to match the 0-6 range
    df["valence"] = df["q1_feelings"].to_numpy() + 3.0

    # direct mappings already on the 0-6 scale
    df["arousal"] = df["q2_intensity"]
    df["productivity"] = df["q9_productivity"]

    return df


def combine_state_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement and overall_wellbeing from normalized columns

    Engagement averages q4_enthusiasm and q5_immersion.
    Overall wellbeing averages stress, engagement, and valence.
    Excludes fatigue and arousal to maximize internal consistency.

    Args:
        df: pd.DataFrame with normalized target columns from normalize_answers

    Returns:
        pd.DataFrame with engagement and overall_wellbeing columns added
    """
    missing_eng = sorted(set(_ENGAGEMENT_COLUMNS) - set(df.columns))
    if missing_eng:
        raise ValueError(f"missing engagement columns: {missing_eng}")

    result = df.copy()
    result["engagement"] = result[_ENGAGEMENT_COLUMNS].mean(axis=1)

    missing_wb = sorted(set(_WELLBEING_COLUMNS) - set(result.columns))
    if missing_wb:
        raise ValueError(f"missing wellbeing columns: {missing_wb}")

    result["overall_wellbeing"] = result[_WELLBEING_COLUMNS].mean(axis=1)

    return result
