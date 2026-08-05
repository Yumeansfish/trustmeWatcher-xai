"""Define data exchanged by the runtime inference pipeline"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

LOCAL_TIMEZONE = "Europe/Zurich"

RAW_QUESTION_COLUMNS = [
    "q1_feelings",
    "q2_intensity",
    "q3_tiredness",
    "q4_enthusiasm",
    "q5_immersion",
    "q6_comfort",
    "q7_social",
    "q8_stress",
    "q9_productivity",
]

MODEL_TARGETS = [
    "mood_valence",
    "arousal",
    "restfulness",
    "stress_management",
    "productivity",
    "engagement",
    "overall_wellbeing",
]

TARGET_TITLES = {
    "mood_valence": "Mood valence",
    "arousal": "Arousal",
    "restfulness": "Restfulness",
    "stress_management": "Stress management",
    "productivity": "Productivity",
    "engagement": "Engagement",
    "overall_wellbeing": "Overall wellbeing",
}

BASE_COLUMNS = [
    "bucket",
    "user_id",
    "hostname",
    "timestamp",
    "event_end",
    "date",
    "duration_seconds",
    "source_type",
]

WINDOW_COLUMNS = [
    *BASE_COLUMNS,
    "app",
    "category",
    "category_confidence",
    "category_rule",
    "title",
    "url",
]

WEB_COLUMNS = [
    *BASE_COLUMNS,
    "domain",
    "category",
    "category_confidence",
    "category_rule",
    "url",
    "title",
    "audible",
    "tab_count",
]

INPUT_COLUMNS = [
    *BASE_COLUMNS,
    "presses",
    "clicks",
    "mouse_distance",
    "scroll_abs",
]


class ParsedActivityWatchEvents(NamedTuple):
    """Store parsed ActivityWatch event tables"""

    window: pd.DataFrame
    web: pd.DataFrame
    input: pd.DataFrame
