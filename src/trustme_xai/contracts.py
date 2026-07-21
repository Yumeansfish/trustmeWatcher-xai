"""Define the data exchanged by the runtime inference pipeline"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

LOCAL_TIMEZONE = "Europe/Zurich"

QUESTION_TARGETS = {
    "q1": "q1_feelings",
    "q2": "q2_intensity",
    "q3": "q3_tiredness",
    "q4": "q4_enthusiasm",
    "q5": "q5_immersion",
    "q6": "q6_comfort",
    "q7": "q7_social",
    "q8": "q8_stress",
    "q9": "q9_productivity",
}
QUESTION_IDS = list(QUESTION_TARGETS)
TARGET_COLUMNS = list(QUESTION_TARGETS.values())
TARGET_QUESTIONS = {
    target: question_id for question_id, target in QUESTION_TARGETS.items()
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
    """Store parsed activitywatch event tables"""

    window: pd.DataFrame
    web: pd.DataFrame
    input: pd.DataFrame
