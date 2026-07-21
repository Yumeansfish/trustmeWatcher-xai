"""Expose the ActivityWatch-to-model feature pipeline."""

from trustme_xai.feature_pipeline.activity_categories import (
    CategoryResult,
    categorize_web_event,
    categorize_window_event,
)
from trustme_xai.feature_pipeline.activitywatch_parser import (
    parse_activitywatch_events,
)
from trustme_xai.feature_pipeline.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_WINDOW_MINUTES,
    build_production_features,
)

__all__ = [
    "CategoryResult",
    "PRODUCTION_FEATURE_COLUMNS",
    "PRODUCTION_FEATURE_SET",
    "PRODUCTION_WINDOW_MINUTES",
    "build_production_features",
    "categorize_web_event",
    "categorize_window_event",
    "parse_activitywatch_events",
]
