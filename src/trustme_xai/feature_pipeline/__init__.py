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
from trustme_xai.feature_pipeline.self_report_features import (
    add_self_report_features,
    all_history_columns,
    target_feature_sets,
)

__all__ = [
    "CategoryResult",
    "PRODUCTION_FEATURE_COLUMNS",
    "PRODUCTION_FEATURE_SET",
    "PRODUCTION_WINDOW_MINUTES",
    "add_self_report_features",
    "all_history_columns",
    "build_production_features",
    "categorize_web_event",
    "categorize_window_event",
    "parse_activitywatch_events",
    "target_feature_sets",
]
