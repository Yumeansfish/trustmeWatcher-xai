from trustme_xai.feature_pipeline.activity_categories import (
    CATEGORY_ALIASES,
    normalize_domain,
    result,
    unified_categories,
)

EXPECTED_ALIASES = {
    "cloud_infra": "development",
    "data_analysis": "development",
    "design_creative": "writing",
    "finance_admin": "personal_distraction",
    "local_tool": "development",
    "other_app": "other",
    "search": "other",
    "social_distraction": "personal_distraction",
    "travel_logistics": "personal_distraction",
}

EXPECTED_CATEGORIES = [
    "ai_assistant",
    "browser_uncategorized",
    "calendar_tasks",
    "communication",
    "development",
    "file_management",
    "game_distraction",
    "media",
    "meetings",
    "other",
    "personal_distraction",
    "research",
    "system_admin",
    "writing",
]


def test_category_aliases_are_always_applied() -> None:
    assert CATEGORY_ALIASES == EXPECTED_ALIASES
    for category, expected in EXPECTED_ALIASES.items():
        assert result(category, "high", "test").category == expected


def test_unified_categories_are_the_canonical_categories() -> None:
    assert unified_categories() == EXPECTED_CATEGORIES


def test_normalize_domain_handles_urls_and_internal_pages() -> None:
    assert normalize_domain("https://www.example.com/path") == "example.com"
    assert normalize_domain("docs.example.com/path") == "docs.example.com"
    assert normalize_domain("about:blank") == "about"
    assert normalize_domain("chrome://newtab") == "chrome"
    assert normalize_domain("localhost:5600/dashboard") == "localhost"


def test_normalize_domain_rejects_invalid_ports() -> None:
    assert normalize_domain("https://example.com:not-a-port") == "example.com"
    assert normalize_domain("https://[broken") == ""
