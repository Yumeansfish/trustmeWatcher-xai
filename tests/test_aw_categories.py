from trustme_xai.feature_pipeline.activity_categories import (
    categorize_web_event,
    categorize_window_event,
    normalize_domain,
    unified_categories,
)

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


def test_jsi_categorize_window_event() -> None:
    res = categorize_window_event("Code", "test.py - Visual Studio Code")
    assert res.category == "development"


def test_jsi_categorize_web_event() -> None:
    res = categorize_web_event("https://github.com/my-repo", "GitHub")
    assert res.category == "development"
    assert res.normalized_domain == "github.com"


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
