"""Categorize activitywatch events with taxonomy rules"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast
from urllib.parse import urlsplit

HIDDEN_TEXT_MARKS = {
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\ufeff",
}
RULES_PATH = Path(__file__).with_name("category_rules.json")


class TaxonomyRule(TypedDict, total=False):
    """Store one raw match rule"""

    type: str
    regex: str
    ignore_case: bool


class TaxonomyCategory(TypedDict):
    """Store one raw category"""

    name: list[str]
    rule: TaxonomyRule


class Taxonomy(TypedDict):
    """Store the raw taxonomy"""

    categories: list[TaxonomyCategory]


@dataclass(frozen=True)
class CategoryResult:
    """Store one category result"""

    category: str
    confidence: str = "high"
    rule: str = ""
    normalized_domain: str = ""


@dataclass(frozen=True)
class _CompiledCategory:
    category: str
    depth: int
    pattern: re.Pattern[str] | None


TAXONOMY_TO_LEGACY_CATEGORY: dict[str, str] = {
    "work_meetings": "meetings",
    "work_ai_assistant": "ai_assistant",
    "work": "other",
    "work_programming": "development",
    "work_programming_activitywatch": "development",
    "work_image": "writing",
    "work_video": "media",
    "work_audio": "media",
    "work_3d": "writing",
    "media_games": "game_distraction",
    "media_video": "media",
    "media_social_media": "personal_distraction",
    "media_music": "media",
    "comms": "communication",
    "comms_im": "communication",
    "comms_email": "communication",
    "uncategorized": "other",
    "media": "media",
    "work_office": "writing",
    "work_research_and_reading": "research",
    "system": "system_admin",
    "work_browser": "browser_uncategorized",
    "afk": "other",
    "system_background": "system_admin",
}

LEGACY_CATEGORIES = [
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


def load_rules(path: Path = RULES_PATH) -> Taxonomy:
    """Load the taxonomy from json

    Args:
        path: path to the category rules file

    Returns:
        Taxonomy loaded from json
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(Taxonomy, raw)


def normalize_text(value: object) -> str:
    """Clean text before matching

    Args:
        value: value to clean

    Returns:
        cleaned text
    """

    text = "" if value is None else str(value)
    for mark in HIDDEN_TEXT_MARKS:
        text = text.replace(mark, "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def text_key(value: object) -> str:
    """Clean text and make it lowercase

    Args:
        value: value to clean

    Returns:
        cleaned lowercase text
    """

    return normalize_text(value).casefold()


def normalize_domain(raw_string: object) -> str:
    """Get a clean domain from a url

    Args:
        raw_string: url or domain to clean

    Returns:
        cleaned domain
    """

    text = normalize_text(raw_string).casefold()
    if not text:
        return ""

    if text.startswith(("about:", "chrome:", "chrome-extension:")):
        return text.split(":", 1)[0]

    candidate = text if "://" in text else f"//{text}"
    try:
        domain = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    domain = domain.rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]

    if domain:
        return domain

    if text.startswith(("localhost", "127.")):
        return text.split(":", 1)[0].split("/", 1)[0]
    return ""


def category_slug(path: list[str]) -> str:
    """Convert one category path to a feature-safe name

    Args:
        path: ordered category path

    Returns:
        lowercase snake category name
    """

    text = "_".join(path).casefold().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _compile_categories(taxonomy: Taxonomy) -> tuple[_CompiledCategory, ...]:
    compiled: list[_CompiledCategory] = []
    for item in taxonomy["categories"]:
        raw_rule = item["rule"]
        raw_regex = raw_rule.get("regex", "")
        pattern: re.Pattern[str] | None = None
        if raw_rule.get("type") == "regex" and raw_regex:
            flags = re.IGNORECASE if raw_rule.get("ignore_case", False) else 0
            pattern = re.compile(raw_regex, flags)
        compiled.append(
            _CompiledCategory(
                category=category_slug(item["name"]),
                depth=len(item["name"]),
                pattern=pattern,
            ),
        )
    return tuple(compiled)


RULES = load_rules()
COMPILED_CATEGORIES = _compile_categories(RULES)
CATEGORIES = LEGACY_CATEGORIES


def result(
    category: str,
    confidence: str = "high",
    rule: str = "",
    domain: str = "",
) -> CategoryResult:
    """Build one category result

    Args:
        category: selected category
        confidence: confidence label
        rule: matching rule
        domain: normalized domain

    Returns:
        CategoryResult for event tables
    """

    mapped = TAXONOMY_TO_LEGACY_CATEGORY.get(category, category)
    return CategoryResult(
        category=mapped,
        confidence=confidence,
        rule=rule,
        normalized_domain=domain,
    )


def _match_category(values: tuple[object, ...]) -> str:
    text = "\n".join(normalize_text(value) for value in values)
    best: _CompiledCategory | None = None
    for item in COMPILED_CATEGORIES:
        if item.pattern is None or item.pattern.search(text) is None:
            continue
        if best is None or item.depth > best.depth:
            best = item
    return "uncategorized" if best is None else best.category


def unified_categories() -> list[str]:
    """Return active category names

    Returns:
        sorted category names
    """

    return sorted(CATEGORIES)


def categorize_window_event(
    app: object,
    title: object = "",
    url: object = "",
) -> CategoryResult:
    """Categorize one app window event

    Args:
        app: app name
        title: window title
        url: window url

    Returns:
        CategoryResult for the window event
    """

    domain = normalize_domain(url)
    raw_cat = _match_category((app, title, url))
    return result(category=raw_cat, confidence="high", rule="", domain=domain)


def categorize_web_event(url: object, title: object = "") -> CategoryResult:
    """Categorize one browser tab event

    Args:
        url: tab url
        title: tab title

    Returns:
        CategoryResult for the browser tab
    """

    domain = normalize_domain(url)
    raw_cat = _match_category((title, url))
    return result(category=raw_cat, confidence="high", rule="", domain=domain)
