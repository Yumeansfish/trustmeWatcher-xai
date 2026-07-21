"""Categorize activitywatch events"""

from __future__ import annotations

import json
import re
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


class DomainRule(TypedDict):
    category: str
    rule: str
    domains: list[str]


class TitleRule(TypedDict):
    category: str
    rule: str
    contains: list[str]


class CategoryRules(TypedDict):
    categories: list[str]
    category_aliases: dict[str, str]
    browser_apps: list[str]
    app_exact_rules: dict[str, list[str]]
    web_domain_rules: list[DomainRule]
    title_rules: list[TitleRule]


@dataclass(frozen=True)
class CategoryResult:
    """Store one category result"""

    category: str
    confidence: str
    rule: str
    normalized_domain: str = ""


def load_rules(path: Path = RULES_PATH) -> CategoryRules:
    """Load category rules from json

    Args:
        path: path to the category rule file

    Returns:
        CategoryRules loaded from json
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    return cast(CategoryRules, raw)


RULES = load_rules()
CATEGORIES = RULES["categories"]
CATEGORY_ALIASES = RULES["category_aliases"]
BROWSER_APP_KEYS = {browser.casefold() for browser in RULES["browser_apps"]}
APP_EXACT_RULES = RULES["app_exact_rules"]
WEB_DOMAIN_RULES = RULES["web_domain_rules"]
TITLE_RULES = RULES["title_rules"]


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


def contains_any(text: str, needles: list[str] | tuple[str, ...]) -> bool:
    """Check whether text contains any given value

    Args:
        text: text to search
        needles: values to find

    Returns:
        whether any value appears in the text
    """

    return any(needle in text for needle in needles)


def domain_matches(domain: str, domains: list[str]) -> bool:
    """Match a domain or subdomain

    Args:
        domain: domain to check
        domains: accepted domains

    Returns:
        whether the domain matches
    """

    return any(domain == item or domain.endswith(f".{item}") for item in domains)


def match_title_rule(text: str) -> TitleRule | None:
    """Find the first matching title rule

    Args:
        text: normalized title text

    Returns:
        matching TitleRule or None
    """

    for title_rule in TITLE_RULES:
        if contains_any(text, title_rule["contains"]):
            return title_rule
    return None


def result(
    category: str,
    confidence: str,
    rule: str,
    domain: str = "",
) -> CategoryResult:
    """Build one category result

    Args:
        category: selected category
        confidence: confidence label
        rule: matching rule
        domain: normalized domain

    Returns:
        CategoryResult with category aliases applied
    """

    return CategoryResult(
        category=CATEGORY_ALIASES.get(category, category),
        confidence=confidence,
        rule=rule,
        normalized_domain=domain,
    )


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

    app_norm = normalize_text(app) or "unknown"
    app_l = app_norm.casefold()
    title_l = text_key(title)
    domain = normalize_domain(url)

    if domain:
        web_result = categorize_web_event(url, title)
        category = (
            web_result.category
            if web_result.category != "other"
            else "browser_uncategorized"
        )
        return CategoryResult(
            category=category,
            confidence="high" if web_result.category != "other" else "medium",
            rule=web_result.rule,
            normalized_domain=web_result.normalized_domain,
        )

    if app_l in APP_EXACT_RULES:
        category, rule = APP_EXACT_RULES[app_l]
        return result(category, "high", rule)

    if app_l in BROWSER_APP_KEYS:
        title_rule = match_title_rule(title_l)
        if title_rule is not None:
            return result(
                title_rule["category"],
                "medium",
                title_rule["rule"],
            )
        if not title_l or title_l in {"new tab", "start page"}:
            return result(
                "browser_uncategorized",
                "low",
                "browser:no-title",
            )
        return result(
            "browser_uncategorized",
            "low",
            "browser:unknown-title",
        )

    title_rule = match_title_rule(title_l)
    if title_rule is not None:
        return result(
            title_rule["category"],
            "medium",
            title_rule["rule"],
        )

    return result("other_app", "low", "app:unmapped")


def categorize_web_event(url: object, title: object = "") -> CategoryResult:
    """Categorize one browser tab event

    Args:
        url: tab url
        title: tab title

    Returns:
        CategoryResult for the browser tab
    """

    domain = normalize_domain(url)
    title_l = text_key(title)

    if not domain:
        return result("other", "low", "web:missing-domain")
    if domain in {"newtab", "chrome", "about"}:
        return result(
            "browser_uncategorized",
            "low",
            "domain:browser-internal",
            domain=domain,
        )
    if domain in {"localhost", "127.0.0.1"} or domain.startswith("127."):
        return result("local_tool", "high", "domain:local-dev-tool", domain)

    for domain_rule in WEB_DOMAIN_RULES:
        if domain_matches(domain, domain_rule["domains"]):
            return result(
                domain_rule["category"],
                "high",
                domain_rule["rule"],
                domain=domain,
            )

    title_rule = match_title_rule(title_l)
    if title_rule is not None:
        return result(
            title_rule["category"],
            "medium",
            title_rule["rule"],
            domain,
        )

    if domain.startswith("google.") or domain == "google.com":
        if contains_any(title_l, ("search", "google search")):
            return result(
                "search",
                "high",
                "domain:title:google-search",
                domain,
            )
        return result("search", "medium", "domain:google-general", domain)

    return result("other", "low", "domain:unmapped", domain)
