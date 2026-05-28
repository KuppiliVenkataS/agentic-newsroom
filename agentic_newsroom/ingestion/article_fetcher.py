"""
Full article body fetcher.

Fetches the complete HTML body of RSS articles and extracts clean text.
Sits between RSS ingestion and cleaning in the pipeline.

Why: RSS feeds only provide title + summary (1-3 sentences).
The LLM extraction is therefore working on headlines, missing:
- Specific quotes (Trump remarks, official statements)
- Numerical detail (volume figures, price targets, timelines)
- Context buried in body paragraphs

This module attempts a full fetch for each RSS article.
Failed fetches fall back gracefully to title+summary.

Boilerplate removal: strips nav, footer, ads, scripts using
a simple tag/class blacklist. Not perfect but good enough for
financial news sites.
"""

import logging
import re
import time
from datetime import datetime, timezone

import httpx

from config.settings import REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)

# Tags whose entire content is boilerplate
STRIP_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "advertisement", "iframe", "noscript", "figure", "figcaption",
    "form", "button", "svg", "img",
]

# CSS classes/IDs that signal boilerplate — partial match
STRIP_CLASSES = [
    "nav", "menu", "footer", "header", "sidebar", "ad", "advertisement",
    "cookie", "popup", "modal", "banner", "social", "share", "related",
    "recommended", "newsletter", "subscribe", "promo", "widget",
    "comment", "author-bio", "tag-list", "breadcrumb",
]

# Sites known to block scrapers or require JS — skip full fetch
SKIP_DOMAINS = [
    "ft.com", "bloomberg.com", "wsj.com", "reuters.com",
    "platts.com", "spglobal.com", "argusmedia.com",
    "icis.com", "tradewindsnews.com",
]

MAX_BODY_CHARS = 8000   # cap to keep extraction prompts manageable
MIN_BODY_CHARS = 200    # below this, full fetch probably failed — use summary


def _domain(url: str) -> str:
    try:
        return url.split("/")[2].replace("www.", "")
    except IndexError:
        return ""


def _should_skip(url: str) -> bool:
    domain = _domain(url)
    return any(skip in domain for skip in SKIP_DOMAINS)


def _strip_tag(html: str, tag: str) -> str:
    """Remove all occurrences of a tag and its content."""
    return re.sub(
        rf"<{tag}[^>]*>.*?</{tag}>",
        " ", html, flags=re.DOTALL | re.IGNORECASE
    )


def _strip_boilerplate_classes(html: str) -> str:
    """Blank out block elements with boilerplate class/id names."""
    for cls in STRIP_CLASSES:
        # div, section, article elements with matching class or id
        html = re.sub(
            rf'<(?:div|section|aside|ul)[^>]*(?:class|id)="[^"]*{cls}[^"]*"[^>]*>.*?</(?:div|section|aside|ul)>',
            " ", html, flags=re.DOTALL | re.IGNORECASE
        )
    return html


def _extract_text(html: str) -> str:
    """Strip all HTML and return clean text."""
    for tag in STRIP_TAGS:
        html = _strip_tag(html, tag)
    html = _strip_boilerplate_classes(html)

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode common entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    text = text.replace("&ldquo;", '"').replace("&rdquo;", '"')
    text = text.replace("&lsquo;", "'").replace("&rsquo;", "'")
    text = text.replace("&mdash;", "—").replace("&ndash;", "–")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_article_body(url: str) -> str:
    """
    Fetch and extract the text body of a single article URL.
    Returns empty string on failure.
    """
    if _should_skip(url):
        return ""

    try:
        response = httpx.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-GB,en;q=0.9",
            },
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        if response.status_code != 200:
            return ""

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return ""

        text = _extract_text(response.text)
        return text[:MAX_BODY_CHARS]

    except Exception as exc:
        logger.debug(f"Full fetch failed for {url}: {exc}")
        return ""


def enrich_articles_with_body(articles: list[dict],
                               delay: float = 0.5) -> list[dict]:
    """
    Attempt full-text fetch for all RSS articles.
    Adds 'body' field to each article dict.
    Falls back to empty string on failure — cleaner.py then
    uses title+summary as before.

    delay: seconds between requests to avoid rate limiting.
    """
    total    = len(articles)
    enriched = []
    fetched  = 0
    skipped  = 0
    failed   = 0

    for idx, article in enumerate(articles):
        if article.get("type") != "rss_article":
            enriched.append({**article, "body": ""})
            continue

        url = article.get("url", "")
        if not url:
            enriched.append({**article, "body": ""})
            continue

        logger.info(f"  Fetching body [{idx+1}/{total}]: {url[:80]}")
        body = fetch_article_body(url)

        if len(body) >= MIN_BODY_CHARS:
            fetched += 1
            logger.debug(f"    Got {len(body)} chars")
        elif _should_skip(url):
            skipped += 1
            logger.debug(f"    Skipped (paywalled domain)")
        else:
            failed += 1
            logger.debug(f"    Body too short ({len(body)} chars) — will use summary")

        enriched.append({**article, "body": body})
        time.sleep(delay)

    logger.info(f"Body fetch complete: {fetched} fetched, {skipped} skipped, {failed} failed")
    return enriched