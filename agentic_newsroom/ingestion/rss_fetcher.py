"""
RSS feed fetcher.

Fetches all configured feeds, parses them with feedparser,
filters by dedup registry, and returns clean article dicts.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from config.settings import RSS_FEEDS, REQUEST_TIMEOUT, MAX_ARTICLES_PER_FEED, USER_AGENT
from agentic_newsroom.ingestion.dedup import DedupRegistry

logger = logging.getLogger(__name__)


def _parse_date(entry) -> str:
    """Best-effort ISO timestamp from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.utcnow().isoformat()


def _clean_entry(entry, source_key: str) -> Optional[dict]:
    """Extract the fields we care about from a feedparser entry."""
    url   = getattr(entry, "link", "").strip()
    title = getattr(entry, "title", "").strip()

    if not url or not title:
        return None

    summary = ""
    if hasattr(entry, "summary"):
        summary = entry.summary.strip()
    elif hasattr(entry, "description"):
        summary = entry.description.strip()

    return {
        "source":    source_key,
        "url":       url,
        "title":     title,
        "summary":   summary[:2000],   # cap so JSON stays manageable
        "published": _parse_date(entry),
        "fetched_at": datetime.utcnow().isoformat(),
        "type":      "rss_article",
    }


def fetch_all_feeds(dedup: DedupRegistry) -> list[dict]:
    """
    Fetch every RSS feed in settings.RSS_FEEDS.
    Returns only new (non-duplicate) articles.
    """
    all_articles: list[dict] = []
    headers = {"User-Agent": USER_AGENT}

    for feed_key, feed_url in RSS_FEEDS.items():
        logger.info(f"Fetching RSS: {feed_key}")
        try:
            # Use httpx to fetch the raw bytes (handles redirects cleanly)
            response = httpx.get(feed_url, headers=headers, timeout=REQUEST_TIMEOUT,
                                 follow_redirects=True)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception as exc:
            logger.warning(f"  Failed to fetch {feed_key}: {exc}")
            continue

        count_new = 0
        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            article = _clean_entry(entry, feed_key)
            if article is None:
                continue

            if dedup.is_duplicate(article["url"], article["title"]):
                continue

            dedup.register(article["url"], article["title"])
            all_articles.append(article)
            count_new += 1

        logger.info(f"  {feed_key}: {count_new} new articles")

    return all_articles