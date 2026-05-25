"""
RSS feed fetcher.

Fetches all configured feeds, parses them with feedparser,
filters to oil/energy relevant articles only,
then filters by dedup registry, and returns clean article dicts.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from config.settings import RSS_FEEDS, REQUEST_TIMEOUT, MAX_ARTICLES_PER_FEED, USER_AGENT
from ingestion.dedup import DedupRegistry

logger = logging.getLogger(__name__)

# Keywords to match against title + summary.
# An article must contain at least one of these to be kept.
OIL_KEYWORDS = [
    # Commodities
    "oil", "crude", "brent", "wti", "opec", "petroleum",
    "energy", "barrel", "refinery", "drilling", "rig",
    "natural gas", "lng", "pipeline", "fuel", "gasoline",
    "shale", "offshore", "oilfield", "hydrocarbon",
    "ebob", "naphtha", "gasoil", "diesel", "jet fuel",
    "backwardation", "contango", "spread", "futures",

    # Companies
    "saudi aramco", "exxon", "chevron", "shell", "bp",
    "totalenergies", "conocophillips", "halliburton",
    "schlumberger", "baker hughes", "equinor", "adnoc",
    "rosneft", "gazprom", "petrochina", "sinopec","adnoc",
    "vitol", "trafigura", "glencore", "gunvor", "mercuria",

    # Geopolitical — critical for this client
    "hormuz", "strait of hormuz", "iran", "iranian",
    "sanctions", "russia", "ukraine", "opec+",
    "israel", "lebanon", "hezbollah", "houthi",
    "tanker", "shipping", "vessel", "blockade",
    "nuclear", "enrichment", "uranium",

    # Market signals
    "supply cut", "production cut", "oil price", "energy price",
    "oil market", "oil demand", "oil supply", "oil inventory",
    "drawdown", "build", "inventory", "stockpile",
    "iea", "eia", "opec report", "market outlook",
]


def _is_oil_relevant(title: str, summary: str) -> bool:
    """Return True if article title or summary contains an oil keyword."""
    text = (title + " " + summary).lower()
    return any(kw in text for kw in OIL_KEYWORDS)


def _parse_date(entry) -> str:
    """Best-effort ISO timestamp from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


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
        "source":     source_key,
        "url":        url,
        "title":      title,
        "summary":    summary[:2000],
        "published":  _parse_date(entry),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "type":       "rss_article",
    }


def fetch_all_feeds(dedup: DedupRegistry) -> list[dict]:
    """
    Fetch every RSS feed in settings.RSS_FEEDS.
    Keeps only oil/energy relevant articles.
    Returns only new (non-duplicate) articles.
    """
    all_articles: list[dict] = []
    headers = {"User-Agent": USER_AGENT}

    for feed_key, feed_url in RSS_FEEDS.items():
        logger.info(f"Fetching RSS: {feed_key}")
        try:
            response = httpx.get(feed_url, headers=headers, timeout=REQUEST_TIMEOUT,
                                 follow_redirects=True)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception as exc:
            logger.warning(f"  Failed to fetch {feed_key}: {exc}")
            continue

        count_new = 0
        count_filtered = 0

        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            article = _clean_entry(entry, feed_key)
            if article is None:
                continue

            # Oil relevance filter
            if not _is_oil_relevant(article["title"], article["summary"]):
                count_filtered += 1
                continue

            if dedup.is_duplicate(article["url"], article["title"]):
                continue

            dedup.register(article["url"], article["title"])
            all_articles.append(article)
            count_new += 1

        logger.info(f"  {feed_key}: {count_new} new articles ({count_filtered} filtered as non-oil)")

    return all_articles