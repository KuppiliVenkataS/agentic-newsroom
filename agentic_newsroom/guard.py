"""
Cycle-health guard.

Decides whether a processed cycle is healthy enough to publish a report.
If NOT, the runner must log the reason and SEND NOTHING — silence is safer
than a wrong/stale report reaching the SLT before anyone is awake to catch it.

This is a pure check: it reads the processed cycle and returns a verdict.
It does not send, write, or delete anything.

Schema it keys off (from real processed JSON):
  cycle = {
    "run_id": str,
    "processed_at": ISO str,
    "article_count": int,
    "articles": [ { "source","fetched_at","published","extraction":[{status,...}] } ]
  }
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Thresholds (tune freely) ──────────────────────────────────────────────
MIN_ARTICLES            = 5      # fewer usable articles than this = empty cycle
MIN_OK_EXTRACTIONS      = 3      # need at least this many successful extractions
MAX_DATA_AGE_HOURS      = 18     # freshest article older than this = stale cycle
# Source-concentration warning (does NOT block — just flags for the adviser)
SOURCE_CONCENTRATION_WARN = 0.85  # if one source is >85% of articles, warn


def _parse_dt(value: str):
    """Parse an ISO datetime; return None on failure."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def check_cycle_health(cycle: dict, now: datetime = None) -> dict:
    """
    Return a verdict dict:
      {
        "publish": bool,          # True = safe to generate & send
        "reasons": [str, ...],    # why blocked (empty if publish=True)
        "warnings": [str, ...],   # non-blocking concerns (e.g. source concentration)
        "stats": {...}            # counts for logging
      }
    """
    now = now or datetime.now(timezone.utc)
    reasons, warnings = [], []

    articles = (cycle or {}).get("articles", []) or []
    n_articles = len(articles)

    # 1) Empty cycle
    if n_articles == 0:
        return {
            "publish": False,
            "reasons": ["empty_cycle: zero articles in processed data"],
            "warnings": [],
            "stats": {"articles": 0, "ok_extractions": 0, "freshest_age_hours": None},
        }

    # 2) Count successful extractions
    ok_extractions = 0
    for a in articles:
        for c in a.get("extraction", []) or []:
            if c.get("status") == "ok":
                ok_extractions += 1

    # 3) Freshness — newest fetched_at (fallback to published)
    newest_dt = None
    for a in articles:
        dt = _parse_dt(a.get("fetched_at")) or _parse_dt(a.get("published"))
        if dt and (newest_dt is None or dt > newest_dt):
            newest_dt = dt
    freshest_age_hours = None
    if newest_dt:
        freshest_age_hours = (now - newest_dt).total_seconds() / 3600.0

    # ── Blocking checks ────────────────────────────────────────────────────
    if n_articles < MIN_ARTICLES:
        reasons.append(f"too_few_articles: {n_articles} < {MIN_ARTICLES}")
    if ok_extractions < MIN_OK_EXTRACTIONS:
        reasons.append(f"too_few_extractions: {ok_extractions} ok < {MIN_OK_EXTRACTIONS}")
    if freshest_age_hours is None:
        reasons.append("no_timestamps: cannot determine data freshness")
    elif freshest_age_hours > MAX_DATA_AGE_HOURS:
        reasons.append(
            f"stale_data: freshest article {freshest_age_hours:.1f}h old "
            f"> {MAX_DATA_AGE_HOURS}h"
        )

    # ── Non-blocking warnings ──────────────────────────────────────────────
    from collections import Counter
    srcs = Counter(a.get("source", "unknown") for a in articles)
    if srcs:
        top_src, top_n = srcs.most_common(1)[0]
        share = top_n / n_articles
        if share > SOURCE_CONCENTRATION_WARN:
            warnings.append(
                f"source_concentration: '{top_src}' is {share:.0%} of articles "
                f"({top_n}/{n_articles}) — read may inherit one source's bias"
            )

    publish = len(reasons) == 0
    stats = {
        "articles": n_articles,
        "ok_extractions": ok_extractions,
        "freshest_age_hours": round(freshest_age_hours, 1) if freshest_age_hours is not None else None,
        "distinct_sources": len(srcs),
    }
    return {"publish": publish, "reasons": reasons, "warnings": warnings, "stats": stats}


def log_verdict(verdict: dict) -> None:
    """Helper to log a verdict consistently."""
    s = verdict["stats"]
    logger.info(
        f"Cycle health: publish={verdict['publish']} | "
        f"articles={s['articles']} ok_extractions={s['ok_extractions']} "
        f"freshest_age_h={s['freshest_age_hours']} sources={s.get('distinct_sources')}"
    )
    for w in verdict["warnings"]:
        logger.warning(f"  ⚠ {w}")
    if not verdict["publish"]:
        for r in verdict["reasons"]:
            logger.error(f"  ✖ BLOCKED: {r}")
        logger.error("  → Sending nothing this cycle (guard).")