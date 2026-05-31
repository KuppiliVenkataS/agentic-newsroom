"""
Emerging-story scan.

Watches ONLY the broad/general feeds (Reuters, EIA, FT, Bloomberg) for a
developing oil-market story that is NOT already on the current watchlist —
a "new story in the making". It does NOT change focus automatically: it
surfaces a note PROMPTING THE ADVISER to add keywords. The human decides.

Design choices (per project discussion):
- Broad feeds only: general sources surface a new driver before it becomes a
  dominant theme. Iran/ME-specific feeds are excluded so we don't re-detect
  the theme we already track.
- LLM scan (Haiku): cheaper triage model, titles+summaries only (not full
  text), so cost stays negligible at 1-3 cycles/day.
- Persistence: detections are logged across runs so a theme seen repeatedly
  ESCALATES, while a one-off flurry fades — handling false positives without
  acting on a single noisy spike.
- Skippable: set SKIP_EMERGING=true to turn it off.

This module proposes; it never edits the watchlist.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx

from config.settings import (
    ANTHROPIC_API_KEY, WATCHLIST_THEMES, BROAD_FEEDS, STORAGE_ROOT,
)

logger = logging.getLogger(__name__)

SKIP_EMERGING = os.getenv("SKIP_EMERGING", "false").lower() == "true"

# Where rolling detections persist (mirrors the run_history.json pattern)
EMERGING_LOG = STORAGE_ROOT / "data" / "emerging_themes.json"

# A theme must appear in this many distinct cycles before it ESCALATES from
# "possible" to "recurring" in the note to the adviser.
ESCALATE_AFTER_CYCLES = 2
# Drop persisted themes not seen for this many days (stale).
FORGET_AFTER_DAYS = 14

EMERGING_PROMPT = """You are an oil-market editor scanning general financial and energy
headlines for a DEVELOPING story that could become a new driver of oil prices.

Here are recent headlines + summaries from broad news feeds:
{headlines}

The desk is ALREADY tracking these themes — IGNORE anything that fits them:
{known_themes}

Identify ONLY genuinely NEW themes (not in the list above) that:
- appear across MULTIPLE separate items (not a single one-off story), AND
- have a plausible physical oil supply, demand, logistics, or weather mechanism.

Be conservative. If nothing new is genuinely recurring, return an empty list.
Do NOT invent a trend from a single article.

Return ONLY valid JSON, no markdown:
{{
  "emerging": [
    {{
      "theme": "<short plain name, e.g. 'Rhine low water'>",
      "why_it_matters": "<one sentence on the oil mechanism>",
      "suggested_keywords": ["<kw1>", "<kw2>"],
      "example_headlines": ["<headline 1>", "<headline 2>"]
    }}
  ]
}}
"""


def _known_theme_terms() -> str:
    """Flatten the active watchlist themes into a readable list for the prompt."""
    lines = []
    for theme, terms in WATCHLIST_THEMES.items():
        if terms:
            lines.append(f"- {theme}: {', '.join(terms)}")
    return "\n".join(lines) if lines else "- (none active)"


def _select_broad_articles(articles: list[dict]) -> list[dict]:
    """Keep only articles from the broad feeds, by their settings.py source key."""
    out = []
    for a in articles:
        src = (a.get("source") or "").lower()
        if any(bf.lower() in src or src in bf.lower() for bf in BROAD_FEEDS):
            out.append(a)
    return out


def _format_headlines(articles: list[dict], cap: int = 40) -> str:
    lines = []
    for a in articles[:cap]:
        title = (a.get("title") or "").strip()
        summ  = (a.get("summary") or "").strip()[:160]
        src   = a.get("source", "")
        if title:
            lines.append(f"- [{src}] {title}" + (f" — {summ}" if summ else ""))
    return "\n".join(lines)


def _call_haiku(prompt: str) -> str:
    """Cheap triage call. Returns raw text (JSON expected)."""
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 800,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


def _parse_json(raw: str) -> dict:
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    return json.loads(raw)


def _load_history() -> dict:
    try:
        return json.loads(EMERGING_LOG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_history(hist: dict) -> None:
    try:
        EMERGING_LOG.parent.mkdir(parents=True, exist_ok=True)
        EMERGING_LOG.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not persist emerging themes: {e}")


def _update_history(hist: dict, themes: list[dict]) -> dict:
    """Increment cycle counts for seen themes; drop stale ones."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    for t in themes:
        name = (t.get("theme") or "").strip().lower()
        if not name:
            continue
        entry = hist.get(name, {"cycles": 0, "first_seen": now_iso, "details": t})
        entry["cycles"] = entry.get("cycles", 0) + 1
        entry["last_seen"] = now_iso
        entry["details"] = t
        hist[name] = entry

    # Forget stale themes
    cutoff = now - timedelta(days=FORGET_AFTER_DAYS)
    for name in list(hist.keys()):
        try:
            last = datetime.fromisoformat(hist[name]["last_seen"])
            if last < cutoff:
                del hist[name]
        except Exception:
            continue
    return hist


def scan_emerging_stories(articles: list[dict]) -> str:
    """
    Run the emerging-story scan. Returns a markdown "Focus watch" note for the
    adviser (empty string if nothing to report or if disabled/unavailable).

    NEVER edits the watchlist. Proposes only.
    """
    if SKIP_EMERGING:
        return ""
    if not ANTHROPIC_API_KEY:
        logger.info("Emerging scan skipped — no Anthropic key (Haiku required).")
        return ""

    broad = _select_broad_articles(articles or [])
    if len(broad) < 3:
        logger.info("Emerging scan skipped — too few broad-feed articles this cycle.")
        return ""

    try:
        prompt = EMERGING_PROMPT.format(
            headlines=_format_headlines(broad),
            known_themes=_known_theme_terms(),
        )
        parsed = _parse_json(_call_haiku(prompt))
        themes = parsed.get("emerging", []) or []
    except Exception as e:
        logger.warning(f"Emerging scan failed: {e}")
        return ""

    # Persist + escalate
    hist = _update_history(_load_history(), themes)
    _save_history(hist)

    if not themes:
        return ""

    # Build the adviser-facing note, marking recurring themes
    lines = []
    for t in themes:
        name = (t.get("theme") or "").strip()
        if not name:
            continue
        cycles = hist.get(name.lower(), {}).get("cycles", 1)
        tag = "RECURRING" if cycles >= ESCALATE_AFTER_CYCLES else "new"
        why = t.get("why_it_matters", "")
        kws = ", ".join(t.get("suggested_keywords", []) or [])
        lines.append(
            f"- **{name}** ({tag}, seen {cycles}x): {why}"
            + (f"\n  Suggested keywords to add: {kws}" if kws else "")
        )

    if not lines:
        return ""

    return (
        "\n\n---\n\n## 🔎 Focus Watch — possible new story forming\n"
        "*The broad feeds are surfacing themes outside your current watchlist. "
        "Review and, if a real driver, add keywords to the relevant theme in "
        "settings.py (WATCHLIST_THEMES). Not auto-applied.*\n\n"
        + "\n".join(lines)
    )