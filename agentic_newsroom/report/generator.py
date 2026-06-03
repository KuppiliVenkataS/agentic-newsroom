"""
Report generator.

Uses Ollama to write a structured oil market analyst report combining:
- Price prediction signal
- Top mentioned organisations from knowledge graph
- Most relevant articles from vector DB semantic search
- Latest EIA price and inventory data
- Recent events from knowledge graph

Output: markdown report saved to STORAGE_ROOT/reports/
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import os

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REPORT_DIR, USER_WATCHLIST, WATCHLIST_BOOST, ANTHROPIC_API_KEY, USE_CLAUDE_REPORT

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
from vectordb.store import VectorStore
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are an experienced oil market analyst writing a morning briefing note.
You are sharp, balanced, and intellectually honest. Your opinions are driven by evidence, not habit.
Write in first person, flowing paragraphs, no headers, no bullet points.

Today's date and time: {date_time}

## Price Data
Two reference sources provided. Use directionally; the adviser confirms precise
live figures at review from Argus/Platts.
Brent (article-quoted): {art_brent}
WTI (article-quoted): {art_wti}
Brent (EIA, {eia_brent_age}): {brent_price} USD/barrel
WTI (EIA, {eia_wti_age}): {wti_price} USD/barrel
WTI 5-day trend score: {wti_trend}
Brent 5-day trend score: {brent_trend}

PRICE-HANDLING RULES (important):
- EIA figures are authoritative open data but may be 1-3 days old; article-quoted
  figures are fresher but sourced from news text. Use whichever is more recent as
  directional context. Neither replaces the adviser's live Argus/Platts read.
- For assessed prices (Dated Brent, Ebob, gasoil crack): DO NOT invent numbers.
  Leave slots blank — the adviser fills from their terminal.
- Your DIRECTIONAL read does not depend on the exact decimal — write the analysis
  on events and direction; the adviser drops in the precise live numbers.

## Market Signal
Overall direction: {direction} (confidence: {confidence}, score: {score})
News sentiment — Bullish weight: {bullish_w}, Bearish weight: {bearish_w}
GDELT global news tone: {gdelt_tone} ({gdelt_records} articles)
Geopolitical risk level: {geo_level}

## HIGH-IMPORTANCE EVENTS (importance score ≥ 0.7) — LEAD WITH THESE
{high_importance_events}

## Breaking News (last 24h)
{breaking_news}

## Key Organisations in the News
{top_orgs}

## Other Relevant Stories
{relevant_news}

## Recent Events and Developments
{recent_events}

## 3-Day Outlook
Direction: {direction_3d}, Confidence: {confidence_3d}
Rationale: {rationale_3d}

---

OUTPUT STRUCTURE — TWO SECTIONS, IN THIS ORDER:

You write for TWO readers in one document. Produce both sections, separated
exactly by a line containing only "===ANALYST===".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — PROMOTER BRIEF (this goes to the refinery owner / top brass)
This is the most important part. The reader is a busy refinery promoter, not a
trader. They have 20 seconds. They care about ONE thing: does anything today
change our crude/feedstock cost, our margin, or our risk — and should we act?

RULES FOR THE PROMOTER BRIEF — follow exactly:
- LENGTH: ~150 words. HARD CAP 170. Never longer, even on a big news day.
- LINE 1: a plain headline that IS the conclusion. e.g.
  "Crude easing on ceasefire bets — no action needed today" — not "Daily Brief".
- LINE 2: Brent and WTI levels, one line.
- THEN 2-3 short paragraphs in a sharp, named-adviser voice. Speak to the owner
  directly. Translate everything into THEIR world:
    * say "feedstock cost" / "crude cost" / "margin" — NOT "backwardation",
      "5-day trend score", "$91.50 on volume", or other trader jargon.
    * if the move is good for a refiner's buying cost, say so plainly.
- Give exactly ONE clear carry-away — the single thing to know walking into the day.
- Be opinionated on direction and what it means for US, but do NOT instruct a
  specific trade or hedge. "The drift favours our crude cost; nothing forces a
  move today" — inform the decision, don't make it for them.
- A calm day is stated calm. Never manufacture urgency. "Nothing today demands
  action" is a valuable sentence — use it when true.

After the promoter brief, output the separator line: ===ANALYST===

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — ANALYST NOTE (this stays with the adviser, for their own judgment)
This is the fuller analytical view — the depth behind the brief. Here you MAY
use market structure, spreads, price levels, trader framing. Its length is
governed by the DELIVERY MODE block at the very top of this prompt.

STRUCTURE for the analyst note:
1. First line: date, time, location (e.g. "30/05/2026, 5AM, London")
2. Second line: Brent and WTI prices, key spread or structure data if available
3. Body: flowing paragraphs, no headers, no bullet points
4. LENGTH: governed by the DELIVERY MODE block above (quiet = short, loud = up to
   700 words for the adviser). The promoter brief above is ALWAYS ~150 regardless.

TONE — this is critical:
You are a versatile analyst, not a permabull or permabear. Your tone follows the data:
- When a peace deal or ceasefire genuinely reduces risk, say so clearly and price in the bearish impact — do not reflexively dismiss it
- When supply fundamentals are actually tight, be bullish with conviction
- When geopolitical noise is overdone relative to actual supply impact, say the market is overreacting
- When something positive happens — a deal, a de-escalation, a supply restoration — acknowledge it genuinely, not grudgingly
- When something is bad, be direct and specific about why, not vague
- Avoid defaulting to skepticism as a personality trait — skepticism should be earned by the data
- Vary your sentence structure — avoid starting consecutive sentences with "I think" or "my view is"
- Each paragraph should have a distinct emotional register: one might be cautious, the next genuinely optimistic, the next analytical

CONTENT PRIORITY:
1. Lead with the most market-moving event from HIGH-IMPORTANCE EVENTS — name actors, locations, specifics
2. Connect directly to price impact: direction, magnitude, timeframe
3. Cover each distinct geopolitical thread separately — Iran, Russia, Israel/Lebanon are different stories
4. Price action and market structure — backwardation, spreads, curve shape — only if data supports
5. Supply/demand fundamentals — inventory, production, demand signals
6. Personal 12-24h outlook — specific, with a Brent price level to watch

USING THE DATA:
- Work through every story in Other Relevant Stories — each gets at least a sentence
- Use actual numbers from the data — prices, volumes, percentages
- Do not invent facts not in the data above
- If a story is genuinely bullish, say so. If bearish, say so. If mixed, say so and explain why

AVOIDING STALENESS:
- If an event has no new development, note it briefly and move on
- Do not repeat the same point across paragraphs
- Do not use: "oil markets remain volatile", "uncertainty persists", "closely watched"

END the ANALYST NOTE with your personal 12-24h outlook — include a specific Brent price level, direction, and the one event that would change your view either way. (Do NOT put price levels or trade triggers in the promoter brief.)
"""

OIL_MARKET_KNOWLEDGE = """
## Oil Market Domain Knowledge — apply this when interpreting data

### Investment and Capex
- Lower upstream investment / capex cuts = BULLISH for oil prices (less future supply coming to market)
- Higher upstream investment = BEARISH long-term (more future supply), but neutral short-term
- Do NOT apply equities logic here — in oil markets, less investment means supply constraints ahead
- IEA or industry reports showing capex decline should be flagged as a medium-term bullish signal

### Spreads and Market Structure
- Brent-WTI spread widening (Brent premium increasing):
  = US crude at a discount, caused by: US export bottlenecks, domestic oversupply, pipeline constraints, or weak US refinery demand
  = NOT a sign of weak global demand — Brent reflects global/seaborne market separately
- Brent-WTI spread narrowing:
  = US crude catching up, usually due to strong US exports or Gulf Coast demand
- Backwardation (prompt > forward): market is tight NOW, supply shortage, bullish signal
- Contango (prompt < forward): oversupply or weak demand, bearish signal
- Steepening backwardation = supply stress increasing
- Collapsing backwardation = supply stress easing, potentially bearish

### Geopolitical Price Impact
- Hormuz closure (full): removes ~20mb/d, would spike Brent $30-50 immediately
- Hormuz partial disruption: $10-20 spike, depending on tanker rerouting capacity
- Abqaiq-style Saudi infrastructure attack: removes 5-10mb/d, $20-30 spike
- OPEC surprise cut of 1mb/d: typically adds $3-5 to Brent within days
- Iran sanctions restored fully: removes ~1.5mb/d, adds $5-8 to Brent
- Peace deal removing sanctions on major producer: bearish $5-10 once confirmed

### Demand signals
- India/China demand growth slowdown = bearish, but gradual
- US SPR release = short-term bearish, typically $2-4 impact
- US SPR purchase = bullish signal (government sees value at current prices)
- IEA demand forecast cut = bearish sentiment, not always immediate price impact

### Common LLM errors to avoid
- Do NOT say "lower investment is bearish" — it is bullish for prices
- Do NOT attribute Brent-WTI spread moves to "global demand" — it is a US-specific supply/logistics signal
- Do NOT conflate "oil company profits down" with "oil prices going down" — they can diverge
- Do NOT say "rising dollar is bullish for oil" — it is bearish (oil is dollar-denominated)
"""

APPENDIX_TEMPLATE = """
---

## Data Appendix

### Price Data
| Metric | Value | Period | Source |
|--------|-------|--------|--------|
| Brent (article-quoted) | {art_brent} | — | OilPrice.com (ingested) |
| WTI (article-quoted) | {art_wti} | — | OilPrice.com (ingested) |
| Brent (EIA) | ${brent_price} | {brent_period} ({eia_brent_age}) | {brent_source} |
| WTI (EIA) | ${wti_price} | {wti_period} ({eia_wti_age}) | {wti_source} |
| WTI 5-day trend | {wti_trend} | | EIA |
| Brent 5-day trend | {brent_trend} | | EIA |

### Prediction Signal — 12 Hour
| Signal | Score | Weight |
|--------|-------|--------|
| News Sentiment 24h | {sent_score} | 20% |
| GDELT Tone 24h | {gdelt_24h_score} | 20% |
| GDELT Trend 7d | {gdelt_7d_score} | 10% |
| EIA Price Momentum | {eia_score} | 20% |
| EIA Inventory | {inv_score} | 10% |
| Geopolitical Risk | {geo_score} | 15% |
| Supply Disruption | {disruption_score} | 5% |
| **Composite** | **{composite_score}** | |
| **Direction 12h** | **{direction}** | |
| **Confidence** | **{confidence}** | |

### 3-Day Outlook
| | |
|---|---|
| Direction | {direction_3d} |
| Confidence | {confidence_3d} |
| Score | {score_3d} |
| Rationale | {rationale_3d} |

### Geopolitical Risk
| | |
|---|---|
| Level | {geo_level} |
| Triggers | {geo_triggers} |
| Supply Disruption | {disruption} |
| Inventory Signal | {inventory_signal} |

### News Sentiment
- Bullish articles: {bullish}
- Bearish articles: {bearish}
- Neutral/Unclear: {neutral_unclear}
- GDELT tone score: {gdelt_tone} ({gdelt_records} articles analysed)

### Top Mentioned Organisations
{top_orgs}

### Most Relevant News
{relevant_news}

### Recent Extracted Events
{recent_events}

### EIA Market Data
- US Crude Inventory (latest weekly): from EIA series PET.WCESTUS1.W
- US Crude Production (latest weekly): from EIA series PET.WCRFPUS2.W

*Report generated: {generated_at}*
"""


def _format_orgs(orgs: list[dict]) -> str:
    if not orgs:
        return "No organisation data available."
    return "\n".join(f"- {o['organisation']}: {o['mentions']} mentions" for o in orgs)


def _format_news(results: list[dict]) -> str:
    if not results:
        return "No relevant news found."
    lines = []
    for r in results:
        title  = r["metadata"].get("title", "").strip()
        source = r["metadata"].get("source", "")
        score  = r["score"]
        chunk  = r["chunk"][:600].strip()   # enough for LLM to extract detail
        if title:
            lines.append(f"- [{source}] {title} (relevance: {score:.2f})\n  {chunk}")
        else:
            lines.append(f"- [{source}] {chunk[:400]} (relevance: {score:.2f})")
    return "\n".join(lines)


def _format_events(events: list[dict]) -> str:
    if not events:
        return "No events extracted yet."
    lines = []
    for e in events:
        lines.append(f"- [{e.get('type','unknown')}] {e.get('description','')} (from: {e.get('article','')[:60]})")
    return "\n".join(lines)


def _load_previous_report(n: int = 1) -> str:
    """
    Load the body of the nth most recent report (default: last report).
    Returns empty string if no previous report exists.
    Strips the YAML frontmatter and data appendix — just the narrative.
    """
    try:
        reports = sorted(REPORT_DIR.glob("*.md"), reverse=True)
        if len(reports) <= n - 1:
            return ""
        prev = reports[n - 1]
        text = prev.read_text(encoding="utf-8")
        # Strip YAML frontmatter
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:].strip()
        # Strip data appendix
        if "## Data Appendix" in text:
            text = text[:text.index("## Data Appendix")].strip()
        # Cap length — don't blow out the context window
        return text[:2000]
    except Exception:
        return ""


def _extract_high_importance_events(enriched_articles: list[dict]) -> tuple[str, str]:
    """
    Scan extraction results for high-importance and breaking events.
    Returns (high_importance_str, breaking_news_str) for the prompt.

    High importance = importance_score >= 0.7
    Breaking = is_breaking flag set true by the extractor
    Sorted by importance_score descending so the most critical leads.
    """
    high_events = []
    breaking    = []

    now = datetime.now(timezone.utc)

    for article in enriched_articles:
        title      = article.get("title", "")
        source     = article.get("source", "")
        published  = article.get("published") or article.get("fetched_at", "")

        # Age label
        age_label = ""
        if published:
            try:
                pub = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_h = (now - pub).total_seconds() / 3600
                age_label = f"{int(age_h)}h ago" if age_h < 48 else f"{int(age_h/24)}d ago"
            except (ValueError, TypeError):
                pass

        for chunk in article.get("extraction", []):
            if chunk.get("status") != "ok":
                continue

            importance = float(chunk.get("importance_score", 0.0))
            reason      = chunk.get("importance_reason", "")
            is_breaking = chunk.get("is_breaking", False)
            hormuz      = chunk.get("hormuz_risk", False)
            sanctions   = chunk.get("sanctions_event", False)

            flags = []
            if hormuz:    flags.append("⚠ HORMUZ RISK")
            if sanctions: flags.append("SANCTIONS")
            if chunk.get("supply_disruption"):    flags.append("SUPPLY DISRUPTION")
            if chunk.get("logistics_disruption"): flags.append("LOGISTICS DISRUPTION")

            # Watchlist boost — user-defined topics always surface
            text = " ".join(filter(None, [article.get("title",""), article.get("summary","")])).lower()
            watchlist_hits = [kw for kw in USER_WATCHLIST if kw.lower() in text]
            if watchlist_hits:
                boost = min(WATCHLIST_BOOST, len(watchlist_hits) * (WATCHLIST_BOOST / 2))
                importance = min(1.0, importance + boost)
                flags.extend([f"WATCHLIST:{kw}" for kw in watchlist_hits[:2]])

            flag_str  = f" [{', '.join(flags)}]" if flags else ""
            age_str   = f" ({age_label})" if age_label else ""
            source_str = f"[{source}] " if source else ""

            line = f"- importance {importance:.2f}{flag_str} — {source_str}{title}{age_str}"
            if reason:
                line += f"\n  Reason: {reason}"

            if importance >= 0.7:
                high_events.append((importance, line))

            if is_breaking and importance >= 0.5:
                breaking.append((importance, line))

    # Sort by importance descending
    high_events.sort(key=lambda x: x[0], reverse=True)
    breaking.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate (breaking events often also high-importance)
    high_lines     = [l for _, l in high_events[:6]]
    breaking_lines = [l for _, l in breaking[:4]]

    return (
        "\n".join(high_lines) if high_lines else "None detected in this run.",
        "\n".join(breaking_lines) if breaking_lines else "None detected in this run.",
    )


def _compute_day_magnitude(enriched_articles: list[dict],
                           is_alert: bool = False) -> tuple[str, str]:
    """
    Aggregate the day's boosted importance scores into a magnitude tier
    that drives report length and tone. Reuses the same per-chunk scoring
    (importance_score + watchlist boost, Hormuz/sanctions flags) as
    _extract_high_importance_events, so no new scoring is introduced.

    Tiers (the SCORE sets the floor; the LLM refines within it):
      loud   — is_alert, OR any Hormuz/sanctions flag, OR >=2 events >=0.7
      normal — at least one event >=0.5 but not loud
      quiet  — nothing material

    Returns (tier, lead_line) where lead_line is the single highest-importance
    event line, used to anchor the headline/takeaway.
    """
    high_count = 0
    has_critical_flag = False
    best_importance = 0.0
    best_line = ""

    for article in enriched_articles or []:
        title  = article.get("title", "")
        source = article.get("source", "")
        text   = " ".join(filter(None, [title, article.get("summary", "")])).lower()
        watchlist_hits = [kw for kw in USER_WATCHLIST if kw.lower() in text]

        for chunk in article.get("extraction", []):
            if chunk.get("status") != "ok":
                continue

            importance = float(chunk.get("importance_score", 0.0))
            if watchlist_hits:
                boost = min(WATCHLIST_BOOST, len(watchlist_hits) * (WATCHLIST_BOOST / 2))
                importance = min(1.0, importance + boost)

            if (chunk.get("hormuz_risk") or chunk.get("sanctions_event")
                    or chunk.get("supply_disruption") or chunk.get("logistics_disruption")):
                has_critical_flag = True

            if importance >= 0.7:
                high_count += 1

            if importance > best_importance:
                best_importance = importance
                src = f"[{source}] " if source else ""
                best_line = f"{src}{title}".strip()

    # Determine tier — score sets the floor
    if is_alert or has_critical_flag or high_count >= 2:
        tier = "loud"
    elif best_importance >= 0.5:
        tier = "normal"
    else:
        tier = "quiet"

    return tier, (best_line or "No single dominant story.")


_THESIS_PROMPT = """You are a chief oil-trading adviser. Below are the day's
ranked market events (already sorted by importance — the scores PROPOSE the
ranking; you may REFINE it if your judgement differs, like a desk head would).

Ranked events:
{events}

Decide the SINGLE organising read for today's brief, so the whole report speaks
in one coherent voice rather than listing disconnected facts. Return ONLY valid
JSON, no markdown:
{{
  "thesis": "<one sentence: the central read of the day>",
  "structure": "<'weave' if the events are related and should flow as one
     connected read, or 'rank' if they are unrelated and should be presented
     dominant-first then briefly>",
  "ranked_events": ["<event 1 (most important)>", "<event 2>", "..."],
  "connecting_read": "<one sentence on how the events relate — do they reinforce,
     offset, or are they independent? which dominates and why?>"
}}

Apply seasoned market judgement:
- Distinguish SENTIMENT moves (headline/political reactions that often outrun
  fundamentals and tend to retrace) from SUBSTANCE moves (real supply/demand
  shifts). Say which a move looks like.
- NEVER characterise a named real person (their competence, motives, depth).
  Read the MARKET'S BEHAVIOUR instead: e.g. "the market is reacting to a
  political headline that doesn't change supply" — not "X's shallow remarks".
- If only one event matters, thesis = that event; structure = 'weave'.
"""


def _generate_thesis(high_importance_str: str, breaking_str: str) -> dict:
    """Pass 1 of two-pass generation: decide the day's single organising read
    and how to structure multiple events (weave vs rank). Falls back to an
    empty thesis if the call fails, so the main report still generates."""
    events = (high_importance_str or "").strip() or (breaking_str or "").strip()
    if not events:
        return {"thesis": "", "structure": "weave", "ranked_events": [],
                "connecting_read": ""}
    try:
        raw = _call_llm(_THESIS_PROMPT.format(events=events))
        cleaned = raw.strip()
        if "```" in cleaned:
            for part in cleaned.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"Thesis pass failed ({e}); proceeding without it.")
        return {"thesis": "", "structure": "weave", "ranked_events": [],
                "connecting_read": ""}


def _format_thesis_block(thesis: dict) -> str:
    """Render the thesis as a prompt instruction block for pass 2."""
    if not thesis.get("thesis"):
        return ""
    struct = thesis.get("structure", "weave")
    struct_instruction = (
        "These events are RELATED — weave them into one connected read, showing "
        "how they relate (reinforce / offset), not as a list."
        if struct == "weave" else
        "These events are UNRELATED — lead with the dominant one, then cover the "
        "others briefly. Keep them ranked, most important first."
    )
    ranked = "\n".join(f"  {i+1}. {e}" for i, e in enumerate(thesis.get("ranked_events", [])))
    return (
        "## TODAY'S ORGANISING READ (write the whole report to serve this — "
        "one coherent voice, every statement supporting or testing the thesis)\n"
        f"THESIS: {thesis['thesis']}\n"
        f"HOW EVENTS RELATE: {thesis.get('connecting_read','')}\n"
        f"STRUCTURE: {struct_instruction}\n"
        + (f"RANKED EVENTS:\n{ranked}\n" if ranked else "")
        + "Carry this single read through both the promoter brief and the analyst "
          "note. Frame actor-driven moves as market behaviour (sentiment vs "
          "substance), never as judgements of a named person.\n"
    )


# Per-tier delivery instructions. These govern the ANALYST NOTE (Section 2) ONLY.
# The PROMOTER BRIEF (Section 1) is ALWAYS ~150 words regardless of tier.
_DELIVERY_BLOCKS = {
    "quiet": """## DELIVERY MODE: QUIET DAY
Applies to the ANALYST NOTE (Section 2) only. The promoter brief stays ~150 words.
- Nothing materially moved the market today. The analyst note should be SHORT too:
  3-5 sentences. Do not pad. State the steady state and one thing to watch.
- The promoter brief's headline should plainly say it's a quiet day, e.g.
  "Quiet day — nothing to act on; crude steady."
- Do NOT manufacture urgency. Calm days reported calm — this is what makes the
  loud days credible.""",

    "normal": """## DELIVERY MODE: NORMAL DAY
Applies to the ANALYST NOTE (Section 2) only. The promoter brief stays ~150 words.
- Analyst note: ~250-350 words, the day's real driver and what it means.
- Cover the genuine driver; don't force coverage of minor items.
- Sharp, evidence-led voice in the analyst note; business-translated in the brief.""",

    "loud": """## DELIVERY MODE: LOUD DAY — BIG NEWS
Applies to the ANALYST NOTE (Section 2) only. The promoter brief STILL stays ~150
words — a big news day does NOT make the promoter brief longer, it makes it
SHARPER. The owner still has 20 seconds.
- Analyst note: fuller, up to 500-700 words MAXIMUM for the adviser's own use.
  Lead hard with the single most important thing, its price impact, and what it
  means for our crude cost.
- Promoter brief: lead with the one big thing in plain business terms, one
  carry-away, ~150 words. Urgency must be EARNED by the data, never manufactured.""",
}


def _extract_article_prices(enriched_articles: list[dict]) -> dict:
    """
    Extract the most recent Brent and WTI price mentions from already-ingested
    article key_figures. OilPrice.com articles embed live ticker prices in their
    text; the extractor captures these as key_figures with USD/barrel units.

    Returns: {
      'brent': {'value': float, 'fetched_at': str, 'source': str} | None,
      'wti':   {'value': float, 'fetched_at': str, 'source': str} | None,
    }
    Uses the article's fetched_at timestamp so the staleness label is accurate.
    """
    candidates = {"brent": [], "wti": []}
    for a in enriched_articles or []:
        fetched = a.get("fetched_at") or a.get("published", "")
        src = a.get("source", "")
        for c in a.get("extraction", []) or []:
            if c.get("status") != "ok":
                continue
            for f in c.get("key_figures", []) or []:
                val  = f.get("value", "")
                unit = str(f.get("unit",    "")).lower()
                ctx  = str(f.get("context", "")).lower()
                try:
                    v = float(str(val).replace(",", ""))
                except (ValueError, TypeError):
                    continue
                # plausible crude price: USD/barrel range only
                if not (50 <= v <= 200):
                    continue
                if "barrel" not in unit and "bbl" not in unit:
                    continue
                if "brent" in ctx:
                    candidates["brent"].append(
                        {"value": v, "fetched_at": fetched, "source": src})
                elif "wti" in ctx or "west texas" in ctx or ("crude" in ctx and "brent" not in ctx):
                    candidates["wti"].append(
                        {"value": v, "fetched_at": fetched, "source": src})
    result = {}
    for k, v in candidates.items():
        if v:
            result[k] = sorted(v, key=lambda x: x["fetched_at"], reverse=True)[0]
    return result


def _price_age_label(period_or_ts: str, now: datetime) -> str:
    """Return a human-readable age label for a price timestamp."""
    if not period_or_ts:
        return "age unknown"
    try:
        dt = datetime.fromisoformat(str(period_or_ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (now - dt).total_seconds() / 3600
        if hours < 1:
            return f"{int(hours*60)}min ago"
        elif hours < 24:
            return f"{hours:.0f}h ago"
        else:
            return f"{hours/24:.0f}d ago"
    except (ValueError, TypeError):
        # EIA period is a date string like "2026-06-01"
        try:
            dt = datetime.strptime(str(period_or_ts), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            hours = (now - dt).total_seconds() / 3600
            return f"{hours/24:.0f}d ago"
        except (ValueError, TypeError):
            return period_or_ts
    if ANTHROPIC_API_KEY and USE_CLAUDE_REPORT:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 3000,
                "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()
    else:
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4, "num_predict": 3000}
        }
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


def generate_report(prediction: dict, kg: KnowledgeGraph,
                    enriched_articles: list[dict] = None,
                    is_alert: bool = False) -> Path | None:
    """
    Generate a markdown analyst report and save it to REPORT_DIR.
    Returns the path of the saved report, or None if the cycle-health guard
    blocked publication (empty/stale/failed cycle — nothing is sent).
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cycle-health guard ─────────────────────────────────────────────────
    # If the cycle is empty/stale/failed, log the reason and SEND NOTHING.
    # Silence is safer than a wrong or stale report reaching the SLT.
    # (is_alert reports bypass the freshness/volume gate — an alert is itself
    #  the signal — but still log health for the record.)
    try:
        from guard import check_cycle_health, log_verdict
        _cycle = {"articles": enriched_articles or []}
        _verdict = check_cycle_health(_cycle)
        log_verdict(_verdict)
        if not _verdict["publish"] and not is_alert:
            logger.error("Cycle failed health guard — no report generated or sent this cycle.")
            return None
    except Exception as e:
        # Guard must never itself block a run by crashing — log and continue.
        logger.warning(f"Cycle-health guard unavailable ({e}); proceeding without it.")

    # ── Pull data for the prompt ───────────────────────────────────────────
    signals    = prediction.get("signals", {})
    eia        = signals.get("eia", {})
    sent       = signals.get("sentiment_24h", {})
    gdelt      = signals.get("gdelt_tone_24h", {})
    geo        = signals.get("geopolitical", {})
    disruption = signals.get("disruption", {})
    inv        = signals.get("eia_inventory", {})
    outlook_3d = prediction.get("outlook_3d", {})

    wti_latest   = eia.get("wti_latest", {})
    brent_latest = eia.get("brent_latest", {})

    # ── Article-quoted prices (OilPrice.com ticker via existing ingestion) ───
    # These are extracted from already-ingested articles — no new source,
    # no new subscription. Fresher than EIA; labelled with actual age so
    # the adviser can judge at a glance whether they need updating.
    _now = datetime.now(timezone.utc)
    art_prices = _extract_article_prices(enriched_articles or [])
    _ap_brent  = art_prices.get("brent")
    _ap_wti    = art_prices.get("wti")

    # Build the price reference line shown in the report header and prompt.
    # Two rows: article-quoted (fresher) and EIA (authoritative, dated).
    _eia_brent_age = _price_age_label(brent_latest.get("period",""), _now)
    _eia_wti_age   = _price_age_label(wti_latest.get("period",""),   _now)
    _art_brent_str = (
        f"${_ap_brent['value']:.2f} ({_price_age_label(_ap_brent['fetched_at'], _now)}, "
        f"{_ap_brent['source']})"
        if _ap_brent else "not available this cycle"
    )
    _art_wti_str = (
        f"${_ap_wti['value']:.2f} ({_price_age_label(_ap_wti['fetched_at'], _now)}, "
        f"{_ap_wti['source']})"
        if _ap_wti else "not available this cycle"
    )

    top_orgs      = kg.query_top_organisations(limit=8)
    recent_events = kg.query_recent_events(limit=8)

    # Semantic search — queries built from active signal mechanisms, not hardcoded topics
    store = VectorStore()

    geo        = signals.get("geopolitical", {})
    disruption = signals.get("disruption", {})
    inv        = signals.get("eia_inventory", {})

    # Build primary query from what the signals say is actually moving the market
    query_terms = ["oil price"]
    if geo.get("supply_cut_hits", 0) > 0:
        query_terms.append("supply disruption shutdown attack export halt")
    if geo.get("hormuz_boost", 0) > 0:
        query_terms.append("Hormuz strait closure blockade")
    if geo.get("escalation_hits", 0) > 0:
        query_terms.append("escalation military conflict retaliation")
    if geo.get("sanctions_boost", 0) > 0:
        query_terms.append("sanctions embargo oil producer")
    if geo.get("opec_boost", 0) > 0:
        query_terms.append("OPEC production decision output cut")
    if geo.get("supply_add_hits", 0) > 0:
        query_terms.append("supply increase ceasefire peace deal sanctions relief")
    if inv.get("signal") == "draw_bullish":
        query_terms.append("inventory draw crude stockpile decline")
    if disruption.get("detected"):
        query_terms.append("force majeure production shutdown pipeline")
    if len(query_terms) == 1:
        # No strong signal — use broad oil market query
        query_terms.append("OPEC geopolitical risk supply demand outlook")

    primary_query   = " ".join(query_terms)
    recent_news     = store.search_important(primary_query, n_results=6)
    background_news = store.search_important(
        "oil market trend producer country export shipping tanker demand outlook",
        n_results=4
    )
    relevant_news   = recent_news + background_news

    # High-importance and breaking events from enriched extraction
    high_importance_str, breaking_news_str = _extract_high_importance_events(enriched_articles or [])

    # Previous report for continuity
    prev_report = _load_previous_report()

    # Brent-WTI spread
    brent_val = brent_latest.get("value")
    wti_val   = wti_latest.get("value")
    spread    = round(float(brent_val) - float(wti_val), 2) if brent_val and wti_val else "N/A"

    # ── Magnitude tier — drives report length/tone (score sets floor) ───────
    day_tier, lead_line = _compute_day_magnitude(enriched_articles or [], is_alert=is_alert)
    logger.info(f"Day magnitude tier: {day_tier} (lead: {lead_line[:80]})")
    delivery_block = _DELIVERY_BLOCKS.get(day_tier, _DELIVERY_BLOCKS["normal"])

    # ── Thesis pass (pass 1 of 2) — one organising read for coherence ───────
    # Decides the day's single thesis and how to structure multiple events
    # (weave if related, rank if not). Scores proposed the ranking; the LLM
    # refines it. Keeps the whole report in one coherent voice.
    #
    # OFF by default during the trial period: set USE_THESIS_PASS=true to enable.
    # While off, the existing (known-good) generation is unchanged AND no extra
    # LLM call is made — so it costs nothing until you turn it on.
    thesis_block = ""
    if os.getenv("USE_THESIS_PASS", "false").lower() == "true":
        thesis = _generate_thesis(high_importance_str, breaking_news_str)
        if thesis.get("thesis"):
            logger.info(f"Thesis: {thesis['thesis'][:90]} | structure={thesis.get('structure')}")
        thesis_block = _format_thesis_block(thesis)
    else:
        logger.info("Thesis pass: OFF (set USE_THESIS_PASS=true to enable)")

    # ── Build prompt ───────────────────────────────────────────────────────
    now_london = datetime.now(timezone.utc).strftime("%d/%m/%Y, %I%p, London")

    prompt = REPORT_PROMPT.format(
        date_time           = now_london,
        direction           = prediction.get("direction", "neutral"),
        confidence          = prediction.get("confidence", "low"),
        score               = prediction.get("score", 0.0),
        direction_3d        = outlook_3d.get("direction", "neutral"),
        confidence_3d       = outlook_3d.get("confidence", "low"),
        score_3d            = outlook_3d.get("score", 0.0),
        rationale_3d        = outlook_3d.get("rationale", ""),
        geo_level           = geo.get("level", "minimal"),
        geo_triggers        = ", ".join(geo.get("triggered_by", [])) or "none",
        disruption          = "YES — " + ", ".join(disruption.get("triggers", [])) if disruption.get("detected") else "No",
        inventory_signal    = inv.get("signal", "no_data"),
        wti_price           = wti_latest.get("value", "N/A"),
        wti_period          = wti_latest.get("period", "N/A"),
        wti_source          = wti_latest.get("source", "EIA").upper(),
        brent_price         = brent_latest.get("value", "N/A"),
        brent_period        = brent_latest.get("period", "N/A"),
        brent_source        = brent_latest.get("source", "EIA").upper(),
        art_brent           = f"${_ap_brent['value']:.2f} ({_price_age_label(_ap_brent['fetched_at'], _now)}, {_ap_brent['source']})" if _ap_brent else "not available this cycle",
        art_wti             = f"${_ap_wti['value']:.2f} ({_price_age_label(_ap_wti['fetched_at'], _now)}, {_ap_wti['source']})" if _ap_wti else "not available this cycle",
        eia_brent_age       = _eia_brent_age,
        eia_wti_age         = _eia_wti_age,
        wti_trend           = eia.get("wti_trend", 0.0),
        brent_trend         = eia.get("brent_trend", 0.0),
        bullish_w           = sent.get("bullish_w", 0),
        bearish_w           = sent.get("bearish_w", 0),
        gdelt_tone          = gdelt.get("avg_tone", "N/A"),
        gdelt_records       = gdelt.get("records", 0),
        high_importance_events = high_importance_str,
        breaking_news       = breaking_news_str,
        top_orgs            = _format_orgs(top_orgs),
        relevant_news       = _format_news(relevant_news),
        recent_events       = _format_events(recent_events),
    )

    # Prepend domain knowledge so the LLM applies correct oil market logic,
    # the thesis block so the report stays coherent around one read, and the
    # delivery block so length/tone flex with the day's magnitude.
    prompt = delivery_block + "\n\n" + thesis_block + "\n\n" + OIL_MARKET_KNOWLEDGE + "\n\n" + prompt

    # ── Call Ollama ────────────────────────────────────────────────────────
    logger.info(f"Generating report via {'Claude API' if ANTHROPIC_API_KEY else 'Ollama'}...")
    report_text = _call_llm(prompt)

    # Split the two audience sections on the separator the prompt requested.
    # Render a clear labelled divider so the promoter brief (top) is visually
    # distinct from the analyst note (below, for the adviser's own use).
    if "===ANALYST===" in report_text:
        brief_part, analyst_part = report_text.split("===ANALYST===", 1)
        report_text = (
            brief_part.rstrip()
            + "\n\n---\n\n"
            + "## 📊 Analyst Note — *adviser detail (not for distribution as-is)*\n\n"
            + analyst_part.lstrip()
        )
    else:
        logger.warning("Report missing ===ANALYST=== separator — single-section output.")

    # ── Emerging-story scan (proposes new themes to the adviser; never auto-applies) ─
    focus_note = ""
    try:
        from focus.emerging import scan_emerging_stories
        focus_note = scan_emerging_stories(enriched_articles or [])
    except Exception as e:
        logger.warning(f"Emerging-story scan unavailable: {e}")

    # ── Save report ────────────────────────────────────────────────────────
    now      = datetime.now(timezone.utc)
    prefix   = "ALERT_" if is_alert else ""
    filename = now.strftime(f"{prefix}%Y-%m-%d_%H-%M-%S_report.md")
    filepath = REPORT_DIR / filename

    alert_banner = "\n> ⚠ **ALERT REPORT** — High geopolitical risk detected. Sent outside normal schedule.\n" \
                   if is_alert else ""

    # First non-empty line of the report is the conclusion-carrying headline.
    # Expose it as `subject:` in the frontmatter for the email sender to use.
    subject_line = ""
    for _line in report_text.splitlines():
        _stripped = _line.strip().lstrip("#").strip()
        if _stripped:
            subject_line = _stripped[:120]
            break
    if not subject_line:
        subject_line = f"Oil Brief — {now.strftime('%d/%m/%Y')}"
    # YAML-safe: double-quote the value and escape backslashes/quotes so that
    # any special leading character (*, &, :, #, {, [, |, >, @, etc.) in a
    # headline is treated as a literal string, not YAML syntax.
    _safe_subject = subject_line.replace("\\", "\\\\").replace('"', '\\"')

    header = (
        f"---\n"
        f"generated_at: {now.isoformat()}\n"
        f"alert: {is_alert}\n"
        f"magnitude: {day_tier}\n"
        f'subject: "{_safe_subject}"\n'
        f"direction: {prediction.get('direction')}\n"
        f"confidence: {prediction.get('confidence')}\n"
        f"score: {prediction.get('score')}\n"
        f"wti: {wti_latest.get('value')} ({wti_latest.get('period')})\n"
        f"brent: {brent_latest.get('value')} ({brent_latest.get('period')})\n"
        f"---\n"
        f"{alert_banner}\n"
    )
    # Build data appendix
    signals = prediction.get("signals", {})
    sent    = signals.get("sentiment", {})
    eia     = signals.get("eia", {})
    gdelt   = signals.get("gdelt", {})

    gdelt_24h = signals.get("gdelt_tone_24h", {})
    gdelt_7d  = signals.get("gdelt_trend_7d", {})
    inv       = signals.get("eia_inventory", {})
    geo       = signals.get("geopolitical", {})
    dis       = signals.get("disruption", {})
    sent      = signals.get("sentiment_24h", {})

    appendix = APPENDIX_TEMPLATE.format(
        brent_price      = brent_latest.get("value", "N/A"),
        brent_period     = brent_latest.get("period", "N/A"),
        brent_source     = brent_latest.get("source", "EIA").upper(),
        wti_price        = wti_latest.get("value", "N/A"),
        wti_period       = wti_latest.get("period", "N/A"),
        wti_source       = wti_latest.get("source", "EIA").upper(),
        art_brent        = f"${_ap_brent['value']:.2f}" if _ap_brent else "N/A",
        art_wti          = f"${_ap_wti['value']:.2f}"   if _ap_wti   else "N/A",
        eia_brent_age    = _eia_brent_age,
        eia_wti_age      = _eia_wti_age,
        wti_trend        = eia.get("wti_trend", "N/A"),
        brent_trend      = eia.get("brent_trend", "N/A"),
        sent_score       = sent.get("score", "N/A"),
        gdelt_24h_score  = gdelt_24h.get("score", "N/A"),
        gdelt_7d_score   = gdelt_7d.get("score", "N/A"),
        eia_score        = eia.get("score", "N/A"),
        inv_score        = inv.get("score", "N/A"),
        geo_score        = geo.get("score", "N/A"),
        disruption_score = dis.get("score", "N/A"),
        composite_score  = prediction.get("score", "N/A"),
        direction        = prediction.get("direction", "neutral").upper(),
        confidence       = prediction.get("confidence", "low").upper(),
        direction_3d     = outlook_3d.get("direction", "neutral").upper(),
        confidence_3d    = outlook_3d.get("confidence", "low").upper(),
        score_3d         = outlook_3d.get("score", "N/A"),
        rationale_3d     = outlook_3d.get("rationale", ""),
        geo_level        = geo.get("level", "minimal"),
        geo_triggers     = ", ".join(geo.get("triggered_by", [])) or "none",
        disruption       = "YES — " + ", ".join(dis.get("triggers", [])) if dis.get("detected") else "No",
        inventory_signal = inv.get("signal", "no_data"),
        bullish          = sent.get("bullish", 0),
        bearish          = sent.get("bearish", 0),
        neutral_unclear  = sent.get("neutral", 0) + sent.get("unclear", 0),
        gdelt_tone       = gdelt_24h.get("avg_tone", "N/A"),
        gdelt_records    = gdelt_24h.get("records", 0),
        top_orgs         = _format_orgs(top_orgs),
        relevant_news    = _format_news(relevant_news),
        recent_events    = _format_events(recent_events),
        generated_at     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    # ── Assessed-price slots — adviser fills these from Argus/Platts at review ─
    # Placed at the very top so it's the first thing the adviser sees and cannot
    # miss before forwarding to the SLT. EIA numbers shown for reference only.
    assessed_block = (
        "\n> **⚠ ADVISER ACTION — confirm assessed prices before sending.**\n>\n"
        "> | Price | Article-quoted (OilPrice.com) | EIA reference | Assessed (Argus/Platts) |\n"
        "> |---|---|---|---|\n"
        f"> | Brent | {_art_brent_str} | ${brent_latest.get('value','N/A')} ({_eia_brent_age}) | __________ |\n"
        f"> | WTI | {_art_wti_str} | ${wti_latest.get('value','N/A')} ({_eia_wti_age}) | __________ |\n"
        "> | Dated Brent | — | — | __________ |\n"
        "> | Ebob gasoline | — | — | __________ |\n"
        "> | Gasoil/diesel crack | — | — | __________ |\n"
        ">\n"
        "> *Article prices: fresher, sourced from ingested news text. "
        "EIA: authoritative open data, age shown. "
        "Assessed: enter from Argus/Platts terminal.*\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + assessed_block + report_text + appendix + focus_note)

    logger.info(f"Report saved: {filepath}")

    # Auto-convert to docx if pandoc is available
    try:
        import subprocess, shutil

        # cron runs with minimal PATH — search common install locations explicitly
        pandoc_cmd = shutil.which("pandoc") or next(
            (p for p in [
                "/usr/local/bin/pandoc",
                "/opt/homebrew/bin/pandoc",
                "/usr/bin/pandoc",
                "/home/linuxbrew/.linuxbrew/bin/pandoc",
            ] if Path(p).exists()),
            None
        )

        if not pandoc_cmd:
            logger.info("Pandoc not found in PATH or known locations — skipping DOCX conversion")
        else:
            docx_path = filepath.with_suffix(".docx")
            result = subprocess.run(
                [pandoc_cmd, str(filepath), "-o", str(docx_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                logger.info(f"DOCX saved: {docx_path}")
            else:
                logger.warning(f"Pandoc failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"DOCX conversion error: {e}")

    return filepath