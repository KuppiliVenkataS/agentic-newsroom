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

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REPORT_DIR, USER_WATCHLIST, WATCHLIST_BOOST

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
from vectordb.store import VectorStore
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are an experienced oil trader writing your morning briefing note to trading desk colleagues.
Write exactly like the example style below — direct, opinionated, personal. First person.
No formal headers. No bullet point lists. Flowing paragraphs only.

Today's date and time: {date_time}

## Price Data
Brent: {brent_price} USD/barrel (as of {brent_period}, source: {brent_source})
WTI: {wti_price} USD/barrel (as of {wti_period}, source: {wti_source})
WTI 5-day trend score: {wti_trend} (positive = rising)
Brent 5-day trend score: {brent_trend}

## Market Signal
Overall direction: {direction} (confidence: {confidence}, score: {score})
News sentiment — Bullish weight: {bullish_w}, Bearish weight: {bearish_w} (importance-weighted, not raw count)
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

Writing instructions — follow these exactly:

STRUCTURE:
1. First line: date, time, location (e.g. "28/05/2026, 5AM, London")
2. Second line: Brent and WTI prices, and any key spread or structure data if available
3. Body: 3-5 flowing paragraphs, no headers, no bullet points

CONTENT PRIORITY — strictly in this order:
1. If HIGH-IMPORTANCE EVENTS exist, open the first paragraph with the most significant one — name the specific event, location, actors. Never bury a military strike or Hormuz closure.
2. Connect that event directly to price impact with your own view ("I think this could add $X to Brent")
3. Then cover other geopolitical threads separately — do not merge Iran, Russia, Israel into one vague paragraph
4. Price action and market structure (backwardation/contango, spreads) — only if data supports it
5. Your personal 12-24h outlook — say what YOU expect and why

STYLE RULES:
- Be specific: name countries, people, locations (Iran, Hormuz, Kyiv, Netanyahu, Trump)
- Be opinionated: "I am skeptical", "my view is", "I think markets will"
- Quantify risks where possible: "a Hormuz closure could spike Brent $20-30"
- Flag sticking points and unresolved tensions explicitly — do not smooth over conflict
- If peace talks are ongoing, be skeptical by default unless data says otherwise
- Do NOT repeat the same point across paragraphs — each paragraph covers a distinct thread
- Do NOT write generic phrases like "oil markets remain volatile" or "uncertainty persists"
- Keep it under 400 words
- Do not make up prices or facts not in the data above

STALENESS CHECK:
- If the same organisation or event appears in both HIGH-IMPORTANCE EVENTS and OTHER RELEVANT STORIES, discuss it once and move on
- If an event has no new development since the last report, note it briefly and move on ("Iran talks continue, no new progress")
- Always end with your outlook for the next 12-24 hours

EXAMPLE STYLE (do not copy — match the tone):
"28/05/2026, 5AM, London
Brent at 97.8, WTI at 94.2. June/July backwardation has come down to below $3.

Israeli strikes on Southern Lebanon overnight are the main story. I think the market is underpricing the escalation risk — if Hezbollah retaliates against Gulf infrastructure, we could see a $5-8 spike quickly. Watch the next 12 hours.

Iran talks are moving but I remain skeptical. Tehran agreeing to give up enriched uranium in principle is a big concession — too big, frankly. The devil is in the details and Iran's track record on sticking points is not good. Hormuz closure probability I'd put at 15% if talks break down.

Russia hit Kyiv with Oreshnik missiles overnight, likely retaliation for the student dorm strike. This keeps the Russia risk premium in the market but I don't see a direct oil supply angle unless attacks hit Novorossiysk or Caspian infrastructure.

My view for the next 24 hours: sideways to slightly higher. Iran headline risk keeps a floor under Brent. I'd watch the $96 level — if we break below, sentiment shifts fast."
"""

APPENDIX_TEMPLATE = """
---

## Data Appendix

### Price Data
| Metric | Value | Period | Source |
|--------|-------|--------|--------|
| Brent Crude | ${brent_price} | {brent_period} | {brent_source} |
| WTI Crude | ${wti_price} | {wti_period} | {wti_source} |
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
        chunk  = r["chunk"][:200].strip()
        if title:
            lines.append(f"- [{source}] {title} (relevance: {score:.2f})\n  {chunk}")
        else:
            lines.append(f"- [{source}] {chunk[:150]} (relevance: {score:.2f})")
    return "\n".join(lines)


def _format_events(events: list[dict]) -> str:
    if not events:
        return "No events extracted yet."
    lines = []
    for e in events:
        lines.append(f"- [{e.get('type','unknown')}] {e.get('description','')} (from: {e.get('article','')[:60]})")
    return "\n".join(lines)


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

            # Watchlist boost — user-defined topics always surface
            text = " ".join(filter(None, [article.get("title",""), article.get("summary","")])).lower()
            watchlist_hits = [kw for kw in USER_WATCHLIST if kw.lower() in text]
            if watchlist_hits:
                boost = min(WATCHLIST_BOOST, len(watchlist_hits) * (WATCHLIST_BOOST / 2))
                importance = min(1.0, importance + boost)
                flags.extend([f"WATCHLIST:{kw}" for kw in watchlist_hits[:2]])
            reason     = chunk.get("importance_reason", "")
            is_breaking = chunk.get("is_breaking", False)
            hormuz      = chunk.get("hormuz_risk", False)
            sanctions   = chunk.get("sanctions_event", False)

            flags = []
            if hormuz:    flags.append("⚠ HORMUZ RISK")
            if sanctions: flags.append("SANCTIONS")

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


def _call_llm(prompt: str) -> str:
    """Use Claude API if key available, else fall back to Ollama."""
    if ANTHROPIC_API_KEY:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
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
            "options": {"temperature": 0.4, "num_predict": 2000}
        }
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


def generate_report(prediction: dict, kg: KnowledgeGraph, enriched_articles: list[dict] = None) -> Path:
    """
    Generate a markdown analyst report and save it to REPORT_DIR.
    Returns the path of the saved report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

    # ── Call Ollama ────────────────────────────────────────────────────────
    logger.info(f"Generating report via {'Claude API' if ANTHROPIC_API_KEY else 'Ollama'}...")
    report_text = _call_llm(prompt)

    # ── Save report ────────────────────────────────────────────────────────
    now      = datetime.now(timezone.utc)
    filename = now.strftime("%Y-%m-%d_%H-%M-%S_report.md")
    filepath = REPORT_DIR / filename

    # Add metadata header
    header = f"""---
generated_at: {now.isoformat()}
direction: {prediction.get('direction')}
confidence: {prediction.get('confidence')}
score: {prediction.get('score')}
wti: {wti_latest.get('value')} ({wti_latest.get('period')})
brent: {brent_latest.get('value')} ({brent_latest.get('period')})
---

"""
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

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + report_text + appendix)

    logger.info(f"Report saved: {filepath}")

    # Auto-convert to docx if pandoc is available
    try:
        import subprocess
        docx_path = filepath.with_suffix(".docx")
        result = subprocess.run(
            ["pandoc", str(filepath), "-o", str(docx_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"DOCX saved: {docx_path}")
        else:
            logger.warning(f"Pandoc failed: {result.stderr}")
    except FileNotFoundError:
        logger.info("Pandoc not installed — skipping DOCX conversion")
    except Exception as e:
        logger.warning(f"DOCX conversion error: {e}")

    return filepath