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

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REPORT_DIR

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
from vectordb.store import VectorStore
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are an experienced oil trader writing your morning briefing note.
Write exactly like the example below — conversational, direct, opinionated. First person.
No formal headers. No bullet point lists. Just flowing paragraphs like a trader would write to colleagues.

Today's date and time: {date_time}, London

## Price Data
Brent: {brent_price} USD/barrel (as of {brent_period})
WTI: {wti_price} USD/barrel (as of {wti_period})
WTI 5-day trend: {wti_trend}
Brent 5-day trend: {brent_trend}

## Market Signal
Overall direction: {direction} (confidence: {confidence}, score: {score})
News sentiment — Bullish: {bullish}, Bearish: {bearish}, Neutral/Unclear: {neutral_unclear}
GDELT global news tone: {gdelt_tone} ({gdelt_records} articles)

## Key Organisations in the News
{top_orgs}

## Most Relevant Stories
{relevant_news}

## Recent Events and Developments
{recent_events}

---

Write the morning briefing note now. Style rules:
- Angle for this report: {angle}
- Start with the date, time and location on the first line e.g. "25/05/2026, 5AM, London"
- Then immediately give Brent and WTI prices on the next line
- Write in flowing paragraphs, no headers, no bullet points
- Be opinionated — say what YOU think will happen and why
- Highlight geopolitical risks — Iran war risk, Hormuz closure probability, Russia, Ukraine, OPEC+ tensions
- If any Iran conflict, strike, or Hormuz closure news exists, lead with it — this is the single biggest upside risk to oil prices
- Quantify the Iran risk where possible e.g. 'a Hormuz closure could spike Brent by $20-30'
- Mention spreads and market structure if data supports it (backwardation, contango)
- End with your personal outlook for the next 12-24 hours
- Keep it under 400 words
- Do not make up prices or facts not in the data above
- Use phrases like "I think", "my view is", "markets will likely", "watch out for"
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

### Prediction Signal
| Signal | Score | Weight |
|--------|-------|--------|
| News Sentiment | {sent_score} | 40% |
| EIA Price Trend | {eia_score} | 40% |
| GDELT Tone | {gdelt_score} | 20% |
| **Composite** | **{composite_score}** | |
| **Direction** | **{direction}** | |
| **Confidence** | **{confidence}** | |

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


def generate_report(prediction: dict, kg: KnowledgeGraph) -> Path:
    """
    Generate a markdown analyst report and save it to REPORT_DIR.
    Returns the path of the saved report.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pull data for the prompt ───────────────────────────────────────────
    signals    = prediction.get("signals", {})
    eia        = signals.get("eia", {})
    sent       = signals.get("sentiment", {})
    gdelt      = signals.get("gdelt", {})

    wti_latest   = eia.get("wti_latest", {})
    brent_latest = eia.get("brent_latest", {})

    top_orgs     = kg.query_top_organisations(limit=8)
    recent_events= kg.query_recent_events(limit=8)

    # Semantic search for most relevant recent news
    store = VectorStore()
    relevant_news = store.search(
        "oil price forecast supply demand OPEC production",
        n_results=8
    )

    # ── Build prompt ───────────────────────────────────────────────────────
    now_london = datetime.now(timezone.utc).strftime("%d/%m/%Y, %I%p, London")

    # Rotate focus angle based on hour — forces different report structure each run
    angles = [
        "Lead with geopolitical risk and its price impact.",
        "Lead with price action and momentum, then explain the drivers.",
        "Take a contrarian view — challenge the market consensus.",
        "Focus on supply/demand fundamentals first, then geopolitics.",
    ]
    from datetime import datetime as _dt
    angle = angles[_dt.now().hour % len(angles)]

    prompt = REPORT_PROMPT.format(
        date_time    = now_london,
        angle        = angle,
        direction    = prediction.get("direction", "neutral"),
        confidence   = prediction.get("confidence", "low"),
        score        = prediction.get("score", 0.0),
        wti_price    = wti_latest.get("value", "N/A"),
        wti_period   = wti_latest.get("period", "N/A"),
        brent_price  = brent_latest.get("value", "N/A"),
        brent_period = brent_latest.get("period", "N/A"),
        wti_trend    = eia.get("wti_trend", 0.0),
        brent_trend  = eia.get("brent_trend", 0.0),
        bullish      = sent.get("bullish", 0),
        bearish      = sent.get("bearish", 0),
        neutral_unclear = sent.get("neutral", 0) + sent.get("unclear", 0),
        gdelt_tone   = gdelt.get("avg_tone", "N/A"),
        gdelt_records= gdelt.get("records", 0),
        top_orgs     = _format_orgs(top_orgs),
        relevant_news= _format_news(relevant_news),
        recent_events= _format_events(recent_events),
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

    appendix = APPENDIX_TEMPLATE.format(
        brent_price    = brent_latest.get("value", "N/A"),
        brent_period   = brent_latest.get("period", "N/A"),
        brent_source   = brent_latest.get("source", "EIA").upper(),
        wti_price      = wti_latest.get("value", "N/A"),
        wti_period     = wti_latest.get("period", "N/A"),
        wti_source     = wti_latest.get("source", "EIA").upper(),
        wti_trend      = eia.get("wti_trend", "N/A"),
        brent_trend    = eia.get("brent_trend", "N/A"),
        sent_score     = sent.get("score", "N/A"),
        eia_score      = eia.get("score", "N/A"),
        gdelt_score    = gdelt.get("score", "N/A"),
        composite_score= prediction.get("score", "N/A"),
        direction      = prediction.get("direction", "neutral").upper(),
        confidence     = prediction.get("confidence", "low").upper(),
        bullish        = sent.get("bullish", 0),
        bearish        = sent.get("bearish", 0),
        neutral_unclear= sent.get("neutral", 0) + sent.get("unclear", 0),
        gdelt_tone     = gdelt.get("avg_tone", "N/A"),
        gdelt_records  = gdelt.get("records", 0),
        top_orgs       = _format_orgs(top_orgs),
        relevant_news  = _format_news(relevant_news),
        recent_events  = _format_events(recent_events),
        generated_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
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