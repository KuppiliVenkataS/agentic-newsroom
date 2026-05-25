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

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REPORT_DIR
from vectordb.store import VectorStore
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are a senior oil market analyst writing a concise briefing report.

Use only the data provided below. Do not invent facts or prices not in the data.

## Prediction Signal
Direction: {direction}
Confidence: {confidence}
Composite Score: {score}

## Price Data (EIA)
WTI Latest: {wti_price} USD/barrel (as of {wti_period})
Brent Latest: {brent_price} USD/barrel (as of {brent_period})
WTI 5-day trend: {wti_trend}
Brent 5-day trend: {brent_trend}

## Sentiment from News (last run)
Bullish articles: {bullish}
Bearish articles: {bearish}
Neutral/Unclear: {neutral_unclear}

## GDELT Global News Tone
Average tone score: {gdelt_tone} (negative = bearish coverage, positive = bullish)
Articles analysed: {gdelt_records}

## Top Mentioned Organisations
{top_orgs}

## Most Relevant News (semantic search)
{relevant_news}

## Recent Events Extracted
{recent_events}

---

Write a professional oil market briefing report in markdown format with these sections:
1. Executive Summary (3-4 sentences, include the price direction call)
2. Price Overview (current prices, recent trend)
3. Market Sentiment (what the news is saying)
4. Key Organisations and Actors
5. Notable Events and Risks
6. Outlook (next 12-24 hours based on available signals)

Be concise. Use bullet points where appropriate. Do not add information not present in the data above.
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


def _call_ollama(prompt: str) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2000,
        }
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
    prompt = REPORT_PROMPT.format(
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
    logger.info("Generating report via Ollama...")
    report_text = _call_ollama(prompt)

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
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + report_text)

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