# """
# Report generator.

# Uses Ollama to write a structured oil market analyst report combining:
# - Price prediction signal
# - Top mentioned organisations from knowledge graph
# - Most relevant articles from vector DB semantic search
# - Latest EIA price and inventory data
# - Recent events from knowledge graph

# Output: markdown report saved to STORAGE_ROOT/reports/
# """

# import json
# import logging
# from datetime import datetime, timezone
# from pathlib import Path

# import httpx
# import os

# from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, REPORT_DIR

# ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
# from vectordb.store import VectorStore
# from graph.knowledge_graph import KnowledgeGraph

# logger = logging.getLogger(__name__)

# REPORT_PROMPT = """You are an experienced oil trader writing your morning briefing note.
# Write exactly like the example below — conversational, direct, opinionated. First person.
# No formal headers. No bullet point lists. Just flowing paragraphs like a trader would write to colleagues.

# Today's date and time: {date_time}, London

# ## Price Data
# Brent: {brent_price} USD/barrel (as of {brent_period})
# WTI: {wti_price} USD/barrel (as of {wti_period})
# WTI 5-day trend: {wti_trend}
# Brent 5-day trend: {brent_trend}

# ## Market Signal
# Overall direction: {direction} (confidence: {confidence}, score: {score})
# News sentiment — Bullish: {bullish}, Bearish: {bearish}, Neutral/Unclear: {neutral_unclear}
# GDELT global news tone: {gdelt_tone} ({gdelt_records} articles)

# ## Key Organisations in the News
# {top_orgs}

# ## Most Relevant Stories
# {relevant_news}

# ## Recent Events and Developments
# {recent_events}

# ---

# Write the morning briefing note now. Style rules:
# - Angle for this report: {angle}
# - Start with the date, time and location on the first line e.g. "25/05/2026, 5AM, London"
# - Then immediately give Brent and WTI prices on the next line
# - Write in flowing paragraphs, no headers, no bullet points
# - Be opinionated — say what YOU think will happen and why
# - Highlight geopolitical risks — Iran war risk, Hormuz closure probability, Russia, Ukraine, OPEC+ tensions
# - If any Iran conflict, strike, or Hormuz closure news exists, lead with it — this is the single biggest upside risk to oil prices
# - Quantify the Iran risk where possible e.g. 'a Hormuz closure could spike Brent by $20-30'
# - Mention spreads and market structure if data supports it (backwardation, contango)
# - End with your personal outlook for the next 12-24 hours
# - Keep it under 400 words
# - Do not make up prices or facts not in the data above
# - Use phrases like "I think", "my view is", "markets will likely", "watch out for"
# """

# APPENDIX_TEMPLATE = """
# ---

# ## Data Appendix

# ### Price Data
# | Metric | Value | Period | Source |
# |--------|-------|--------|--------|
# | Brent Crude | ${brent_price} | {brent_period} | {brent_source} |
# | WTI Crude | ${wti_price} | {wti_period} | {wti_source} |
# | WTI 5-day trend | {wti_trend} | | EIA |
# | Brent 5-day trend | {brent_trend} | | EIA |

# ### Prediction Signal
# | Signal | Score | Weight |
# |--------|-------|--------|
# | News Sentiment | {sent_score} | 40% |
# | EIA Price Trend | {eia_score} | 40% |
# | GDELT Tone | {gdelt_score} | 20% |
# | **Composite** | **{composite_score}** | |
# | **Direction** | **{direction}** | |
# | **Confidence** | **{confidence}** | |

# ### News Sentiment
# - Bullish articles: {bullish}
# - Bearish articles: {bearish}
# - Neutral/Unclear: {neutral_unclear}
# - GDELT tone score: {gdelt_tone} ({gdelt_records} articles analysed)

# ### Top Mentioned Organisations
# {top_orgs}

# ### Most Relevant News
# {relevant_news}

# ### Recent Extracted Events
# {recent_events}

# ### EIA Market Data
# - US Crude Inventory (latest weekly): from EIA series PET.WCESTUS1.W
# - US Crude Production (latest weekly): from EIA series PET.WCRFPUS2.W

# *Report generated: {generated_at}*
# """


# def _format_orgs(orgs: list[dict]) -> str:
#     if not orgs:
#         return "No organisation data available."
#     return "\n".join(f"- {o['organisation']}: {o['mentions']} mentions" for o in orgs)


# def _format_news(results: list[dict]) -> str:
#     if not results:
#         return "No relevant news found."
#     lines = []
#     for r in results:
#         title  = r["metadata"].get("title", "").strip()
#         source = r["metadata"].get("source", "")
#         score  = r["score"]
#         chunk  = r["chunk"][:200].strip()
#         if title:
#             lines.append(f"- [{source}] {title} (relevance: {score:.2f})\n  {chunk}")
#         else:
#             lines.append(f"- [{source}] {chunk[:150]} (relevance: {score:.2f})")
#     return "\n".join(lines)


# def _format_events(events: list[dict]) -> str:
#     if not events:
#         return "No events extracted yet."
#     lines = []
#     for e in events:
#         lines.append(f"- [{e.get('type','unknown')}] {e.get('description','')} (from: {e.get('article','')[:60]})")
#     return "\n".join(lines)


# def _call_llm(prompt: str) -> str:
#     """Use Claude API if key available, else fall back to Ollama."""
#     if ANTHROPIC_API_KEY:
#         response = httpx.post(
#             "https://api.anthropic.com/v1/messages",
#             headers={
#                 "x-api-key": ANTHROPIC_API_KEY,
#                 "anthropic-version": "2023-06-01",
#                 "content-type": "application/json",
#             },
#             json={
#                 "model": "claude-sonnet-4-20250514",
#                 "max_tokens": 2000,
#                 "temperature": 0.4,
#                 "messages": [{"role": "user", "content": prompt}]
#             },
#             timeout=120
#         )
#         response.raise_for_status()
#         return response.json()["content"][0]["text"].strip()
#     else:
#         payload = {
#             "model":  OLLAMA_MODEL,
#             "prompt": prompt,
#             "stream": False,
#             "options": {"temperature": 0.4, "num_predict": 2000}
#         }
#         response = httpx.post(
#             f"{OLLAMA_BASE_URL}/api/generate",
#             json=payload,
#             timeout=300
#         )
#         response.raise_for_status()
#         return response.json().get("response", "").strip()


# def generate_report(prediction: dict, kg: KnowledgeGraph) -> Path:
#     """
#     Generate a markdown analyst report and save it to REPORT_DIR.
#     Returns the path of the saved report.
#     """
#     REPORT_DIR.mkdir(parents=True, exist_ok=True)

#     # ── Pull data for the prompt ───────────────────────────────────────────
#     signals    = prediction.get("signals", {})
#     eia        = signals.get("eia", {})
#     sent       = signals.get("sentiment", {})
#     gdelt      = signals.get("gdelt", {})

#     wti_latest   = eia.get("wti_latest", {})
#     brent_latest = eia.get("brent_latest", {})

#     top_orgs     = kg.query_top_organisations(limit=8)
#     recent_events= kg.query_recent_events(limit=8)

#     # Semantic search for most relevant recent news
#     store = VectorStore()
#     relevant_news = store.search(
#         "oil price forecast supply demand OPEC production",
#         n_results=8
#     )

#     # ── Build prompt ───────────────────────────────────────────────────────
#     now_london = datetime.now(timezone.utc).strftime("%d/%m/%Y, %I%p, London")

#     # Rotate focus angle based on hour — forces different report structure each run
#     angles = [
#         "Lead with geopolitical risk and its price impact.",
#         "Lead with price action and momentum, then explain the drivers.",
#         "Take a contrarian view — challenge the market consensus.",
#         "Focus on supply/demand fundamentals first, then geopolitics.",
#     ]
#     from datetime import datetime as _dt
#     angle = angles[_dt.now().hour % len(angles)]

#     prompt = REPORT_PROMPT.format(
#         date_time    = now_london,
#         angle        = angle,
#         direction    = prediction.get("direction", "neutral"),
#         confidence   = prediction.get("confidence", "low"),
#         score        = prediction.get("score", 0.0),
#         wti_price    = wti_latest.get("value", "N/A"),
#         wti_period   = wti_latest.get("period", "N/A"),
#         brent_price  = brent_latest.get("value", "N/A"),
#         brent_period = brent_latest.get("period", "N/A"),
#         wti_trend    = eia.get("wti_trend", 0.0),
#         brent_trend  = eia.get("brent_trend", 0.0),
#         bullish      = sent.get("bullish", 0),
#         bearish      = sent.get("bearish", 0),
#         neutral_unclear = sent.get("neutral", 0) + sent.get("unclear", 0),
#         gdelt_tone   = gdelt.get("avg_tone", "N/A"),
#         gdelt_records= gdelt.get("records", 0),
#         top_orgs     = _format_orgs(top_orgs),
#         relevant_news= _format_news(relevant_news),
#         recent_events= _format_events(recent_events),
#     )

#     # ── Call Ollama ────────────────────────────────────────────────────────
#     logger.info(f"Generating report via {'Claude API' if ANTHROPIC_API_KEY else 'Ollama'}...")
#     report_text = _call_llm(prompt)

#     # ── Save report ────────────────────────────────────────────────────────
#     now      = datetime.now(timezone.utc)
#     filename = now.strftime("%Y-%m-%d_%H-%M-%S_report.md")
#     filepath = REPORT_DIR / filename

#     # Add metadata header
#     header = f"""---
# generated_at: {now.isoformat()}
# direction: {prediction.get('direction')}
# confidence: {prediction.get('confidence')}
# score: {prediction.get('score')}
# wti: {wti_latest.get('value')} ({wti_latest.get('period')})
# brent: {brent_latest.get('value')} ({brent_latest.get('period')})
# ---

# """
#     # Build data appendix
#     signals = prediction.get("signals", {})
#     sent    = signals.get("sentiment", {})
#     eia     = signals.get("eia", {})
#     gdelt   = signals.get("gdelt", {})

#     appendix = APPENDIX_TEMPLATE.format(
#         brent_price    = brent_latest.get("value", "N/A"),
#         brent_period   = brent_latest.get("period", "N/A"),
#         brent_source   = brent_latest.get("source", "EIA").upper(),
#         wti_price      = wti_latest.get("value", "N/A"),
#         wti_period     = wti_latest.get("period", "N/A"),
#         wti_source     = wti_latest.get("source", "EIA").upper(),
#         wti_trend      = eia.get("wti_trend", "N/A"),
#         brent_trend    = eia.get("brent_trend", "N/A"),
#         sent_score     = sent.get("score", "N/A"),
#         eia_score      = eia.get("score", "N/A"),
#         gdelt_score    = gdelt.get("score", "N/A"),
#         composite_score= prediction.get("score", "N/A"),
#         direction      = prediction.get("direction", "neutral").upper(),
#         confidence     = prediction.get("confidence", "low").upper(),
#         bullish        = sent.get("bullish", 0),
#         bearish        = sent.get("bearish", 0),
#         neutral_unclear= sent.get("neutral", 0) + sent.get("unclear", 0),
#         gdelt_tone     = gdelt.get("avg_tone", "N/A"),
#         gdelt_records  = gdelt.get("records", 0),
#         top_orgs       = _format_orgs(top_orgs),
#         relevant_news  = _format_news(relevant_news),
#         recent_events  = _format_events(recent_events),
#         generated_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
#     )

#     with open(filepath, "w", encoding="utf-8") as f:
#         f.write(header + report_text + appendix)

#     logger.info(f"Report saved: {filepath}")

#     # Auto-convert to docx if pandoc is available
#     try:
#         import subprocess
#         docx_path = filepath.with_suffix(".docx")
#         result = subprocess.run(
#             ["pandoc", str(filepath), "-o", str(docx_path)],
#             capture_output=True, text=True, timeout=30
#         )
#         if result.returncode == 0:
#             logger.info(f"DOCX saved: {docx_path}")
#         else:
#             logger.warning(f"Pandoc failed: {result.stderr}")
#     except FileNotFoundError:
#         logger.info("Pandoc not installed — skipping DOCX conversion")
#     except Exception as e:
#         logger.warning(f"DOCX conversion error: {e}")

#     return filepath
"""
Prediction module.

Combines three independent signals into a single price direction call:

Signal 1 — Sentiment ratio (from knowledge graph)
    Ratio of bullish vs bearish articles extracted by Ollama.
    Weighted 40%.

Signal 2 — EIA price trend (from raw archive market data)
    Direction of WTI and Brent over the last 5 available data points.
    Weighted 40%.

Signal 3 — GDELT tone (from raw archive GDELT records)
    Average tone score from GDELT GKG. Negative = bearish news coverage.
    Weighted 20%.

Output: {
    "direction":   "bullish" | "bearish" | "neutral",
    "confidence":  "high" | "medium" | "low",
    "score":       float (-1.0 to 1.0),
    "signals":     { breakdown of each signal },
    "generated_at": ISO timestamp
}
"""

import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import RAW_DIR
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Signal weights — must sum to 1.0
W_SENTIMENT = 0.40
W_EIA       = 0.40
W_GDELT     = 0.20


def _sentiment_signal(kg: KnowledgeGraph) -> tuple[float, dict]:
    """
    Returns a score from -1.0 (fully bearish) to +1.0 (fully bullish).
    Uses the knowledge graph signal summary.
    """
    summary  = kg.query_signal_summary()
    bullish  = summary.get("bullish",  0)
    bearish  = summary.get("bearish",  0)
    neutral  = summary.get("neutral",  0)
    unclear  = summary.get("unclear",  0)
    total    = bullish + bearish + neutral + unclear

    if total == 0:
        return 0.0, {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0}

    # Score: bullish pushes positive, bearish pushes negative
    score = (bullish - bearish) / total

    detail = {
        "bullish":  bullish,
        "bearish":  bearish,
        "neutral":  neutral,
        "unclear":  unclear,
        "total":    total,
        "score":    round(score, 3),
    }
    return score, detail


def _eia_signal(market_data: list[dict]) -> tuple[float, dict]:
    """
    Returns a score from -1.0 to +1.0 based on WTI and Brent price trends.
    Compares most recent price to 5-period average.
    """
    def trend(series_label: str) -> float:
        points = [
            d for d in market_data
            if d.get("label") == series_label and d.get("value") is not None
        ]
        # Sort by period descending (most recent first)
        points.sort(key=lambda x: x.get("period", ""), reverse=True)

        if len(points) < 2:
            return 0.0

        recent  = float(points[0]["value"])
        older   = float(points[min(4, len(points)-1)]["value"])

        if older == 0:
            return 0.0

        change_pct = (recent - older) / older
        # Cap at ±10% change mapping to ±1.0 score
        return max(-1.0, min(1.0, change_pct * 10))

    wti_score   = trend("wti_spot")
    brent_score = trend("brent_spot")
    avg_score   = (wti_score + brent_score) / 2

    # Get latest prices for reporting
    def latest_price(live_label, eia_label):
        # Prefer live yfinance price, fall back to EIA
        live = [d for d in market_data if d.get("label") == live_label and d.get("value") is not None]
        if live:
            live.sort(key=lambda x: x.get("period", ""), reverse=True)
            return {"period": live[0]["period"], "value": live[0]["value"], "source": "live"}
        eia = [d for d in market_data if d.get("label") == eia_label and d.get("value") is not None]
        eia.sort(key=lambda x: x.get("period", ""), reverse=True)
        return {"period": eia[0]["period"], "value": eia[0]["value"], "source": "eia"} if eia else {}

    detail = {
        "wti_trend":   round(wti_score, 3),
        "brent_trend": round(brent_score, 3),
        "score":       round(avg_score, 3),
        "wti_latest":  latest_price("wti_live", "wti_spot"),
        "brent_latest":latest_price("brent_live", "brent_spot"),
    }
    return avg_score, detail


def _gdelt_signal(articles: list[dict]) -> tuple[float, dict]:
    """
    Returns a score from -1.0 to +1.0 based on average GDELT tone.
    Uses ONLY 24h window records for primary signal.
    7d window used for context in report but not prediction.
    """
    tones = []
    for a in articles:
        if a.get("type") != "gdelt_gkg":
            continue
        # Only use 24h window for prediction signal
        if a.get("window", "24h") != "24h":
            continue
        tone = a.get("tone", {})
        if isinstance(tone, dict) and tone.get("tone") is not None:
            tones.append(float(tone["tone"]))

    if not tones:
        return 0.0, {"records": 0, "avg_tone": None, "score": 0.0}

    avg_tone = sum(tones) / len(tones)
    # Normalise: GDELT tone typically ranges -10 to +10
    score = max(-1.0, min(1.0, avg_tone / 10.0))

    detail = {
        "records":  len(tones),
        "avg_tone": round(avg_tone, 3),
        "score":    round(score, 3),
    }
    return score, detail


def _load_last_archive() -> tuple[list[dict], list[dict]]:
    """Load articles and market data from the most recent raw archive."""
    files = sorted(glob.glob(str(RAW_DIR / "**" / "*_run.json"), recursive=True))
    if not files:
        logger.warning("No raw archive found for prediction.")
        return [], []

    last = files[-1]
    logger.info(f"Prediction loading archive: {last}")
    data = json.load(open(last))
    return data.get("articles", []), data.get("market_data", [])


def _direction_from_score(score: float) -> str:
    if score > 0.1:
        return "bullish"
    elif score < -0.1:
        return "bearish"
    return "neutral"


def _confidence_from_score(score: float) -> str:
    abs_score = abs(score)
    if abs_score > 0.5:
        return "high"
    elif abs_score > 0.2:
        return "medium"
    return "low"


def generate_prediction(kg: KnowledgeGraph, market_data: list[dict] = None,
                        articles: list[dict] = None) -> dict:
    """
    Generate a price direction prediction by combining all three signals.
    Accepts market_data and articles directly from the pipeline.
    Falls back to loading last archive if not provided.
    Returns a structured prediction dict.
    """
    if market_data is None or articles is None:
        _articles, _market_data = _load_last_archive()
        if articles is None:
            articles = _articles
        if market_data is None:
            market_data = _market_data

    # Compute individual signals
    sent_score, sent_detail = _sentiment_signal(kg)
    eia_score,  eia_detail  = _eia_signal(market_data)
    gdelt_score,gdelt_detail= _gdelt_signal(articles)

    logger.info(f"Signals — sentiment: {sent_score:.3f}, EIA: {eia_score:.3f}, GDELT: {gdelt_score:.3f}")

    # Weighted composite score
    composite = (
        W_SENTIMENT * sent_score +
        W_EIA       * eia_score  +
        W_GDELT     * gdelt_score
    )

    direction  = _direction_from_score(composite)
    confidence = _confidence_from_score(composite)

    prediction = {
        "direction":    direction,
        "confidence":   confidence,
        "score":        round(composite, 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": {
            "sentiment": {**sent_detail,  "weight": W_SENTIMENT},
            "eia":       {**eia_detail,   "weight": W_EIA},
            "gdelt":     {**gdelt_detail, "weight": W_GDELT},
        }
    }

    logger.info(f"Prediction: {direction} ({confidence}) score={composite:.4f}")
    return prediction