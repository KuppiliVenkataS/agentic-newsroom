# """
# Prediction module.

# Combines three independent signals into a single price direction call:

# Signal 1 — Sentiment ratio (from knowledge graph)
#     Ratio of bullish vs bearish articles extracted by Ollama.
#     Weighted 40%.

# Signal 2 — EIA price trend (from raw archive market data)
#     Direction of WTI and Brent over the last 5 available data points.
#     Weighted 40%.

# Signal 3 — GDELT tone (from raw archive GDELT records)
#     Average tone score from GDELT GKG. Negative = bearish news coverage.
#     Weighted 20%.

# Output: {
#     "direction":   "bullish" | "bearish" | "neutral",
#     "confidence":  "high" | "medium" | "low",
#     "score":       float (-1.0 to 1.0),
#     "signals":     { breakdown of each signal },
#     "generated_at": ISO timestamp
# }
# """

# import glob
# import json
# import logging
# from datetime import datetime, timezone
# from pathlib import Path

# from config.settings import RAW_DIR
# from graph.knowledge_graph import KnowledgeGraph

# logger = logging.getLogger(__name__)

# # Signal weights — must sum to 1.0
# W_SENTIMENT = 0.40
# W_EIA       = 0.40
# W_GDELT     = 0.20


# def _sentiment_signal(kg: KnowledgeGraph) -> tuple[float, dict]:
#     """
#     Returns a score from -1.0 (fully bearish) to +1.0 (fully bullish).
#     Uses the knowledge graph signal summary.
#     """
#     summary  = kg.query_signal_summary()
#     bullish  = summary.get("bullish",  0)
#     bearish  = summary.get("bearish",  0)
#     neutral  = summary.get("neutral",  0)
#     unclear  = summary.get("unclear",  0)
#     total    = bullish + bearish + neutral + unclear

#     if total == 0:
#         return 0.0, {"bullish": 0, "bearish": 0, "neutral": 0, "total": 0}

#     # Score: bullish pushes positive, bearish pushes negative
#     score = (bullish - bearish) / total

#     detail = {
#         "bullish":  bullish,
#         "bearish":  bearish,
#         "neutral":  neutral,
#         "unclear":  unclear,
#         "total":    total,
#         "score":    round(score, 3),
#     }
#     return score, detail


# def _eia_signal(market_data: list[dict]) -> tuple[float, dict]:
#     """
#     Returns a score from -1.0 to +1.0 based on WTI and Brent price trends.
#     Compares most recent price to 5-period average.
#     """
#     def trend(series_label: str) -> float:
#         points = [
#             d for d in market_data
#             if d.get("label") == series_label and d.get("value") is not None
#         ]
#         # Sort by period descending (most recent first)
#         points.sort(key=lambda x: x.get("period", ""), reverse=True)

#         if len(points) < 2:
#             return 0.0

#         recent  = float(points[0]["value"])
#         older   = float(points[min(4, len(points)-1)]["value"])

#         if older == 0:
#             return 0.0

#         change_pct = (recent - older) / older
#         # Cap at ±10% change mapping to ±1.0 score
#         return max(-1.0, min(1.0, change_pct * 10))

#     wti_score   = trend("wti_spot")
#     brent_score = trend("brent_spot")
#     avg_score   = (wti_score + brent_score) / 2

#     # Get latest prices for reporting
#     def latest_price(live_label, eia_label):
#         # Prefer live yfinance price, fall back to EIA
#         live = [d for d in market_data if d.get("label") == live_label and d.get("value") is not None]
#         if live:
#             live.sort(key=lambda x: x.get("period", ""), reverse=True)
#             return {"period": live[0]["period"], "value": live[0]["value"], "source": "live"}
#         eia = [d for d in market_data if d.get("label") == eia_label and d.get("value") is not None]
#         eia.sort(key=lambda x: x.get("period", ""), reverse=True)
#         return {"period": eia[0]["period"], "value": eia[0]["value"], "source": "eia"} if eia else {}

#     detail = {
#         "wti_trend":   round(wti_score, 3),
#         "brent_trend": round(brent_score, 3),
#         "score":       round(avg_score, 3),
#         "wti_latest":  latest_price("wti_live", "wti_spot"),
#         "brent_latest":latest_price("brent_live", "brent_spot"),
#     }
#     return avg_score, detail


# def _gdelt_signal(articles: list[dict]) -> tuple[float, dict]:
#     """
#     Returns a score from -1.0 to +1.0 based on average GDELT tone.
#     GDELT tone is negative for bad news, positive for good news.
#     Typical range is -10 to +10; we normalise to -1 to +1.
#     """
#     tones = []
#     for a in articles:
#         if a.get("type") != "gdelt_gkg":
#             continue
#         tone = a.get("tone", {})
#         if isinstance(tone, dict) and tone.get("tone") is not None:
#             tones.append(float(tone["tone"]))

#     if not tones:
#         return 0.0, {"records": 0, "avg_tone": None, "score": 0.0}

#     avg_tone = sum(tones) / len(tones)
#     # Normalise: GDELT tone typically ranges -10 to +10
#     score = max(-1.0, min(1.0, avg_tone / 10.0))

#     detail = {
#         "records":  len(tones),
#         "avg_tone": round(avg_tone, 3),
#         "score":    round(score, 3),
#     }
#     return score, detail


# def _load_last_archive() -> tuple[list[dict], list[dict]]:
#     """Load articles and market data from the most recent raw archive."""
#     files = sorted(glob.glob(str(RAW_DIR / "**" / "*_run.json"), recursive=True))
#     if not files:
#         logger.warning("No raw archive found for prediction.")
#         return [], []

#     last = files[-1]
#     logger.info(f"Prediction loading archive: {last}")
#     data = json.load(open(last))
#     return data.get("articles", []), data.get("market_data", [])


# def _direction_from_score(score: float) -> str:
#     if score > 0.1:
#         return "bullish"
#     elif score < -0.1:
#         return "bearish"
#     return "neutral"


# def _confidence_from_score(score: float) -> str:
#     abs_score = abs(score)
#     if abs_score > 0.5:
#         return "high"
#     elif abs_score > 0.2:
#         return "medium"
#     return "low"


# def generate_prediction(kg: KnowledgeGraph, market_data: list[dict] = None,
#                         articles: list[dict] = None) -> dict:
#     """
#     Generate a price direction prediction by combining all three signals.
#     Accepts market_data and articles directly from the pipeline.
#     Falls back to loading last archive if not provided.
#     Returns a structured prediction dict.
#     """
#     if market_data is None or articles is None:
#         _articles, _market_data = _load_last_archive()
#         if articles is None:
#             articles = _articles
#         if market_data is None:
#             market_data = _market_data

#     # Compute individual signals
#     sent_score, sent_detail = _sentiment_signal(kg)
#     eia_score,  eia_detail  = _eia_signal(market_data)
#     gdelt_score,gdelt_detail= _gdelt_signal(articles)

#     logger.info(f"Signals — sentiment: {sent_score:.3f}, EIA: {eia_score:.3f}, GDELT: {gdelt_score:.3f}")

#     # Weighted composite score
#     composite = (
#         W_SENTIMENT * sent_score +
#         W_EIA       * eia_score  +
#         W_GDELT     * gdelt_score
#     )

#     direction  = _direction_from_score(composite)
#     confidence = _confidence_from_score(composite)

#     prediction = {
#         "direction":    direction,
#         "confidence":   confidence,
#         "score":        round(composite, 4),
#         "generated_at": datetime.now(timezone.utc).isoformat(),
#         "signals": {
#             "sentiment": {**sent_detail,  "weight": W_SENTIMENT},
#             "eia":       {**eia_detail,   "weight": W_EIA},
#             "gdelt":     {**gdelt_detail, "weight": W_GDELT},
#         }
#     }

#     logger.info(f"Prediction: {direction} ({confidence}) score={composite:.4f}")
#     return prediction
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