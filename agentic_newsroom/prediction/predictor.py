"""
Hybrid prediction module.

Rule-based engine with 7 signals. Designed for ML layer addition later.

Signals:
    1. News sentiment ratio        (Ollama extraction, 24h)     weight: 0.20
    2. GDELT tone 24h              (global news coverage)        weight: 0.20
    3. GDELT tone trend 7d         (improving or deteriorating)  weight: 0.10
    4. EIA price momentum          (5-day WTI/Brent trend)       weight: 0.20
    5. EIA inventory signal        (draw=bullish, build=bearish) weight: 0.10
    6. Geopolitical risk score     (Iran/Hormuz/Russia intensity) weight: 0.15
    7. Supply disruption flag      (explicit event detection)    weight: 0.05

Output:
    12-hour direction + confidence
    3-day outlook direction + confidence
    Full signal breakdown for transparency

ML hook: When enough historical data accumulates (90+ days),
    replace weights with trained coefficients from XGBoost/LightGBM.
"""

import glob
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from config.settings import RAW_DIR
from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# ── Signal weights ─────────────────────────────────────────────────────────────
W = {
    "sentiment_24h":   0.20,
    "gdelt_tone_24h":  0.20,
    "gdelt_trend_7d":  0.10,
    "eia_momentum":    0.20,
    "eia_inventory":   0.10,
    "geopolitical":    0.15,
    "disruption":      0.05,
}

# ── Geopolitical keywords ──────────────────────────────────────────────────────
HIGH_RISK_KEYWORDS = [
    "hormuz", "strait of hormuz", "blockade", "iran attack", "iran strike",
    "nuclear", "missile", "explosion", "pipeline attack", "tanker seized",
    "houthi", "red sea", "oil facility", "refinery attack", "drone strike",
    "war", "invasion", "escalation", "sanctions imposed", "embargo",
]

MEDIUM_RISK_KEYWORDS = [
    "iran", "sanctions", "russia", "ukraine", "opec cut", "supply disruption",
    "tension", "conflict", "israel", "hezbollah", "gulf", "military",
    "negotiations", "deal collapsed", "ceasefire", "airstrikes",
]

LOW_RISK_KEYWORDS = [
    "talks", "negotiations", "diplomacy", "agreement", "deal progress",
    "ceasefire", "peace", "sanctions relief", "reopened",
]


# ── Signal 1 — News sentiment ──────────────────────────────────────────────────
def _sentiment_signal(kg: KnowledgeGraph) -> tuple[float, dict]:
    summary  = kg.query_signal_summary()
    bullish  = summary.get("bullish",  0)
    bearish  = summary.get("bearish",  0)
    neutral  = summary.get("neutral",  0)
    unclear  = summary.get("unclear",  0)
    total    = bullish + bearish + neutral + unclear

    if total == 0:
        return 0.0, {"bullish": 0, "bearish": 0, "total": 0, "score": 0.0, "reliability": "no_data"}

    score = (bullish - bearish) / total
    reliability = "high" if total >= 20 else "medium" if total >= 10 else "low"

    return score, {
        "bullish": bullish, "bearish": bearish,
        "neutral": neutral, "unclear": unclear,
        "total": total, "score": round(score, 3),
        "reliability": reliability
    }


# ── Signal 2 — GDELT tone 24h ─────────────────────────────────────────────────
def _gdelt_tone_24h(articles: list[dict]) -> tuple[float, dict]:
    tones = [
        float(a["tone"]["tone"])
        for a in articles
        if a.get("type") == "gdelt_gkg"
        and a.get("window", "24h") == "24h"
        and isinstance(a.get("tone"), dict)
        and a["tone"].get("tone") is not None
    ]
    if not tones:
        return 0.0, {"records": 0, "avg_tone": None, "score": 0.0}

    avg = sum(tones) / len(tones)
    score = max(-1.0, min(1.0, avg / 10.0))
    return score, {"records": len(tones), "avg_tone": round(avg, 3), "score": round(score, 3)}


# ── Signal 3 — GDELT tone trend 7d ────────────────────────────────────────────
def _gdelt_trend_7d(articles: list[dict]) -> tuple[float, dict]:
    """Compare 7d background tone to 24h tone — is sentiment improving?"""
    tones_24h = [
        float(a["tone"]["tone"])
        for a in articles
        if a.get("type") == "gdelt_gkg" and a.get("window") == "24h"
        and isinstance(a.get("tone"), dict) and a["tone"].get("tone") is not None
    ]
    tones_7d = [
        float(a["tone"]["tone"])
        for a in articles
        if a.get("type") == "gdelt_gkg" and a.get("window") == "7d"
        and isinstance(a.get("tone"), dict) and a["tone"].get("tone") is not None
    ]

    if not tones_24h or not tones_7d:
        return 0.0, {"score": 0.0, "trend": "insufficient_data"}

    avg_24h = sum(tones_24h) / len(tones_24h)
    avg_7d  = sum(tones_7d)  / len(tones_7d)
    delta   = avg_24h - avg_7d  # positive = improving sentiment

    score = max(-1.0, min(1.0, delta / 5.0))
    trend = "improving" if delta > 0.5 else "deteriorating" if delta < -0.5 else "stable"

    return score, {
        "avg_24h": round(avg_24h, 3),
        "avg_7d":  round(avg_7d, 3),
        "delta":   round(delta, 3),
        "trend":   trend,
        "score":   round(score, 3),
    }


# ── Signal 4 — EIA price momentum ─────────────────────────────────────────────
def _eia_momentum(market_data: list[dict]) -> tuple[float, dict]:
    def trend(label: str) -> float:
        points = [
            d for d in market_data
            if d.get("label") == label and d.get("value") is not None
        ]
        points.sort(key=lambda x: x.get("period", ""), reverse=True)
        if len(points) < 2:
            return 0.0
        recent = float(points[0]["value"])
        older  = float(points[min(4, len(points)-1)]["value"])
        if older == 0:
            return 0.0
        return max(-1.0, min(1.0, ((recent - older) / older) * 10))

    def latest(label_live: str, label_eia: str) -> dict:
        live = [d for d in market_data if d.get("label") == label_live and d.get("value") is not None]
        if live:
            live.sort(key=lambda x: x.get("period", ""), reverse=True)
            return {"value": live[0]["value"], "period": live[0]["period"], "source": "live"}
        eia = [d for d in market_data if d.get("label") == label_eia and d.get("value") is not None]
        eia.sort(key=lambda x: x.get("period", ""), reverse=True)
        return {"value": eia[0]["value"], "period": eia[0]["period"], "source": "eia"} if eia else {}

    wti_score   = trend("wti_spot")
    brent_score = trend("brent_spot")
    avg_score   = (wti_score + brent_score) / 2

    return avg_score, {
        "wti_trend":    round(wti_score, 3),
        "brent_trend":  round(brent_score, 3),
        "score":        round(avg_score, 3),
        "wti_latest":   latest("wti_live", "wti_spot"),
        "brent_latest": latest("brent_live", "brent_spot"),
    }


# ── Signal 5 — EIA inventory ──────────────────────────────────────────────────
def _eia_inventory(market_data: list[dict]) -> tuple[float, dict]:
    """Inventory draw = bullish, build = bearish."""
    points = [
        d for d in market_data
        if d.get("label") == "us_inventory" and d.get("value") is not None
    ]
    points.sort(key=lambda x: x.get("period", ""), reverse=True)

    if len(points) < 2:
        return 0.0, {"score": 0.0, "signal": "no_data"}

    latest_val = float(points[0]["value"])
    prev_val   = float(points[1]["value"])
    change     = latest_val - prev_val
    change_pct = (change / prev_val) * 100 if prev_val else 0

    # Draw (negative change) = bullish, build = bearish
    score  = max(-1.0, min(1.0, -change_pct / 2.0))
    signal = "draw_bullish" if change < 0 else "build_bearish"

    return score, {
        "latest_mb":  round(latest_val, 1),
        "prev_mb":    round(prev_val, 1),
        "change_mb":  round(change, 1),
        "signal":     signal,
        "score":      round(score, 3),
    }


# ── Signal 6 — Geopolitical risk ──────────────────────────────────────────────
def _geopolitical_risk(articles: list[dict]) -> tuple[float, dict]:
    """
    Score based on presence of high/medium/low risk keywords in 24h articles.
    High risk keywords (Hormuz closure, attack) = strongly bullish (supply shock).
    Low risk keywords (peace deal, diplomacy) = bearish.
    """
    text_24h = " ".join([
        " ".join(filter(None, [
            a.get("title") or "",
            a.get("summary") or "",
            a.get("names") or "",
            a.get("themes") or "",
        ])).lower()
        for a in articles
        if a.get("window", "24h") == "24h" or a.get("type") == "rss_article"
    ])

    high_hits   = sum(1 for kw in HIGH_RISK_KEYWORDS   if kw in text_24h)
    medium_hits = sum(1 for kw in MEDIUM_RISK_KEYWORDS if kw in text_24h)
    low_hits    = sum(1 for kw in LOW_RISK_KEYWORDS    if kw in text_24h)

    # High risk = bullish (supply disruption fear)
    # Low risk  = bearish (de-escalation)
    raw_score = (high_hits * 0.3 + medium_hits * 0.1 - low_hits * 0.15)
    score     = max(-1.0, min(1.0, raw_score))

    level = "critical" if high_hits >= 3 else \
            "high"     if high_hits >= 1 else \
            "medium"   if medium_hits >= 3 else \
            "low"      if medium_hits >= 1 else "minimal"

    triggered = [kw for kw in HIGH_RISK_KEYWORDS if kw in text_24h][:5]

    return score, {
        "level":          level,
        "high_hits":      high_hits,
        "medium_hits":    medium_hits,
        "low_hits":       low_hits,
        "triggered_by":   triggered,
        "score":          round(score, 3),
    }


# ── Signal 7 — Supply disruption flag ─────────────────────────────────────────
def _disruption_flag(articles: list[dict]) -> tuple[float, dict]:
    """
    Explicit supply disruption detection.
    If a major disruption is detected, override with strong bullish signal.
    """
    disruption_keywords = [
        "pipeline explosion", "refinery fire", "oil facility attack",
        "force majeure", "production shutdown", "field shut",
        "export halt", "port closure", "tanker attack",
        "hormuz closed", "hormuz blocked",
    ]

    text = " ".join([
        " ".join(filter(None, [
            a.get("title") or "",
            a.get("summary") or "",
        ])).lower()
        for a in articles
        if a.get("type") == "rss_article"
    ])

    hits     = [kw for kw in disruption_keywords if kw in text]
    detected = len(hits) > 0
    score    = 0.8 if detected else 0.0

    return score, {
        "detected":  detected,
        "triggers":  hits[:3],
        "score":     score,
    }


# ── 3-day outlook ──────────────────────────────────────────────────────────────
def _three_day_outlook(signals: dict, composite_12h: float) -> dict:
    """
    3-day outlook uses 7d trend and geopolitical trajectory.
    More conservative than 12h signal.
    """
    gdelt_trend = signals["gdelt_trend_7d"].get("trend", "stable")
    geo_level   = signals["geopolitical"].get("level", "minimal")
    inv_signal  = signals["eia_inventory"].get("signal", "no_data")

    # Start from 12h signal but dampen it
    score_3d = composite_12h * 0.6

    # Adjust for 7d trend direction
    if gdelt_trend == "deteriorating":
        score_3d -= 0.1
    elif gdelt_trend == "improving":
        score_3d += 0.1

    # Geopolitical overhang
    if geo_level in ("critical", "high"):
        score_3d += 0.15   # sustained supply risk
    elif geo_level == "minimal":
        score_3d -= 0.05

    # Inventory trend
    if inv_signal == "draw_bullish":
        score_3d += 0.05
    elif inv_signal == "build_bearish":
        score_3d -= 0.05

    score_3d = max(-1.0, min(1.0, score_3d))

    return {
        "direction":  _direction(score_3d),
        "confidence": _confidence(score_3d),
        "score":      round(score_3d, 4),
        "rationale":  f"7d sentiment {gdelt_trend}, geopolitical risk {geo_level}, inventory {inv_signal}",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _direction(score: float) -> str:
    if score > 0.1:  return "bullish"
    if score < -0.1: return "bearish"
    return "neutral"

def _confidence(score: float) -> str:
    abs_s = abs(score)
    if abs_s > 0.5: return "high"
    if abs_s > 0.2: return "medium"
    return "low"

def _load_last_archive() -> tuple[list[dict], list[dict]]:
    files = sorted(glob.glob(str(RAW_DIR / "**" / "*_run.json"), recursive=True))
    if not files:
        return [], []
    data = json.load(open(files[-1]))
    return data.get("articles", []), data.get("market_data", [])


# ── Main ───────────────────────────────────────────────────────────────────────
def generate_prediction(
    kg: KnowledgeGraph,
    market_data: list[dict] = None,
    articles: list[dict]    = None,
) -> dict:
    """
    Generate price direction prediction using 7 signals.
    Returns 12h and 3-day outlooks with full signal breakdown.
    """
    if market_data is None or articles is None:
        _articles, _market_data = _load_last_archive()
        if articles    is None: articles    = _articles
        if market_data is None: market_data = _market_data

    logger.info("Running prediction engine (7 signals)...")

    # Compute all signals
    s1_score, s1 = _sentiment_signal(kg)
    s2_score, s2 = _gdelt_tone_24h(articles)
    s3_score, s3 = _gdelt_trend_7d(articles)
    s4_score, s4 = _eia_momentum(market_data)
    s5_score, s5 = _eia_inventory(market_data)
    s6_score, s6 = _geopolitical_risk(articles)
    s7_score, s7 = _disruption_flag(articles)

    logger.info(f"  sentiment={s1_score:.3f} gdelt_24h={s2_score:.3f} gdelt_7d={s3_score:.3f}")
    logger.info(f"  eia_mom={s4_score:.3f} inventory={s5_score:.3f} geo={s6_score:.3f} disruption={s7_score:.3f}")

    # Weighted composite
    composite = (
        W["sentiment_24h"]  * s1_score +
        W["gdelt_tone_24h"] * s2_score +
        W["gdelt_trend_7d"] * s3_score +
        W["eia_momentum"]   * s4_score +
        W["eia_inventory"]  * s5_score +
        W["geopolitical"]   * s6_score +
        W["disruption"]     * s7_score
    )

    signals = {
        "sentiment_24h":  {**s1, "weight": W["sentiment_24h"]},
        "gdelt_tone_24h": {**s2, "weight": W["gdelt_tone_24h"]},
        "gdelt_trend_7d": {**s3, "weight": W["gdelt_trend_7d"]},
        "eia":            {**s4, "weight": W["eia_momentum"]},
        "eia_inventory":  {**s5, "weight": W["eia_inventory"]},
        "geopolitical":   {**s6, "weight": W["geopolitical"]},
        "disruption":     {**s7, "weight": W["disruption"]},
    }

    outlook_3d = _three_day_outlook(signals, composite)

    prediction = {
        "direction":    _direction(composite),
        "confidence":   _confidence(composite),
        "score":        round(composite, 4),
        "horizon":      "12h",
        "outlook_3d":   outlook_3d,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals":      signals,
        "weights":      W,
        # ML hook — replace weights with trained coefficients when ready
        "ml_ready":     False,
        "ml_note":      "Accumulate 90+ days of labeled data then train XGBoost on signal scores vs next-day price change",
    }

    logger.info(f"12h: {prediction['direction']} ({prediction['confidence']}) score={composite:.4f}")
    logger.info(f"3d:  {outlook_3d['direction']} ({outlook_3d['confidence']}) score={outlook_3d['score']:.4f}")
    return prediction