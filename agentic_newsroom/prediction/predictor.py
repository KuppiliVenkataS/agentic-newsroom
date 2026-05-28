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

from config.settings import RAW_DIR, USER_WATCHLIST, WATCHLIST_BOOST
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

# ── Supply impact scoring — topic-agnostic ────────────────────────────────────
# These keywords detect the MECHANISM of price impact, not the topic.
# New events (Abqaiq attack, Venezuela collapse, Libyan shutdown) will score
# correctly as long as they describe physical supply/demand effects.

SUPPLY_REDUCTION_KEYWORDS = [
    # Physical shutdown language
    "shut down", "shutdown", "offline", "force majeure", "export halt",
    "port closure", "blockade", "pipeline explosion", "refinery fire",
    "production cut", "output cut", "supply cut", "field shut",
    # Attack/damage language
    "attack", "strike", "explosion", "sabotage", "drone strike",
    "missile strike", "bombed", "destroyed", "damaged",
    # Disruption confirmation
    "disruption", "disrupted", "suspended", "halted", "stopped",
    "seized", "impounded", "detained",
]

SUPPLY_INCREASE_KEYWORDS = [
    "production increase", "output increase", "ramp up", "restored",
    "reopened", "lifted sanctions", "sanctions relief", "ceasefire",
    "peace deal", "agreement reached", "deal signed", "normalisation",
    "spare capacity", "flood the market", "increase supply",
]

DEMAND_SHOCK_KEYWORDS = [
    "recession", "economic collapse", "demand destruction", "lockdown",
    "slowdown", "contraction", "demand surge", "economic boom",
    "industrial boom", "recovery stronger",
]

RISK_ESCALATION_KEYWORDS = [
    # Escalation signals — not yet supply impact but high probability
    "escalation", "retaliation", "war", "invasion", "naval",
    "strait", "blockade threatened", "threatened to close",
    "tensions rising", "brink", "imminent", "ultimatum",
]


# ── Signal 1 — News sentiment (importance-weighted, recency-decayed) ───────────
def _watchlist_boost(article: dict) -> float:
    """
    Returns an additive importance boost if the article matches
    any keyword in USER_WATCHLIST. Stacks per match, capped at WATCHLIST_BOOST.
    Matching is against title + summary, case-insensitive.
    """
    if not USER_WATCHLIST or WATCHLIST_BOOST == 0.0:
        return 0.0
    text = " ".join(filter(None, [
        article.get("title", ""),
        article.get("summary", ""),
    ])).lower()
    hits = sum(1 for kw in USER_WATCHLIST if kw.lower() in text)
    return min(WATCHLIST_BOOST, hits * (WATCHLIST_BOOST / 2)) if hits else 0.0
    """
    Weighted sentiment where each article contributes its importance_score
    multiplied by a recency decay factor, not a flat +1/-1 count.

    Urgency multipliers (on top of importance_score):
        critical  → 3.0x  (active strike, Hormuz closure)
        high      → 2.0x  (sanctions, ceasefire collapse)
        medium    → 1.0x
        low       → 0.5x

    Recency decay: articles older than 24h get 0.5 weight; older than 48h get 0.25.
    This ensures last night's bombing dominates a 3-day-old analyst note.

    Falls back to flat KG counts if no enriched articles available.
    """
    URGENCY_MULTIPLIER = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}

    bullish_w = 0.0
    bearish_w = 0.0
    article_count = 0

    now = datetime.now(timezone.utc)

    if enriched_articles:
        for article in enriched_articles:
            for chunk_result in article.get("extraction", []):
                if chunk_result.get("status") != "ok":
                    continue

                direction = chunk_result.get("price_signals", {}).get("direction", "unclear")
                if direction not in ("bullish", "bearish"):
                    continue

                importance  = float(chunk_result.get("importance_score", 0.3))
                importance  = min(1.0, importance + _watchlist_boost(article))
                urgency_key = chunk_result.get("events", [{}])[0].get("urgency", "medium") \
                              if chunk_result.get("events") else "medium"
                urgency_mult = URGENCY_MULTIPLIER.get(urgency_key, 1.0)

                # Recency decay from article published time
                published_str = article.get("published") or article.get("fetched_at", "")
                decay = 1.0
                if published_str:
                    try:
                        pub = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        age_hours = (now - pub).total_seconds() / 3600
                        decay = 1.0 if age_hours <= 24 else 0.5 if age_hours <= 48 else 0.25
                    except (ValueError, TypeError):
                        decay = 1.0

                weight = importance * urgency_mult * decay
                article_count += 1

                if direction == "bullish":
                    bullish_w += weight
                else:
                    bearish_w += weight

    # Fall back to flat KG counts when no enriched data
    if bullish_w == 0 and bearish_w == 0:
        summary = kg.query_signal_summary()
        bullish_w = float(summary.get("bullish", 0))
        bearish_w = float(summary.get("bearish", 0))
        article_count = int(summary.get("bullish", 0) + summary.get("bearish", 0)
                            + summary.get("neutral", 0) + summary.get("unclear", 0))

    total_w = bullish_w + bearish_w
    if total_w == 0:
        return 0.0, {"bullish_w": 0, "bearish_w": 0, "articles": 0, "score": 0.0, "reliability": "no_data", "method": "none"}

    score = (bullish_w - bearish_w) / total_w
    reliability = "high" if article_count >= 20 else "medium" if article_count >= 10 else "low"

    return score, {
        "bullish_w":   round(bullish_w, 3),
        "bearish_w":   round(bearish_w, 3),
        "bullish":     sum(1 for a in (enriched_articles or []) for c in a.get("extraction", [])
                          if c.get("price_signals", {}).get("direction") == "bullish"),
        "bearish":     sum(1 for a in (enriched_articles or []) for c in a.get("extraction", [])
                          if c.get("price_signals", {}).get("direction") == "bearish"),
        "articles":    article_count,
        "score":       round(score, 3),
        "reliability": reliability,
        "method":      "weighted" if enriched_articles else "flat_kg",
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


# ── Signal 6 — Supply impact score (topic-agnostic) ───────────────────────────
def _geopolitical_risk(articles: list[dict], enriched_articles: list[dict] = None) -> tuple[float, dict]:
    """
    Scores the net supply/demand impact of current events.
    Topic-agnostic: works whether the driver is Iran, Venezuela,
    Abqaiq, a hurricane, or something entirely new.

    Two layers:
    Layer 1 — mechanism keyword scan of raw article text.
              Detects supply reduction, supply increase, demand shocks,
              and escalation risk regardless of geography or actor.
    Layer 2 — extraction importance scores (LLM-reasoned).
              High-importance articles add a weighted boost.
              The LLM already reasoned about price mechanism in Layer 1
              of extraction, so this captures nuance keywords miss.
    """
    text_all = " ".join([
        " ".join(filter(None, [
            a.get("title") or "",
            a.get("summary") or "",
        ])).lower()
        for a in articles
        if a.get("type") == "rss_article"
    ])

    # Layer 1 — mechanism keyword scan
    supply_cut_hits  = sum(1 for kw in SUPPLY_REDUCTION_KEYWORDS  if kw in text_all)
    supply_add_hits  = sum(1 for kw in SUPPLY_INCREASE_KEYWORDS   if kw in text_all)
    demand_hits      = sum(1 for kw in DEMAND_SHOCK_KEYWORDS       if kw in text_all)
    escalation_hits  = sum(1 for kw in RISK_ESCALATION_KEYWORDS   if kw in text_all)

    # Net keyword score: supply cuts and escalation = bullish, supply adds = bearish
    kw_score = max(-1.0, min(1.0,
        supply_cut_hits  * 0.15 +
        escalation_hits  * 0.10 -
        supply_add_hits  * 0.15 +
        demand_hits      * 0.05   # demand shocks can go either way — small weight
    ))

    # Layer 2 — importance-weighted extraction boost
    bullish_imp  = 0.0
    bearish_imp  = 0.0
    hormuz_boost = 0.0
    sanctions_b  = 0.0
    opec_b       = 0.0
    n_extractions = 0

    if enriched_articles:
        for article in enriched_articles:
            for chunk in article.get("extraction", []):
                if chunk.get("status") != "ok":
                    continue
                imp       = min(1.0, float(chunk.get("importance_score", 0.0)) + _watchlist_boost(article))
                direction = chunk.get("price_signals", {}).get("direction", "unclear")
                if direction == "bullish":
                    bullish_imp += imp
                elif direction == "bearish":
                    bearish_imp += imp
                if chunk.get("hormuz_risk"):
                    hormuz_boost += imp * 0.5
                if chunk.get("sanctions_event"):
                    sanctions_b  += imp * 0.3
                if chunk.get("opec_event"):
                    opec_b       += imp * 0.2
                n_extractions += 1

    total_imp = bullish_imp + bearish_imp
    imp_score = (bullish_imp - bearish_imp) / total_imp if total_imp > 0 else 0.0
    imp_boost = min(0.5, hormuz_boost + sanctions_b + opec_b)

    final_score = max(-1.0, min(1.0, kw_score * 0.4 + imp_score * 0.4 + imp_boost * 0.2))

    # Level derived from score, not hardcoded topics
    level = "critical" if final_score > 0.6  else \
            "high"     if final_score > 0.35 else \
            "medium"   if final_score > 0.1  else \
            "low"      if final_score > -0.1 else "bearish_risk"

    triggered = [kw for kw in SUPPLY_REDUCTION_KEYWORDS if kw in text_all][:5]

    return final_score, {
        "level":            level,
        "supply_cut_hits":  supply_cut_hits,
        "supply_add_hits":  supply_add_hits,
        "escalation_hits":  escalation_hits,
        "triggered_by":     triggered,
        "hormuz_boost":     round(hormuz_boost, 3),
        "sanctions_boost":  round(sanctions_b, 3),
        "opec_boost":       round(opec_b, 3),
        "kw_score":         round(kw_score, 3),
        "imp_score":        round(imp_score, 3),
        "score":            round(final_score, 3),
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

    # enriched_articles = articles that have gone through extraction (have importance scores, flags)
    # articles = all raw articles including GDELT (used for tone/geo keyword signals)
    enriched = [a for a in articles if a.get("extraction")]

    # Compute all signals
    s1_score, s1 = _sentiment_signal(kg, enriched_articles=enriched)
    s2_score, s2 = _gdelt_tone_24h(articles)
    s3_score, s3 = _gdelt_trend_7d(articles)
    s4_score, s4 = _eia_momentum(market_data)
    s5_score, s5 = _eia_inventory(market_data)
    s6_score, s6 = _geopolitical_risk(articles, enriched_articles=enriched)
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