"""
Prediction scorer.

Runs after each ingestion. Looks at the previous prediction,
compares it to actual price movement, and records accuracy.

After 90+ days of labeled data, use score history to retrain
signal weights via XGBoost (the ml_ready hook in predictor.py).

Output appended to: STORAGE_ROOT/data/logs/score_history.jsonl

Each record:
{
    "run_id":           "2026-05-28_06-00-00",
    "predicted_at":     ISO timestamp of the prediction being evaluated,
    "evaluated_at":     ISO timestamp of this evaluation,
    "predicted":        "bullish" | "bearish" | "neutral",
    "confidence":       "high" | "medium" | "low",
    "score":            float,
    "actual_direction": "up" | "down" | "flat",
    "actual_change_pct": float,
    "correct":          true | false | null,   # null if price data unavailable
    "horizon_hours":    12,
    "signals":          { full signal breakdown from the prediction }
}
"""

import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import RAW_DIR, LOG_DIR

logger = logging.getLogger(__name__)

SCORE_LOG = LOG_DIR / "score_history.jsonl"
FLAT_THRESHOLD = 0.003   # <0.3% price move = flat, not directional


def _load_archive(filepath: str) -> dict:
    try:
        return json.load(open(filepath))
    except Exception as exc:
        logger.warning(f"Could not load archive {filepath}: {exc}")
        return {}


def _get_latest_price(market_data: list[dict], label: str) -> float | None:
    """Get the most recent value for a price label from market_data."""
    points = [
        d for d in market_data
        if d.get("label") == label and d.get("value") is not None
    ]
    if not points:
        return None
    points.sort(key=lambda x: x.get("period", ""), reverse=True)
    return float(points[0]["value"])


def _actual_direction(prev_price: float, curr_price: float) -> tuple[str, float]:
    """
    Compare two prices and return direction + percentage change.
    """
    if prev_price == 0:
        return "flat", 0.0
    change_pct = (curr_price - prev_price) / prev_price
    if change_pct > FLAT_THRESHOLD:
        direction = "up"
    elif change_pct < -FLAT_THRESHOLD:
        direction = "down"
    else:
        direction = "flat"
    return direction, round(change_pct * 100, 4)


def _prediction_correct(predicted: str, actual: str) -> bool | None:
    """
    True if prediction matched actual direction.
    None if actual is flat (can't score neutral predictions easily).
    """
    if actual == "flat":
        return None
    if predicted == "bullish" and actual == "up":
        return True
    if predicted == "bearish" and actual == "down":
        return True
    if predicted == "neutral":
        return None   # neutral predictions not scored directionally
    return False


def score_last_prediction(current_market_data: list[dict],
                           current_run_id: str) -> dict | None:
    """
    Find the second-to-last archive, extract its prediction,
    compare to current prices, write score record.

    Returns the score dict, or None if insufficient data.
    """
    # Find the two most recent archives
    files = sorted(glob.glob(str(RAW_DIR / "**" / "*_run.json"), recursive=True))
    if len(files) < 2:
        logger.info("Scorer: need at least 2 runs to evaluate — skipping")
        return None

    prev_archive = _load_archive(files[-2])
    if not prev_archive:
        return None

    prediction = prev_archive.get("prediction")
    if not prediction:
        logger.info("Scorer: no prediction in previous archive — skipping")
        return None

    prev_market_data = prev_archive.get("market_data", [])

    # Get Brent price from both runs (prefer live prices)
    for label in ("brent_live", "brent_spot"):
        prev_price = _get_latest_price(prev_market_data, label)
        curr_price = _get_latest_price(current_market_data, label)
        if prev_price and curr_price:
            break

    if not prev_price or not curr_price:
        logger.warning("Scorer: could not find comparable prices — skipping")
        return None

    actual_dir, change_pct = _actual_direction(prev_price, curr_price)
    predicted_dir          = prediction.get("direction", "neutral")
    correct                = _prediction_correct(predicted_dir, actual_dir)

    score_record = {
        "run_id":            current_run_id,
        "predicted_at":      prediction.get("generated_at", ""),
        "evaluated_at":      datetime.now(timezone.utc).isoformat(),
        "predicted":         predicted_dir,
        "confidence":        prediction.get("confidence", ""),
        "score":             prediction.get("score", 0.0),
        "actual_direction":  actual_dir,
        "actual_change_pct": change_pct,
        "prev_brent":        round(prev_price, 2),
        "curr_brent":        round(curr_price, 2),
        "correct":           correct,
        "horizon_hours":     12,
        "signals":           prediction.get("signals", {}),
    }

    # Append to score log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(score_record) + "\n")

    result_str = "CORRECT" if correct is True else "WRONG" if correct is False else "UNSCORED"
    logger.info(
        f"Scorer: {predicted_dir} predicted, Brent moved {change_pct:+.2f}% "
        f"({prev_price:.2f}→{curr_price:.2f}) — {result_str}"
    )
    return score_record


def read_accuracy_summary() -> dict:
    """
    Read score_history.jsonl and return a summary of prediction accuracy.
    Useful for logging and future ML training.
    """
    if not SCORE_LOG.exists():
        return {"total": 0, "correct": 0, "wrong": 0, "unscored": 0, "accuracy": None}

    records  = [json.loads(l) for l in open(SCORE_LOG) if l.strip()]
    total    = len(records)
    correct  = sum(1 for r in records if r.get("correct") is True)
    wrong    = sum(1 for r in records if r.get("correct") is False)
    unscored = sum(1 for r in records if r.get("correct") is None)
    scored   = correct + wrong
    accuracy = round(correct / scored, 3) if scored > 0 else None

    # Breakdown by confidence level
    by_confidence = {}
    for conf in ("high", "medium", "low"):
        conf_records = [r for r in records if r.get("confidence") == conf
                        and r.get("correct") is not None]
        conf_correct = sum(1 for r in conf_records if r.get("correct"))
        by_confidence[conf] = {
            "total":    len(conf_records),
            "correct":  conf_correct,
            "accuracy": round(conf_correct / len(conf_records), 3) if conf_records else None,
        }

    return {
        "total":         total,
        "correct":       correct,
        "wrong":         wrong,
        "unscored":      unscored,
        "accuracy":      accuracy,
        "by_confidence": by_confidence,
    }