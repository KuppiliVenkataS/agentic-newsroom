"""
Real-time oil price fetcher using yfinance.

Pulls live (15-min delayed) Brent and WTI prices from Yahoo Finance.
Used for display in reports and emails — fresher than EIA's 2-4 day lag.

EIA data is still used for trend calculation (official, reliable).

Tickers:
    BZ=F — Brent Crude Futures
    CL=F — WTI Crude Futures
    NG=F — Natural Gas Futures (bonus)
"""

import logging
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = {
    "brent_live": "BZ=F",
    "wti_live":   "CL=F",
    "natgas_live":"NG=F",
}


def fetch_live_prices() -> list[dict]:
    """
    Fetch current oil prices from Yahoo Finance.
    Returns list of price dicts shaped like EIA market_data records.
    """
    results    = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for label, ticker_symbol in TICKERS.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            info   = ticker.fast_info

            price = getattr(info, "last_price", None)
            if price is None:
                # fallback to history
                hist  = ticker.history(period="1d")
                price = float(hist["Close"].iloc[-1]) if not hist.empty else None

            if price is None:
                logger.warning(f"  {label}: no price returned")
                continue

            results.append({
                "source":     "yfinance",
                "series_id":  ticker_symbol,
                "label":      label,
                "period":     today,
                "value":      round(float(price), 2),
                "unit":       "USD/barrel",
                "fetched_at": fetched_at,
                "type":       "market_data_live",
            })
            logger.info(f"  {label} ({ticker_symbol}): ${price:.2f}")

        except Exception as exc:
            logger.warning(f"  {label} fetch failed: {exc}")

    return results