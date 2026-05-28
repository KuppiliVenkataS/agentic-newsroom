"""
Real-time oil price fetcher using yfinance.

Free, no API key, pulls from Yahoo Finance futures data.
Updates continuously during market hours.

Tickers:
    BZ=F  — Brent Crude front-month futures (USD/barrel)
    CL=F  — WTI Crude front-month futures (USD/barrel)

Fallback chain per ticker:
    1. Latest market price (real-time during hours, last close outside)
    2. Most recent 5-day 1h bar close
    3. Skip — EIA data will be used instead
"""

import logging
from datetime import datetime, timezone

from config.settings import OILPRICE_API_KEY

logger = logging.getLogger(__name__)

TICKERS = {
    "brent_live": "BZ=F",
    "wti_live":   "CL=F",
}


def fetch_live_prices() -> list[dict]:
    """
    Fetch current Brent and WTI prices from Yahoo Finance via yfinance.
    Returns list of price dicts shaped like EIA market_data records.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return []

    results    = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for label, ticker_symbol in TICKERS.items():
        price = None
        try:
            ticker = yf.Ticker(ticker_symbol)

            # Try fast_info first — single network call, real-time price
            fast = ticker.fast_info
            price = getattr(fast, "last_price", None)

            # Fallback: pull last 5 days of 1h bars and take the most recent close
            if not price:
                hist = ticker.history(period="5d", interval="1h", auto_adjust=True)
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price is None:
                logger.warning(f"  {label} ({ticker_symbol}): no price available")
                continue

            price = round(float(price), 2)
            results.append({
                "source":     "yfinance",
                "series_id":  ticker_symbol,
                "label":      label,
                "period":     today,
                "value":      price,
                "unit":       "USD/barrel",
                "fetched_at": fetched_at,
                "type":       "market_data_live",
            })
            logger.info(f"  {label} ({ticker_symbol}): ${price}")

        except Exception as exc:
            logger.warning(f"  {label} ({ticker_symbol}) fetch failed: {exc}")

    return results