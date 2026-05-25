"""
Real-time oil price fetcher using OilPriceAPI.com

Free tier: no credit card required.
Get your API token at: https://oilpriceapi.com

Codes used:
    BRENT_CRUDE_USD — Brent crude spot price
    WTI_USD         — WTI crude spot price
"""

import logging
from datetime import datetime, timezone

import httpx

from config.settings import OILPRICE_API_KEY

logger = logging.getLogger(__name__)

BASE_URL = "https://api.oilpriceapi.com/v1/prices/latest"
HEADERS  = {"Content-Type": "application/json"}

PRICE_CODES = {
    "brent_live": "BRENT_CRUDE_USD",
    "wti_live":   "WTI_USD",
}


def fetch_live_prices() -> list[dict]:
    """
    Fetch current Brent and WTI prices from OilPriceAPI.com.
    Returns list of price dicts shaped like EIA market_data records.
    """
    if not OILPRICE_API_KEY or OILPRICE_API_KEY == "YOUR_OILPRICE_API_KEY_HERE":
        logger.warning("OilPrice API key not set — skipping live prices.")
        return []

    results    = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    today      = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    headers    = {**HEADERS, "Authorization": f"Token {OILPRICE_API_KEY}"}

    for label, code in PRICE_CODES.items():
        try:
            response = httpx.get(
                BASE_URL,
                params={"by_code": code},
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            data  = response.json()
            price = data.get("data", {}).get("price")

            if price is None:
                logger.warning(f"  {label}: no price in response")
                continue

            results.append({
                "source":     "oilpriceapi",
                "series_id":  code,
                "label":      label,
                "period":     today,
                "value":      round(float(price), 2),
                "unit":       "USD/barrel",
                "fetched_at": fetched_at,
                "type":       "market_data_live",
            })
            logger.info(f"  {label}: ${price}")

        except Exception as exc:
            logger.warning(f"  {label} fetch failed: {exc}")

    return results