"""
EIA Open Data API fetcher.

Pulls oil price and inventory series defined in settings.EIA_SERIES.
Returns structured dicts alongside the news articles so everything
lands in the same JSON archive.

EIA v2 API docs: https://www.eia.gov/opendata/documentation.php
"""

import logging
from datetime import datetime

import httpx

from config.settings import EIA_API_KEY, EIA_SERIES, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

EIA_BASE = "https://api.eia.gov/v2/seriesid/{series_id}"


def _fetch_series(series_id: str, api_key: str) -> list[dict]:
    """Fetch the most recent 10 data points for one EIA series."""
    url = EIA_BASE.format(series_id=series_id)
    params = {
        "api_key": api_key,
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 10,
        "offset": 0,
    }
    try:
        r = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("response", {}).get("data", [])
        return rows
    except Exception as exc:
        logger.warning(f"  EIA series {series_id} failed: {exc}")
        return []


def fetch_eia_data() -> list[dict]:
    """
    Pull all configured EIA series.
    Returns a list of structured dicts, one per data point,
    shaped the same as article dicts so they go into the same archive.
    """
    if EIA_API_KEY == "YOUR_EIA_API_KEY_HERE":
        logger.warning("EIA API key not set — skipping EIA fetch.")
        return []

    results: list[dict] = []
    fetched_at = datetime.utcnow().isoformat()

    for label, series_id in EIA_SERIES.items():
        logger.info(f"Fetching EIA: {label} ({series_id})")
        rows = _fetch_series(series_id, EIA_API_KEY)

        for row in rows:
            results.append({
                "source":      "eia",
                "series_id":   series_id,
                "label":       label,
                "period":      row.get("period", ""),
                "value":       row.get("value"),
                "unit":        row.get("unit", ""),
                "fetched_at":  fetched_at,
                "type":        "market_data",
            })

        logger.info(f"  {label}: {len(rows)} data points")

    return results