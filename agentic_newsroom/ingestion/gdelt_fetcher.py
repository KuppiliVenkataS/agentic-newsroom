"""
GDELT fetcher via Google BigQuery.

Queries the public GDELT GKG (Global Knowledge Graph) table for oil/energy
related news from today. Returns structured dicts shaped the same as RSS
articles so they land in the same JSON archive.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

from config.settings import GCP_PROJECT_ID, GCP_KEY_FILE

logger = logging.getLogger(__name__)

GDELT_QUERY = """
SELECT
    DATE,
    SourceCommonName,
    DocumentIdentifier AS url,
    V2Themes,
    V2Locations,
    V2Persons,
    V2Organizations,
    V2Tone,
    AllNames,
    Amounts
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    DATE(_PARTITIONTIME) = '{date}'
    AND (
        V2Themes LIKE '%CRUDE_OIL%'
        OR V2Themes LIKE '%ENV_OIL%'
        OR V2Themes LIKE '%ENERGY%'
        OR V2Themes LIKE '%OPEC%'
        OR V2Themes LIKE '%OIL_PRICE%'
        OR V2Themes LIKE '%NATURAL_GAS%'
        OR V2Themes LIKE '%PETROLEUM%'
        OR LOWER(AllNames) LIKE '%brent%'
        OR LOWER(AllNames) LIKE '%wti%'
    )
LIMIT 500
"""


def _build_client() -> bigquery.Client:
    key_path = Path(GCP_KEY_FILE).expanduser()
    if not key_path.exists():
        raise FileNotFoundError(f"GCP key file not found: {key_path}")

    credentials = service_account.Credentials.from_service_account_file(
        str(key_path),
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    return bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)


def _parse_tone(tone_str: str) -> dict:
    """
    V2Tone is comma-separated:
    Tone, Positive, Negative, Polarity, ARD, SGRD, WC
    """
    if not tone_str:
        return {}
    parts = tone_str.split(",")
    keys  = ["tone", "positive", "negative", "polarity", "ard", "sgrd", "word_count"]
    result = {}
    for i, key in enumerate(keys):
        if i < len(parts):
            try:
                result[key] = float(parts[i])
            except ValueError:
                result[key] = None
    return result


def fetch_gdelt_data() -> list[dict]:
    """
    Query GDELT GKG for today's oil/energy news.
    Returns a list of structured dicts.
    """
    if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID_HERE":
        logger.warning("GCP project ID not set — skipping GDELT fetch.")
        return []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    query = GDELT_QUERY.format(date=today)

    logger.info(f"Querying GDELT GKG for date: {today}")

    try:
        client     = _build_client()
        job        = client.query(query)
        rows       = list(job.result())
    except Exception as exc:
        logger.error(f"GDELT query failed: {exc}")
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    for row in rows:
        url = row.get("url", "").strip()
        if not url:
            continue

        results.append({
            "source":        "gdelt",
            "url":           url,
            "title":         "",
            "summary":       "",
            "published":     str(row.get("DATE", "")),
            "fetched_at":    fetched_at,
            "type":          "gdelt_gkg",
            "gdelt_source":  row.get("SourceCommonName", ""),
            "themes":        row.get("V2Themes", ""),
            "locations":     row.get("V2Locations", ""),
            "persons":       row.get("V2Persons", ""),
            "organisations": row.get("V2Organizations", ""),
            "tone":          _parse_tone(row.get("V2Tone", "")),
            "names":         row.get("AllNames", ""),
            "amounts":       row.get("Amounts", ""),
        })

    logger.info(f"GDELT: {len(results)} records returned")
    return results