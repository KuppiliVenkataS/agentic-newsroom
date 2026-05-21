"""
GDELT fetcher via Google BigQuery.

Queries the public GDELT GKG (Global Knowledge Graph) table for oil/energy
related news from the last 12 hours. Returns structured dicts shaped the same
as RSS articles so they land in the same JSON archive.

GDELT GKG schema reference:
https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/

Cost: Each query scans ~50-200MB. Well within the 1TB/month free tier.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account

from config.settings import GCP_PROJECT_ID, GCP_KEY_FILE

logger = logging.getLogger(__name__)

# Oil and energy related themes in GDELT's taxonomy
OIL_THEMES = [
    "CRUDE_OIL",
    "ENV_OIL",
    "ENERGY",
    "OPEC",
    "OIL_PRICE",
    "NATURAL_GAS",
    "PETROLEUM",
]

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
    Amounts,
    TranslationInfo,
    Extras
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    _PARTITIONTIME >= TIMESTAMP('{start}')
    AND _PARTITIONTIME < TIMESTAMP('{end}')
    AND (
        LOWER(V2Themes) LIKE '%oil%'
        OR LOWER(V2Themes) LIKE '%opec%'
        OR LOWER(V2Themes) LIKE '%petroleum%'
        OR LOWER(V2Themes) LIKE '%crude%'
        OR LOWER(V2Themes) LIKE '%energy%'
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
    V2Tone is a comma-separated string:
    Tone, Positive, Negative, Polarity, ARD, SGRD, WC
    Returns a dict with named fields.
    """
    if not tone_str:
        return {}
    parts = tone_str.split(",")
    keys = ["tone", "positive", "negative", "polarity", "ard", "sgrd", "word_count"]
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
    Query GDELT GKG for oil/energy news from the last 12 hours.
    Returns a list of structured dicts.
    """
    if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID_HERE":
        logger.warning("GCP project ID not set — skipping GDELT fetch.")
        return []

    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S")
    end = now.strftime("%Y-%m-%d %H:%M:%S")

    query = GDELT_QUERY.format(start=start, end=end)

    logger.info(f"Querying GDELT GKG: {start} → {end}")

    try:
        client = _build_client()
        job = client.query(query)
        rows = list(job.result())
    except Exception as exc:
        logger.error(f"GDELT query failed: {exc}")
        return []

    fetched_at = now.isoformat()
    results: list[dict] = []

    for row in rows:
        url = row.get("url", "").strip()
        if not url:
            continue

        results.append({
            "source":        "gdelt",
            "url":           url,
            "title":         "",                          # GKG doesn't carry titles
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