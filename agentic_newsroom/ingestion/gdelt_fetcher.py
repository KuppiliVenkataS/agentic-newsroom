# """
# GDELT fetcher via Google BigQuery.

# Queries the public GDELT GKG (Global Knowledge Graph) table for oil/energy
# related news from today. Returns structured dicts shaped the same as RSS
# articles so they land in the same JSON archive.
# """

# import logging
# from datetime import datetime, timezone
# from pathlib import Path

# from google.cloud import bigquery
# from google.oauth2 import service_account

# from config.settings import GCP_PROJECT_ID, GCP_KEY_FILE

# logger = logging.getLogger(__name__)

# GDELT_QUERY = """
# SELECT
#     DATE,
#     SourceCommonName,
#     DocumentIdentifier AS url,
#     V2Themes,
#     V2Locations,
#     V2Persons,
#     V2Organizations,
#     V2Tone,
#     AllNames,
#     Amounts
# FROM
#     `gdelt-bq.gdeltv2.gkg_partitioned`
# WHERE
#     DATE(_PARTITIONTIME) = '{date}'
#     AND (
#         V2Themes LIKE '%CRUDE_OIL%'
#         OR V2Themes LIKE '%ENV_OIL%'
#         OR V2Themes LIKE '%ENERGY%'
#         OR V2Themes LIKE '%OPEC%'
#         OR V2Themes LIKE '%OIL_PRICE%'
#         OR V2Themes LIKE '%NATURAL_GAS%'
#         OR V2Themes LIKE '%PETROLEUM%'
#         OR LOWER(AllNames) LIKE '%brent%'
#         OR LOWER(AllNames) LIKE '%wti%'
#     )
# LIMIT 500
# """


# def _build_client() -> bigquery.Client:
#     key_path = Path(GCP_KEY_FILE).expanduser()
#     if not key_path.exists():
#         raise FileNotFoundError(f"GCP key file not found: {key_path}")

#     credentials = service_account.Credentials.from_service_account_file(
#         str(key_path),
#         scopes=["https://www.googleapis.com/auth/bigquery"]
#     )
#     return bigquery.Client(project=GCP_PROJECT_ID, credentials=credentials)


# def _parse_tone(tone_str: str) -> dict:
#     """
#     V2Tone is comma-separated:
#     Tone, Positive, Negative, Polarity, ARD, SGRD, WC
#     """
#     if not tone_str:
#         return {}
#     parts = tone_str.split(",")
#     keys  = ["tone", "positive", "negative", "polarity", "ard", "sgrd", "word_count"]
#     result = {}
#     for i, key in enumerate(keys):
#         if i < len(parts):
#             try:
#                 result[key] = float(parts[i])
#             except ValueError:
#                 result[key] = None
#     return result


# def fetch_gdelt_data() -> list[dict]:
#     """
#     Query GDELT GKG for today's oil/energy news.
#     Returns a list of structured dicts.
#     """
#     if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID_HERE":
#         logger.warning("GCP project ID not set — skipping GDELT fetch.")
#         return []

#     today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
#     query = GDELT_QUERY.format(date=today)

#     logger.info(f"Querying GDELT GKG for date: {today}")

#     try:
#         client     = _build_client()
#         job        = client.query(query)
#         rows       = list(job.result())
#     except Exception as exc:
#         logger.error(f"GDELT query failed: {exc}")
#         return []

#     fetched_at = datetime.now(timezone.utc).isoformat()
#     results: list[dict] = []

#     for row in rows:
#         url = row.get("url", "").strip()
#         if not url:
#             continue

#         results.append({
#             "source":        "gdelt",
#             "url":           url,
#             "title":         "",
#             "summary":       "",
#             "published":     str(row.get("DATE", "")),
#             "fetched_at":    fetched_at,
#             "type":          "gdelt_gkg",
#             "gdelt_source":  row.get("SourceCommonName", ""),
#             "themes":        row.get("V2Themes", ""),
#             "locations":     row.get("V2Locations", ""),
#             "persons":       row.get("V2Persons", ""),
#             "organisations": row.get("V2Organizations", ""),
#             "tone":          _parse_tone(row.get("V2Tone", "")),
#             "names":         row.get("AllNames", ""),
#             "amounts":       row.get("Amounts", ""),
#         })

#     logger.info(f"GDELT: {len(results)} records returned")
#     return results

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

# 24-hour query — primary signal
GDELT_QUERY_24H = """
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
        OR V2Themes LIKE '%ARMEDCONFLICT%'
        OR V2Themes LIKE '%TAX_FNCACT_MINISTER%'
        OR LOWER(AllNames) LIKE '%brent%'
        OR LOWER(AllNames) LIKE '%wti%'
        OR LOWER(AllNames) LIKE '%hormuz%'
        OR LOWER(AllNames) LIKE '%iran%'
        OR LOWER(AllNames) LIKE '%opec%'
        OR LOWER(AllNames) LIKE '%trump%'
    )
LIMIT 150
"""

# 7-day query — background context and brewing stories
GDELT_QUERY_7D = """
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
    DATE(_PARTITIONTIME) BETWEEN DATE_SUB('{date}', INTERVAL 7 DAY) AND DATE_SUB('{date}', INTERVAL 1 DAY)
    AND (
        V2Themes LIKE '%CRUDE_OIL%'
        OR V2Themes LIKE '%ENV_OIL%'
        OR V2Themes LIKE '%OPEC%'
        OR V2Themes LIKE '%OIL_PRICE%'
        OR V2Themes LIKE '%ARMEDCONFLICT%'
        OR LOWER(AllNames) LIKE '%hormuz%'
        OR LOWER(AllNames) LIKE '%iran%'
        OR LOWER(AllNames) LIKE '%houthi%'
        OR LOWER(AllNames) LIKE '%russia%'
        OR LOWER(AllNames) LIKE '%ukraine%'
        OR LOWER(AllNames) LIKE '%trump%'
        OR LOWER(AllNames) LIKE '%sanctions%'
    )
LIMIT 100
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


def _run_query(client, query: str, window: str) -> list[dict]:
    """Run a single GDELT query and return structured records tagged by window."""
    try:
        rows = list(client.query(query).result())
    except Exception as exc:
        logger.error(f"GDELT {window} query failed: {exc}")
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    results    = []

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
            "window":        window,   # "24h" or "7d"
            "gdelt_source":  row.get("SourceCommonName", ""),
            "themes":        row.get("V2Themes", ""),
            "locations":     row.get("V2Locations", ""),
            "persons":       row.get("V2Persons", ""),
            "organisations": row.get("V2Organizations", ""),
            "tone":          _parse_tone(row.get("V2Tone", "")),
            "names":         row.get("AllNames", ""),
            "amounts":       row.get("Amounts", ""),
        })
    return results


def fetch_gdelt_data() -> list[dict]:
    """
    Query GDELT GKG using two windows:
    - 24h: primary signal for prediction and sentiment
    - 7d:  background context for brewing stories
    Returns combined list tagged by window.
    """
    if not GCP_PROJECT_ID or GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID_HERE":
        logger.warning("GCP project ID not set — skipping GDELT fetch.")
        return []

    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client = _build_client()

    # 24-hour window
    logger.info(f"Querying GDELT 24h window: {today}")
    records_24h = _run_query(client, GDELT_QUERY_24H.format(date=today), "24h")
    logger.info(f"  24h records: {len(records_24h)}")

    # 7-day window
    logger.info(f"Querying GDELT 7d window: last 7 days before {today}")
    records_7d  = _run_query(client, GDELT_QUERY_7D.format(date=today), "7d")
    logger.info(f"  7d records: {len(records_7d)}")

    total = records_24h + records_7d
    logger.info(f"GDELT total: {len(total)} records")
    return total