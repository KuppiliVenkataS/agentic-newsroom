"""
Central config. Edit paths and API keys here before running.
"""

from pathlib import Path

# ── Storage root ──────────────────────────────────────────────────────────────
# Point this at your 2TB external disk.
# Default assumes it's mounted at /Volumes/projects/OilNewsDB on macOS.
# Change to wherever your drive mounts.
STORAGE_ROOT = Path("/Volumes/projects/OilNewsDB/agentic_newsroom")

RAW_DIR  = STORAGE_ROOT / "raw"      # one JSON file per source per run
LOG_DIR  = STORAGE_ROOT / "logs"     # audit logs
DEDUP_DB = STORAGE_ROOT / "dedup.db" # SQLite dedup registry

# ── EIA API ───────────────────────────────────────────────────────────────────
# Free key from https://www.eia.gov/opendata/
EIA_API_KEY = "8rXxybukI1E0yjQxUVFS0JPNL4QtmQiQrceec1oD"

# EIA series to pull each run
EIA_SERIES = {
    "wti_spot":    "PET.RWTC.D",     # WTI crude spot price (daily)
    "brent_spot":  "PET.RBRTE.D",    # Brent crude spot price (daily)
    "us_inventory":"PET.WCESTUS1.W", # US crude inventories (weekly)
    "us_production":"PET.WCRFPUS2.W",# US crude production (weekly)
}

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS = {
    "ft_markets":      "https://www.ft.com/rss/home/uk",
    "ft_companies":    "https://www.ft.com/companies?format=rss",
    "reuters_business":"https://feeds.reuters.com/reuters/businessNews",
    "reuters_energy":  "https://feeds.reuters.com/reuters/UKenergyNews",
    "bloomberg_energy":"https://feeds.bloomberg.com/energy/news.rss",
    "oilprice_news":   "https://oilprice.com/rss/main",
    "rigzone":         "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    "platts_oil":      "https://www.spglobal.com/commodityinsights/en/rss-feed/oil",
    "eia_petroleum_weekly": "https://www.eia.gov/petroleum/weekly/includes/archive.php?format=rss",
    "opec_news":       "https://www.opec.org/opec_web/en/press_room/4_rss_latest.htm",
}

# ── Request settings ──────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 20          # seconds per HTTP request
MAX_ARTICLES_PER_FEED = 50    # cap per feed per run to stay polite
USER_AGENT = "AgenticNewsroom/1.0 (research project; contact: santhilatakv@gmail.com)"

# ── Run cadence ───────────────────────────────────────────────────────────────
CYCLE_HOURS = 12              # how often the full pipeline runs