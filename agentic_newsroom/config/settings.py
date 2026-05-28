"""
Central config — all sensitive values loaded from .env
Safe to commit to GitHub — no hardcoded secrets.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Storage root ──────────────────────────────────────────────────────────────
# Local Mac Mini path. On Render, uses /tmp.
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "/tmp/agentic_newsroom"))
RAW_DIR       = STORAGE_ROOT / "data" / "raw"
LOG_DIR       = STORAGE_ROOT / "data" / "logs"
DEDUP_DB      = STORAGE_ROOT / "data" / "dedup.db"
PROCESSED_DIR = STORAGE_ROOT / "data" / "processed"
VECTOR_DB_DIR = STORAGE_ROOT / "vectordb"
GRAPH_DB_DIR  = STORAGE_ROOT / "graph" / "kuzu"
REPORT_DIR    = STORAGE_ROOT / "reports"

# ── Cloudflare R2 ─────────────────────────────────────────────────────────────
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET     = os.getenv("R2_BUCKET", "oil-reports")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# ── Portal ────────────────────────────────────────────────────────────────────
PORTAL_PASSWORD   = os.getenv("PORTAL_PASSWORD", "")
PORTAL_SECRET_KEY = os.getenv("PORTAL_SECRET_KEY", "change-this-secret-key")
PORTAL_URL        = os.getenv("PORTAL_URL", "http://localhost:8080")

# ── Email / SMTP ──────────────────────────────────────────────────────────────
SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", "")
FIXED_RECIPIENTS = []

# ── OilPrice API ──────────────────────────────────────────────────────────────
OILPRICE_API_KEY = os.getenv("OILPRICE_API_KEY", "")

# ── EIA API ───────────────────────────────────────────────────────────────────
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
EIA_SERIES  = {
    "wti_spot":     "PET.RWTC.D",
    "brent_spot":   "PET.RBRTE.D",
    "us_inventory": "PET.WCESTUS1.W",
    "us_production":"PET.WCRFPUS2.W",
}

# ── GCP / GDELT ───────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_KEY_FILE   = os.getenv("GCP_KEY_FILE", "")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS = {
    # Oil & energy specialist
    "oilprice_news":        "https://oilprice.com/rss/main",
    "rigzone":              "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    "eia_petroleum_weekly": "https://www.eia.gov/petroleum/weekly/includes/archive.php?format=rss",
    "energy_voice":         "https://www.energyvoice.com/feed/",
    "upstream_online":      "https://www.upstreamonline.com/rss",

    # Shipping & tankers
    "hellenic_shipping":    "https://www.hellenicshippingnews.com/feed/",

    # Geopolitical — Middle East, Iran, Russia
    "middle_east_eye":      "https://www.middleeasteye.net/rss",
    "al_monitor":           "https://www.al-monitor.com/rss",
    "iran_international":   "https://www.iranintl.com/en/rss",
    "jerusalem_post":       "https://www.jpost.com/rss/rssfeedsfrontpage.aspx",
    "times_of_israel":      "https://www.timesofisrael.com/feed/",
    "arab_news":            "https://www.arabnews.com/rss.xml",
    "defense_one":          "https://www.defenseone.com/rss/all/",
    "war_on_rocks":         "https://warontherocks.com/feed/",

    # Paywalled — headlines only
    "argus_media":          "https://www.argusmedia.com/rss/news.rss",
    "platts_news":          "https://www.spglobal.com/commodityinsights/en/rss-feed/news",
    "icis_news":            "https://www.icis.com/explore/resources/news/rss/",
    "tradewinds":           "https://www.tradewindsnews.com/rss",

    # Financial / macro
    "ft_markets":           "https://www.ft.com/rss/home/uk",
    "ft_companies":         "https://www.ft.com/companies?format=rss",
    "bloomberg_markets":    "https://feeds.bloomberg.com/markets/news.rss",
    "nasdaq_commodities":   "https://www.nasdaq.com/feed/rssoutbound?category=Commodities",
    "investing_oil":        "https://www.investing.com/rss/news_14.rss",

    # Trump / US policy
    "white_house":          "https://www.whitehouse.gov/feed/",
    "state_department":     "https://www.state.gov/rss-feeds/",
    "reuters_politics":     "https://feeds.reuters.com/Reuters/PoliticsNews",
    "ap_politics":          "https://feeds.ap.org/rss/apf-politics",
    "politico":             "https://www.politico.com/rss/politicopicks.xml",
    "axios":                "https://api.axios.com/feed/",
    "the_hill_energy":      "https://thehill.com/policy/energy-environment/feed/",
    "nitter_trump":         "https://nitter.net/realDonaldTrump/rss",
}

# ── Request settings ──────────────────────────────────────────────────────────
REQUEST_TIMEOUT       = 20
MAX_ARTICLES_PER_FEED = 50
USER_AGENT            = "AgenticNewsroom/1.0"

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Dev flags ─────────────────────────────────────────────────────────────────
SKIP_EXTRACTION = os.getenv("SKIP_EXTRACTION", "false").lower() == "true"
SKIP_INGESTION  = os.getenv("SKIP_INGESTION",  "false").lower() == "true"

# ── Run cadence ───────────────────────────────────────────────────────────────
CYCLE_HOURS = 12

# ── User interest watchlist ───────────────────────────────────────────────────
# Keywords that always get elevated importance regardless of LLM extraction.
# Edit this list freely — no code changes needed.
# Matching is case-insensitive, against article title + summary.
#
# How it works: any article matching one or more of these gets a
# WATCHLIST_BOOST added to its importance_score before scoring.
# Multiple matches stack up to a cap of 1.0.
#
# Current focus: Iran conflict, Hormuz, Trump remarks, peace talks
USER_WATCHLIST = [
    # Iran / Hormuz
    "hormuz",
    "strait of hormuz",
    "iran",
    "iranian",
    "irgc",
    "fordow",
    "natanz",
    "iran nuclear",
    "iran deal",
    "iran sanctions",
    "air strikes",

    # Trump / US policy
    "trump",
    "white house",
    "executive order",
    "iran deal",
    "drill baby drill",
    "strategic reserve",
    "spr release",

    # Peace talks / de-escalation (bearish risk)
    "ceasefire",
    "peace talks",
    "negotiations",
    "agreement reached",
    "hormuz reopened",

    # Russia / Ukraine oil angle
    "novorossiysk",
    "caspian pipeline",
    "russia oil",
    "ukraine drone oil",

    # Gaza / Israel / Lebanon
    "gaza",
    "israel",
    "netanyahu",
    "hezbollah",
    "southern lebanon",
    "west bank",

    # Add your own below — one string per line
]

# How much to boost importance for watchlist matches (additive, capped at 1.0)
# 0.2 = a watchlist match adds 0.2 to the LLM's importance_score
# Set to 0.0 to disable the watchlist boost without removing the list
WATCHLIST_BOOST = 0.20