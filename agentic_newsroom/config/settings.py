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
    # upstream_online — Zephr SSO paywall, removed
    # tradewinds — Zephr SSO paywall, removed

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

    # Paywalled — headlines only (these work via RSS despite paywall on full articles)
    "argus_media":          "https://www.argusmedia.com/rss/news.rss",
    "platts_news":          "https://www.spglobal.com/commodityinsights/en/rss-feed/news",
    "icis_news":            "https://www.icis.com/explore/resources/news/rss/",
    # tradewinds — Zephr SSO paywall, removed

    # Financial / macro
    "ft_markets":           "https://www.ft.com/rss/home/uk",
    "ft_companies":         "https://www.ft.com/companies?format=rss",
    "bloomberg_markets":    "https://feeds.bloomberg.com/markets/news.rss",
    "nasdaq_commodities":   "https://www.nasdaq.com/feed/rssoutbound?category=Commodities",
    "investing_oil":        "https://www.investing.com/rss/news_14.rss",

    # Trump / US policy
    "state_department":     "https://www.state.gov/rss-feeds/",
    "the_hill_energy":      "https://thehill.com/policy/energy-environment/feed/",
    "ap_top_news":          "https://feeds.ap.org/rss/apf-topnews",
    "reuters_energy":       "https://feeds.reuters.com/reuters/businessNews",
    # reuters_politics — DNS fails
    # ap_energy (rsshub) — 403
    # axios_energy — 404

    # Russia / sanctions / shadow fleet
    "kyiv_independent":     "https://kyivindependent.com/feed/",
    "moscow_times":         "https://www.themoscowtimes.com/rss/news",
    "bellcat":              "https://www.bellingcat.com/feed/",
    "tanker_trackers":      "https://www.tankertrackers.com/rss",
    "maritime_executive":   "https://maritime-executive.com/rss",
    "occrp":                "https://www.occrp.org/en/rss",
}

# ── Request settings ──────────────────────────────────────────────────────────
REQUEST_TIMEOUT       = 20
MAX_ARTICLES_PER_FEED = 15    # reduced from 50 — top 15 per feed is enough
MAX_TOTAL_RSS         = 150   # hard ceiling across all feeds combined
USER_AGENT            = "AgenticNewsroom/1.0"

# ── Anthropic API ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
# USE_CLAUDE_EXTRACTION  = os.getenv("USE_CLAUDE_EXTRACTION", "false").lower() == "false"  # set true when rate limits allow
USE_CLAUDE_EXTRACTION = os.getenv("USE_CLAUDE_EXTRACTION", "false").lower() == "true"
USE_CLAUDE_REPORT      = os.getenv("USE_CLAUDE_REPORT", "true").lower() == "true"        # always use Claude for report

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")

# ── Dev flags ─────────────────────────────────────────────────────────────────
SKIP_EXTRACTION = os.getenv("SKIP_EXTRACTION", "false").lower() == "true"
SKIP_INGESTION  = os.getenv("SKIP_INGESTION",  "false").lower() == "true"

# ── Run cadence ───────────────────────────────────────────────────────────────
CYCLE_HOURS = 12

# ── User interest watchlist (THEMED) ──────────────────────────────────────────
# Organised into themes so you can shift focus as the world's attention moves,
# WITHOUT editing code. Each theme is a list of keywords; set a theme to []
# (or comment its terms out) to mute it when it stops driving prices.
#
# Design principle: keep this LEAN. When one driver dominates (e.g. an active
# Iran conflict), that theme carries the weight and the dormant themes stay
# short. Activate a dormant theme only when a real story is forming — the
# system's "emerging story" scan (focus/emerging.py) will prompt the adviser
# when something new surfaces in the broad feeds.
#
# Matching is case-insensitive, against article title + summary.
# Any article matching one or more terms gets WATCHLIST_BOOST added to its
# importance_score (multiple matches stack, capped at 1.0).

WATCHLIST_THEMES = {
    # ── ACTIVE: the dominant driver(s) right now ──────────────────────────
    "geopolitics_iran": [
        "hormuz", "strait of hormuz", "iran", "iranian", "irgc",
        "fordow", "natanz", "iran nuclear", "iran deal", "iran sanctions",
    ],
    "geopolitics_israel": [
        "gaza", "israel", "netanyahu", "hezbollah", "southern lebanon", "west bank",
    ],
    "us_policy": [
        "trump", "white house", "executive order", "drill baby drill",
        "strategic reserve", "spr release",
    ],
    "deescalation": [
        "ceasefire", "peace talks", "negotiations", "agreement reached", "hormuz reopened",
    ],
    "russia_ukraine": [
        "russia oil", "russian crude", "urals", "urals discount",
        "shadow fleet", "dark fleet", "price cap", "oil price cap",
        "arctic lng", "yamal", "novatek", "rosneft", "lukoil",
        "novorossiysk", "primorsk", "ust-luga", "baltic tanker",
        "cpc pipeline", "caspian pipeline",
        "ukraine drone oil", "ukraine strikes refinery",
        "gazprom", "transneft",
        "st petersburg port", "kronstadt",
        "russia sanctions", "ofac russia",
        "g7 price cap", "insurance ban", "ukraine strikes refinery",
        "ukraine drone refinery",
        "ukraine attack oil",
        "ukraine strikes oil",
        "ryazan refinery",
        "saratov refinery",
        "lukoil refinery",
        "rosneft refinery",
        "novoshakhtinsk",
        "tuapse refinery",
        "oil depot strike",
        "fuel depot ukraine",
        "pipeline rupture russia",
        "transneft attack",
    ],

    # ── DORMANT: activate (add terms) when the emerging-story scan flags one ──
    # Kept short on purpose. These are the recurring non-conflict drivers the
    # adviser turns up when attention shifts there.
    "weather_climate": [
        # e.g. "heatwave", "cold snap", "freeze-off", "hurricane", "tropical storm"
    ],
    "waterways_logistics": [
        # e.g. "rhine", "low water", "mississippi barge", "suez", "panama canal"
    ],
    "natural_disaster": [
        # e.g. "earthquake", "tsunami", "refinery fire", "pipeline rupture"
    ],
    "demand_macro": [
        # e.g. "china demand", "india demand", "dollar", "recession", "refinery margin"
    ],
}

# Flatten themes into the flat list the rest of the pipeline consumes.
# (Downstream code is unchanged — it still reads USER_WATCHLIST.)
# De-duplicate while preserving order.
USER_WATCHLIST = list(dict.fromkeys(
    kw for terms in WATCHLIST_THEMES.values() for kw in terms
))

# Broad/general feeds used by the emerging-story scan (focus/emerging.py).
# These are general-coverage sources that surface a NEW driver before it
# becomes a dominant theme — the early-warning sensor. Keep this to the
# genuinely broad ones, not the Iran/ME-specific feeds.
BROAD_FEEDS = ["reuters_energy", "eia_petroleum_weekly", "ft_markets", "bloomberg_markets"]

# How much to boost importance for watchlist matches (additive, capped at 1.0)
# 0.2 = a watchlist match adds 0.2 to the LLM's importance_score
# Set to 0.0 to disable the watchlist boost without removing the list
WATCHLIST_BOOST = 0.20

# ── Russia report ─────────────────────────────────────────────────────────────
RUSSIA_COLLECTION = "russia_oil"
RUSSIA_WATCHLIST  = WATCHLIST_THEMES["russia_ukraine"]