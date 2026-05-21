"""
Main ingestion runner.

Run manually:
    python run_ingestion.py

Or via cron (every 12 hours) — add this to crontab:
    0 6,18 * * * /path/to/venv/bin/python /path/to/agentic_newsroom/run_ingestion.py >> /Volumes/OilNewsDB/agentic_newsroom/logs/cron.log 2>&1

The script is safe to run multiple times — dedup prevents repeated articles.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Make sure local packages resolve regardless of cwd ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import STORAGE_ROOT, DEDUP_DB, LOG_DIR
from agentic_newsroom.ingestion.dedup import DedupRegistry
from agentic_newsroom.ingestion.rss_fetcher import fetch_all_feeds
from agentic_newsroom.ingestion.eia_fetcher import fetch_eia_data
from agentic_newsroom.ingestion.archive_writer import save_run
from agentic_newsroom.ingestion.audit_logger import write_audit

# ── Logging setup ─────────────────────────────────────────────────────────────
print('********',LOG_DIR)
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ingestion.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_ingestion")


def run():
    started_at = datetime.now(timezone.utc).isoformat()
    run_id     = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    errors: list[str] = []

    logger.info(f"=== Ingestion run started: {run_id} ===")
    logger.info(f"Storage root: {STORAGE_ROOT}")

    # ── 1. Dedup registry ──────────────────────────────────────────────────
    dedup = DedupRegistry(DEDUP_DB)

    # ── 2. Fetch RSS feeds ─────────────────────────────────────────────────
    articles: list[dict] = []
    try:
        articles = fetch_all_feeds(dedup)
        logger.info(f"RSS total new articles: {len(articles)}")
    except Exception as exc:
        msg = f"RSS fetch failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 3. Fetch EIA market data ───────────────────────────────────────────
    market_data: list[dict] = []
    try:
        market_data = fetch_eia_data()
        logger.info(f"EIA data points: {len(market_data)}")
    except Exception as exc:
        msg = f"EIA fetch failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 4. Save archive ────────────────────────────────────────────────────
    archive_path = ""
    try:
        saved = save_run(articles, market_data, run_id)
        archive_path = str(saved)
    except Exception as exc:
        msg = f"Archive write failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 5. Write audit log ─────────────────────────────────────────────────
    finished_at = datetime.now(timezone.utc).isoformat()
    status = "ok" if not errors else ("partial" if (articles or market_data) else "failed")

    write_audit(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        article_count=len(articles),
        market_data_count=len(market_data),
        archive_path=archive_path,
        errors=errors,
        status=status,
    )

    dedup.close()
    logger.info(f"=== Run complete: {status} | {len(articles)} articles | {len(market_data)} data points ===")

    # Exit code 1 on full failure so cron can alert
    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    run()