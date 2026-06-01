"""
Weekly report runner — Friday evening.

Run manually:
    python run_weekly.py

Or via cron (Friday 18:00 London time — Mac Mini clock is Europe/London):
    0 18 * * 5 /path/to/venv/bin/python /path/to/agentic_newsroom/run_weekly.py >> /path/to/agentic_newsroom/logs/weekly.log 2>&1

This does NOT run ingestion. It reads the last 7 days of already-processed
cycles (produced by the twice-daily run_ingestion.py) and generates a weekly
retrospective. If the week's data is empty/stale, the guard suppresses it and
nothing is written.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import LOG_DIR
from report.weekly_generator import generate_weekly

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "weekly.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_weekly")


def run():
    logger.info("=== Weekly report run started ===")
    try:
        path = generate_weekly()
        if path:
            logger.info(f"=== Weekly complete: {path} ===")
        else:
            logger.warning("=== Weekly suppressed by guard (empty/stale week) — nothing generated ===")
    except Exception as exc:
        logger.error(f"Weekly run failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    run()