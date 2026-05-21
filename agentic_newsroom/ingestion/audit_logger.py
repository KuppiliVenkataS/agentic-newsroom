"""
Audit logger.

Appends one JSON line per run to a rolling monthly log file.
This lets you quickly check run history, counts, and failures
without opening the large archive files.

Log file: <LOG_DIR>/audit_YYYY-MM.jsonl
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import LOG_DIR

logger = logging.getLogger(__name__)


def write_audit(
    run_id: str,
    started_at: str,
    finished_at: str,
    article_count: int,
    market_data_count: int,
    archive_path: str,
    errors: list[str],
    status: str,          # "ok" | "partial" | "failed"
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow()
    log_file = LOG_DIR / f"audit_{now.year:04d}-{now.month:02d}.jsonl"

    entry = {
        "run_id":            run_id,
        "started_at":        started_at,
        "finished_at":       finished_at,
        "article_count":     article_count,
        "market_data_count": market_data_count,
        "archive_path":      archive_path,
        "errors":            errors,
        "status":            status,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info(f"Audit written: status={status}, articles={article_count}")