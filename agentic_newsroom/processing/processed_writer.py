"""
Processed archive writer.

Saves the enriched articles (cleaned text + extraction results)
as a separate JSON file alongside the raw archive.

File layout:
  <PROCESSED_DIR>/
    2026/
      05/
        2026-05-21_06-00-00_processed.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)


def save_processed(articles: list[dict], run_id: str) -> Path:
    now = datetime.utcnow()
    date_dir = PROCESSED_DIR / f"{now.year:04d}" / f"{now.month:02d}"
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_id}_processed.json"
    filepath = date_dir / filename

    payload = {
        "run_id":          run_id,
        "processed_at":    now.isoformat(),
        "article_count":   len(articles),
        "articles":        articles,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved processed output: {filepath}")
    return filepath