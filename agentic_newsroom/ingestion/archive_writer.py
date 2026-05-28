"""
Archive writer.

Saves every ingestion run as a dated JSON file on the external disk.
This is your source of truth — never delete these files.

File layout on disk:
  <RAW_DIR>/
    2026/
      05/
        2026-05-21_14-00-00_run.json
        2026-05-21_14-00-00_run.json
        ...
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)


def save_run(articles: list[dict], market_data: list[dict],
             run_id: str, prediction: dict = None) -> Path:
    """
    Write one complete ingestion run to a JSON file.
    Includes prediction dict so the scorer can evaluate it next run.
    Returns the path of the saved file.
    """
    now = datetime.utcnow()
    date_dir = RAW_DIR / f"{now.year:04d}" / f"{now.month:02d}"
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_id}_run.json"
    filepath = date_dir / filename

    payload = {
        "run_id":              run_id,
        "run_timestamp":       now.isoformat(),
        "article_count":       len(articles),
        "market_data_count":   len(market_data),
        "articles":            articles,
        "market_data":         market_data,
        "prediction":          prediction or {},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved run to {filepath} ({len(articles)} articles, {len(market_data)} data points)")
    return filepath