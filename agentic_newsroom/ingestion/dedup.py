"""
Deduplication registry backed by SQLite.

Tracks URLs and title fingerprints so the same story from multiple
feeds is only processed once per cycle.
"""

import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime


def _fingerprint(text: str) -> str:
    """SHA-256 of lowercased, stripped text — used for near-duplicate titles."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


class DedupRegistry:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_urls (
                url_hash  TEXT PRIMARY KEY,
                url       TEXT,
                first_seen TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_titles (
                title_hash TEXT PRIMARY KEY,
                title      TEXT,
                first_seen TEXT
            )
        """)
        self.conn.commit()

    def is_duplicate(self, url: str, title: str) -> bool:
        url_hash   = _fingerprint(url)
        title_hash = _fingerprint(title)

        cur = self.conn.execute(
            "SELECT 1 FROM seen_urls WHERE url_hash = ?", (url_hash,)
        )
        if cur.fetchone():
            return True

        cur = self.conn.execute(
            "SELECT 1 FROM seen_titles WHERE title_hash = ?", (title_hash,)
        )
        if cur.fetchone():
            return True

        return False

    def register(self, url: str, title: str):
        now = datetime.utcnow().isoformat()
        url_hash   = _fingerprint(url)
        title_hash = _fingerprint(title)

        self.conn.execute(
            "INSERT OR IGNORE INTO seen_urls VALUES (?, ?, ?)",
            (url_hash, url[:500], now)
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_titles VALUES (?, ?, ?)",
            (title_hash, title[:500], now)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()