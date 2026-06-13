"""
Run script for the Russia oil report.
Queries ChromaDB russia_oil collection and generates Analyst Note.
Designed to run after run_ingestion.py has completed.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

from config.settings import RUSSIA_WATCHLIST, RUSSIA_COLLECTION, VECTOR_DB_DIR
from report.russia_generator import generate_russia_report
import chromadb


def query_russia_articles(n_results: int = 20) -> tuple[list, list]:
    """Query ChromaDB for Russia-relevant articles and events."""
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

        # Try russia_oil collection first, fall back to oil_news
        try:
            collection = client.get_collection(RUSSIA_COLLECTION)
            logger.info(f"Using collection: {RUSSIA_COLLECTION}")
        except Exception:
            collection = client.get_collection("oil_news")
            logger.info("russia_oil collection not found, querying oil_news with Russia filter")

        # Query with Russia-focused terms
        query_terms = " ".join(RUSSIA_WATCHLIST[:10])
        results = collection.query(
            query_texts=[query_terms],
            n_results=min(n_results, collection.count()),
        )

        articles = []
        events = []

        if results and results.get("documents"):
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]

            for meta, doc in zip(metadatas, documents):
                articles.append({
                    "title":   meta.get("title", ""),
                    "source":  meta.get("source", ""),
                    "summary": doc[:300],
                    "url":     meta.get("url", ""),
                })

                # Extract any events from metadata
                event_type = meta.get("event_type", "")
                if event_type:
                    events.append({
                        "type":        event_type,
                        "description": meta.get("event_description", ""),
                        "urgency":     meta.get("urgency", "medium"),
                    })

        logger.info(f"Found {len(articles)} articles, {len(events)} events")
        return articles, events

    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return [], []


def main():
    logger.info("=== Russia Oil Report ===")

    articles, events = query_russia_articles(n_results=20)

    if not articles:
        logger.warning("No Russia-relevant articles found this cycle — skipping report")
        sys.exit(0)

    report = generate_russia_report(
        events=events,
        articles=articles,
        brent=None,       # adviser confirms from terminal
        urals_discount=None,
    )

    print("\n" + "="*60)
    print(report)
    print("="*60)
    logger.info("Russia report complete")


if __name__ == "__main__":
    main()