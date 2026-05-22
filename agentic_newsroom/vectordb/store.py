"""
Vector database module using ChromaDB + sentence-transformers.

Embeds article chunks and stores them in a persistent Chroma collection
on your external disk. Supports semantic search across all ingested articles.

Model: all-MiniLM-L6-v2 (~90MB, runs locally, no API cost)
DB location: STORAGE_ROOT/vectordb/

Usage:
    from vectordb.store import VectorStore
    store = VectorStore()
    store.add_articles(enriched_articles)
    results = store.search("OPEC supply cut", n_results=10)
"""

import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "oil_news"


class VectorStore:
    def __init__(self):
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Vector store ready. Collection size: {self.collection.count()} chunks")

    def _chunk_id(self, url: str, chunk_index: int) -> str:
        """Stable unique ID for a chunk — hash of URL + chunk index."""
        raw = f"{url}::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def add_articles(self, articles: list[dict]) -> int:
        """
        Embed and store all chunks from a list of enriched articles.
        Skips chunks already in the collection (idempotent).
        Returns count of new chunks added.
        """
        texts      = []
        ids        = []
        metadatas  = []

        for article in articles:
            chunks     = article.get("chunks", [])
            extraction = article.get("extraction", [])
            url        = article.get("url", "")
            source     = article.get("source", "")
            title      = article.get("title", "")
            published  = article.get("published", "")
            art_type   = article.get("type", "")

            # Pull price direction from first successful extraction
            direction = ""
            sentiment = ""
            for ext in extraction:
                if ext.get("status") == "ok":
                    direction = ext.get("price_signals", {}).get("direction", "")
                    sentiment = ext.get("sentiment", "")
                    break

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_id = self._chunk_id(url, i)
                texts.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "url":         url[:500],
                    "source":      source,
                    "title":       title[:200],
                    "published":   published,
                    "type":        art_type,
                    "chunk_index": i,
                    "direction":   direction,
                    "sentiment":   sentiment,
                })

        if not texts:
            logger.info("No chunks to add.")
            return 0

        # Embed in batches of 64 to avoid memory issues
        batch_size  = 64
        added_count = 0

        for i in range(0, len(texts), batch_size):
            batch_texts     = texts[i:i+batch_size]
            batch_ids       = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            # upsert — safe to call multiple times, won't duplicate
            self.collection.upsert(
                ids        = batch_ids,
                embeddings = embeddings,
                documents  = batch_texts,
                metadatas  = batch_metadatas,
            )
            added_count += len(batch_texts)
            logger.info(f"  Embedded batch {i//batch_size + 1}: {len(batch_texts)} chunks")

        logger.info(f"Vector store updated. Total chunks in DB: {self.collection.count()}")
        return added_count

    def search(self, query: str, n_results: int = 10, direction_filter: str = "") -> list[dict]:
        """
        Semantic search across all stored chunks.

        Args:
            query: natural language query
            n_results: how many results to return
            direction_filter: optional — filter by "bullish", "bearish", "neutral"

        Returns list of dicts with chunk text and metadata.
        """
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        where = {"direction": direction_filter} if direction_filter else None

        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results        = n_results,
            where            = where,
            include          = ["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk":     results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "score":     1 - results["distances"][0][i],  # cosine similarity
            })

        return output

    def count(self) -> int:
        return self.collection.count()