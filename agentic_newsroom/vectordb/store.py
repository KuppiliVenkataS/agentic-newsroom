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
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config.settings import VECTOR_DB_DIR

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "oil_news"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Recency-decay and staleness-exemption tuning ────────────────────────────
RECENCY_HALF_LIFE_DAYS   = 3.5   # importance contribution halves every ~3.5 days
HARD_CUTOFF_DAYS         = 7     # candidates older than this are dropped by default
HIGH_IMPORTANCE_FLOOR    = 0.7   # matches the prompt's "HIGH-IMPORTANCE EVENTS" threshold
FRESH_UPDATE_WINDOW_DAYS = 2      # window checked for a possible fresher update on the same topic


def _parse_published(published: str):
    """Parse a published timestamp string to a UTC datetime, or None if unparseable/empty."""
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _age_days(published: str, now: datetime) -> float | None:
    dt = _parse_published(published)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 86400.0


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

            # Pull signals from first successful extraction
            direction   = ""
            sentiment   = ""
            importance  = 0.0
            urgency     = "low"
            hormuz_risk = False
            is_breaking = False
            for ext in extraction:
                if ext.get("status") == "ok":
                    direction   = ext.get("price_signals", {}).get("direction", "")
                    sentiment   = ext.get("sentiment", "")
                    importance  = float(ext.get("importance_score", 0.0))
                    hormuz_risk = bool(ext.get("hormuz_risk", False))
                    is_breaking = bool(ext.get("is_breaking", False))
                    events      = ext.get("events", [])
                    urgency     = events[0].get("urgency", "low") if events else "low"
                    break

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_id = self._chunk_id(url, i)
                texts.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "url":            url[:500],
                    "source":         source,
                    "title":          title[:200],
                    "published":      published,
                    "type":           art_type,
                    "chunk_index":    i,
                    "direction":      direction,
                    "sentiment":      sentiment,
                    "importance":     round(importance, 3),
                    "urgency":        urgency,
                    "hormuz_risk":    str(hormuz_risk),   # Chroma metadata must be str/int/float
                    "is_breaking":    str(is_breaking),
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

    def search_important(self, query: str, n_results: int = 10,
                         candidate_pool: int = 40) -> list[dict]:
        """
        Semantic search re-ranked by importance score, with recency decay.

        Fetches a larger candidate pool, then re-ranks combining cosine
        similarity (60%) and a RECENCY-DECAYED importance score (40%).
        High-urgency and breaking articles surface above semantically
        similar but low-importance ones, but that boost fades with age
        rather than persisting forever.

        Recency handling:
        - importance contribution decays with an exponential half-life of
          RECENCY_HALF_LIFE_DAYS, so old "breaking" stories stop dominating.
        - candidates older than HARD_CUTOFF_DAYS are dropped UNLESS they
          are high-importance (>= HIGH_IMPORTANCE_FLOOR) AND no fresher
          same-topic article exists within FRESH_UPDATE_WINDOW_DAYS days
          (checked via a second similarity query) — these are treated as
          still-developing stories with no update and exempted at a floor
          score rather than decayed to zero.
        - if a fresher same-topic article DOES exist, a Haiku call checks
          whether it contradicts/supersedes the older claim. If yes, the
          older candidate is dropped (the fresher one will surface on its
          own merits). If no, the normal hard cutoff applies — old news
          with nothing new to say has no reason to be exempted.

        Args:
            query:          natural language query
            n_results:      final results to return
            candidate_pool: how many candidates to fetch before re-ranking
        """
        now = datetime.now(timezone.utc)

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results        = min(candidate_pool, self.collection.count() or 1),
            include          = ["documents", "metadatas", "distances"]
        )

        candidates = []
        for i in range(len(results["ids"][0])):
            meta       = results["metadatas"][0][i]
            cosine_sim = 1 - results["distances"][0][i]
            importance = float(meta.get("importance", 0.0))
            age        = _age_days(meta.get("published", ""), now)

            # Urgency bonus on top of importance
            urgency_bonus = {"critical": 0.3, "high": 0.2, "medium": 0.0, "low": -0.1}.get(
                meta.get("urgency", "low"), 0.0
            )
            breaking_bonus = 0.1 if meta.get("is_breaking") == "True" else 0.0

            candidate = {
                "chunk":      results["documents"][0][i],
                "metadata":   meta,
                "score":      round(cosine_sim, 4),
                "_cosine":    cosine_sim,
                "_importance": importance,
                "_age_days":  age,
                "_urgency_bonus": urgency_bonus,
                "_breaking_bonus": breaking_bonus,
            }

            if age is None:
                # No parseable publish date — can't apply recency logic.
                # Treat conservatively as undecayed (old behaviour) rather
                # than silently dropping data we can't reason about.
                decay = 1.0
            else:
                decay = math.exp(-age / RECENCY_HALF_LIFE_DAYS)

            if age is not None and age > HARD_CUTOFF_DAYS:
                if importance >= HIGH_IMPORTANCE_FLOOR:
                    candidate["_exemption_candidate"] = True
                else:
                    continue  # stale and not important enough — drop
            else:
                candidate["_exemption_candidate"] = False

            candidate["_decay"] = decay
            candidates.append(candidate)

        # Resolve exemption candidates: check for a fresher update, and if
        # found, check whether it contradicts/supersedes the older claim.
        kept = []
        for c in candidates:
            if not c.get("_exemption_candidate"):
                kept.append(c)
                continue

            fresh = self._find_fresh_update(c, now)
            if fresh is None:
                # Still-developing story, no newer reporting on it at all —
                # exempt from the cutoff, but score at a floor (not boosted
                # as if it were fresh breaking news).
                c["_decay"] = max(c["_decay"], 0.3)
                kept.append(c)
                logger.info(
                    f"Exempting stale high-importance article from cutoff "
                    f"(age={c['_age_days']:.1f}d, importance={c['_importance']:.2f}): "
                    f"{c['metadata'].get('title', '')[:80]}"
                )
                continue

            contradicted = self._check_contradiction(c, fresh)
            if contradicted:
                logger.info(
                    f"Dropping superseded article (age={c['_age_days']:.1f}d): "
                    f"{c['metadata'].get('title', '')[:80]}"
                )
                continue  # drop — the fresh article supersedes it and will
                          # surface on its own merits
            else:
                # Fresher coverage exists but doesn't contradict it — it's
                # just old news with nothing new to add. Normal cutoff applies.
                logger.info(
                    f"Dropping stale article past cutoff with non-contradicting "
                    f"fresher coverage available (age={c['_age_days']:.1f}d): "
                    f"{c['metadata'].get('title', '')[:80]}"
                )
                continue

        for c in kept:
            combined = (
                (c["_cosine"] * 0.6)
                + (c["_importance"] * c["_decay"] * 0.4)
                + c["_urgency_bonus"]
                + c["_breaking_bonus"]
            )
            c["combined"] = round(combined, 4)

        kept.sort(key=lambda x: x["combined"], reverse=True)

        # Return top n, dropping internal scoring fields
        return [{"chunk": c["chunk"], "metadata": c["metadata"], "score": c["score"]}
                for c in kept[:n_results]]

    def count(self) -> int:
        return self.collection.count()

    def _find_fresh_update(self, candidate: dict, now: datetime) -> dict | None:
        """
        For an old, high-importance candidate, look for a same-topic article
        published within FRESH_UPDATE_WINDOW_DAYS. Returns the best-matching
        fresh article's dict (chunk + metadata + score), or None if no
        topically-similar fresh article exists.
        """
        chunk_text = candidate["chunk"]
        query_embedding = self.model.encode(
            [chunk_text], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(10, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        best = None
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            age = _age_days(meta.get("published", ""), now)
            if age is None or age > FRESH_UPDATE_WINDOW_DAYS:
                continue
            sim = 1 - results["distances"][0][i]
            if sim < 0.55:
                # Not actually the same topic, just nearby in embedding space.
                continue
            if best is None or sim > best["score"]:
                best = {
                    "chunk": results["documents"][0][i],
                    "metadata": meta,
                    "score": sim,
                }
        return best

    def _check_contradiction(self, old_article: dict, fresh_article: dict) -> bool:
        """
        Ask Haiku whether fresh_article contradicts, reverses, or supersedes
        the factual claim in old_article. Returns True if it does.

        Fails safe: on any API/parsing error, returns True (treat as
        contradicted/superseded) so a stale high-importance story is never
        silently exempted from the cutoff just because the check broke.
        """
        if not ANTHROPIC_API_KEY:
            logger.warning("No ANTHROPIC_API_KEY set — cannot run contradiction "
                            "check, treating old article as superseded by default.")
            return True

        prompt = (
            "You are checking whether a news update changes an earlier factual claim "
            "about oil markets or geopolitics.\n\n"
            f"OLDER ARTICLE:\n{old_article['chunk'][:2000]}\n\n"
            f"NEWER ARTICLE:\n{fresh_article['chunk'][:2000]}\n\n"
            "Does the newer article contradict, reverse, or supersede the core factual "
            "claim of the older article (e.g. a closure that was reported open is now "
            "closed again, a deal reported as agreed has since fallen apart)?\n"
            "Answer with exactly one word, YES or NO, then a dash and a one-line reason."
        )

        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 60,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"].strip()
            logger.info(f"Contradiction check: {text[:100]}")
            return text.upper().startswith("YES")
        except Exception as e:
            logger.warning(f"Contradiction check failed ({e}) — treating as superseded.")
            return True