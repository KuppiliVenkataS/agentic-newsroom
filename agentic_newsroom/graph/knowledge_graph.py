"""
Knowledge graph module using KuzuDB.

Builds a graph from Ollama-extracted entities and events.

Node types:
    Article     — each ingested article
    Organisation— companies, OPEC, governments
    Person      — named individuals
    Location    — countries, regions, cities
    Commodity   — oil, brent, wti, natural gas etc
    Event       — price moves, supply changes, geopolitical events

Edge types:
    MENTIONS        — Article → Entity
    INVOLVED_IN     — Person/Organisation → Event
    LOCATED_IN      — Event → Location
    PRICE_SIGNAL    — Article → direction (bullish/bearish/neutral)

DB location: STORAGE_ROOT/graph/
"""

import logging
from pathlib import Path

import kuzu

from config.settings import GRAPH_DB_DIR

logger = logging.getLogger(__name__)


SCHEMA = [
    # Node tables
    """CREATE NODE TABLE IF NOT EXISTS Event (
        id                      STRING,
        type                    STRING,
        description             STRING,
        date_mentioned          STRING,
        urgency                 STRING,
        geopolitical_significance STRING,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Article (
        id                STRING,
        url               STRING,
        title             STRING,
        source            STRING,
        published         STRING,
        sentiment         STRING,
        direction         STRING,
        confidence        STRING,
        importance_score  DOUBLE,
        is_breaking       BOOLEAN,
        hormuz_risk       BOOLEAN,
        opec_event        BOOLEAN,
        sanctions_event   BOOLEAN,
        PRIMARY KEY (id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Organisation (
        name        STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Person (
        name        STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Location (
        name        STRING,
        PRIMARY KEY (name)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Commodity (
        name        STRING,
        PRIMARY KEY (name)
    )""",
    # Edge tables
    """CREATE REL TABLE IF NOT EXISTS MENTIONS (
        FROM Article TO Organisation
    )""",
    """CREATE REL TABLE IF NOT EXISTS MENTIONS_PERSON (
        FROM Article TO Person
    )""",
    """CREATE REL TABLE IF NOT EXISTS MENTIONS_LOCATION (
        FROM Article TO Location
    )""",
    """CREATE REL TABLE IF NOT EXISTS MENTIONS_COMMODITY (
        FROM Article TO Commodity
    )""",
    """CREATE REL TABLE IF NOT EXISTS HAS_EVENT (
        FROM Article TO Event
    )""",
]


class KnowledgeGraph:
    def __init__(self):
        # KuzuDB creates its own directory — do not pre-create it
        GRAPH_DB_DIR.parent.mkdir(parents=True, exist_ok=True)
        self.db   = kuzu.Database(str(GRAPH_DB_DIR))
        self.conn = kuzu.Connection(self.db)
        self._init_schema()
        logger.info(f"Knowledge graph ready at {GRAPH_DB_DIR}")

    def _init_schema(self):
        for statement in SCHEMA:
            try:
                self.conn.execute(statement)
            except Exception as e:
                # Table already exists — safe to ignore
                if "already exists" not in str(e).lower():
                    logger.warning(f"Schema error: {e}")

    def _safe_str(self, val) -> str:
        """Clean a value for use in Cypher queries."""
        if not val:
            return ""
        return str(val).replace("'", "''").strip()[:200]

    def _upsert_node(self, table: str, name_field: str, name: str):
        name = self._safe_str(name)
        if not name:
            return
        try:
            self.conn.execute(
                f"MERGE (n:{table} {{{name_field}: '{name}'}})"
            )
        except Exception as e:
            logger.debug(f"Upsert {table} failed: {e}")

    def _upsert_article(self, article: dict, extraction: dict):
        art_id     = self._safe_str(article.get("url", ""))[:64]
        url        = self._safe_str(article.get("url", ""))
        title      = self._safe_str(article.get("title", ""))
        source     = self._safe_str(article.get("source", ""))
        published  = self._safe_str(article.get("published", ""))
        sentiment  = self._safe_str(extraction.get("sentiment", ""))
        signals    = extraction.get("price_signals", {})
        direction  = self._safe_str(signals.get("direction", ""))
        confidence = self._safe_str(signals.get("confidence", ""))

        # New importance fields
        importance   = float(extraction.get("importance_score", 0.0))
        is_breaking  = "true" if extraction.get("is_breaking")  else "false"
        hormuz_risk  = "true" if extraction.get("hormuz_risk")   else "false"
        opec_event   = "true" if extraction.get("opec_event")    else "false"
        sanctions    = "true" if extraction.get("sanctions_event") else "false"

        try:
            self.conn.execute(f"""
                MERGE (a:Article {{id: '{art_id}'}})
                SET a.url             = '{url}',
                    a.title           = '{title}',
                    a.source          = '{source}',
                    a.published       = '{published}',
                    a.sentiment       = '{sentiment}',
                    a.direction       = '{direction}',
                    a.confidence      = '{confidence}',
                    a.importance_score = {importance},
                    a.is_breaking     = {is_breaking},
                    a.hormuz_risk     = {hormuz_risk},
                    a.opec_event      = {opec_event},
                    a.sanctions_event = {sanctions}
            """)
        except Exception as e:
            logger.debug(f"Upsert Article failed: {e}")

        return art_id

    def _link(self, rel: str, from_table: str, from_id: str,
              from_field: str, to_table: str, to_field: str, to_name: str):
        to_name = self._safe_str(to_name)
        if not to_name:
            return
        try:
            self.conn.execute(f"""
                MATCH (a:{from_table} {{{from_field}: '{from_id}'}}),
                      (b:{to_table}  {{{to_field}:  '{to_name}'}})
                MERGE (a)-[:{rel}]->(b)
            """)
        except Exception as e:
            logger.debug(f"Link {rel} failed: {e}")

    def add_articles(self, articles: list[dict]) -> int:
        """
        Load extracted entities from enriched articles into the graph.
        Returns count of articles processed.
        """
        count = 0

        for article in articles:
            if article.get("type") == "gdelt_gkg":
                continue   # GDELT has no Ollama extraction

            extractions = article.get("extraction", [])
            if not extractions:
                continue

            # Use first successful extraction
            extraction = next(
                (e for e in extractions if e.get("status") == "ok"),
                None
            )
            if not extraction:
                continue

            art_id = self._upsert_article(article, extraction)
            entities = extraction.get("entities", {})

            # Organisations
            for org in entities.get("organisations", []):
                self._upsert_node("Organisation", "name", org)
                self._link("MENTIONS", "Article", art_id, "id",
                           "Organisation", "name", org)

            # People
            for person in entities.get("people", []):
                self._upsert_node("Person", "name", person)
                self._link("MENTIONS_PERSON", "Article", art_id, "id",
                           "Person", "name", person)

            # Locations
            for loc in entities.get("locations", []):
                self._upsert_node("Location", "name", loc)
                self._link("MENTIONS_LOCATION", "Article", art_id, "id",
                           "Location", "name", loc)

            # Commodities
            for comm in entities.get("commodities", []):
                self._upsert_node("Commodity", "name", comm)
                self._link("MENTIONS_COMMODITY", "Article", art_id, "id",
                           "Commodity", "name", comm)

            # Events
            for i, event in enumerate(extraction.get("events", [])):
                desc = event.get("description", "")
                if not desc:
                    continue
                event_id = f"{art_id}::event::{i}"
                event_id = self._safe_str(event_id)
                try:
                    self.conn.execute(f"""
                        MERGE (e:Event {{id: '{event_id}'}})
                        SET e.type                     = '{self._safe_str(event.get("type", ""))}',
                            e.description              = '{self._safe_str(desc)}',
                            e.date_mentioned           = '{self._safe_str(event.get("date_mentioned", ""))}',
                            e.urgency                  = '{self._safe_str(event.get("urgency", ""))}',
                            e.geopolitical_significance = '{self._safe_str(event.get("geopolitical_significance", ""))}'
                    """)
                    self._link("HAS_EVENT", "Article", art_id, "id",
                               "Event", "id", event_id)
                except Exception as e:
                    logger.debug(f"Event upsert failed: {e}")

            count += 1

        logger.info(f"Knowledge graph updated: {count} articles processed")
        return count

    def query_top_organisations(self, limit: int = 10) -> list[dict]:
        """Most mentioned organisations across all articles."""
        result = self.conn.execute(f"""
            MATCH (a:Article)-[:MENTIONS]->(o:Organisation)
            RETURN o.name AS organisation, COUNT(a) AS mentions
            ORDER BY mentions DESC
            LIMIT {limit}
        """)
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"organisation": row[0], "mentions": row[1]})
        return rows

    def query_bearish_organisations(self, limit: int = 10) -> list[dict]:
        """Organisations most mentioned in bearish articles."""
        result = self.conn.execute(f"""
            MATCH (a:Article)-[:MENTIONS]->(o:Organisation)
            WHERE a.direction = 'bearish'
            RETURN o.name AS organisation, COUNT(a) AS mentions
            ORDER BY mentions DESC
            LIMIT {limit}
        """)
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({"organisation": row[0], "mentions": row[1]})
        return rows

    def query_recent_events(self, limit: int = 10) -> list[dict]:
        """Most recent events extracted from articles."""
        result = self.conn.execute(f"""
            MATCH (a:Article)-[:HAS_EVENT]->(e:Event)
            RETURN a.title AS article, e.type AS type,
                   e.description AS description, a.published AS published
            ORDER BY a.published DESC
            LIMIT {limit}
        """)
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({
                "article":     row[0],
                "type":        row[1],
                "description": row[2],
                "published":   row[3],
            })
        return rows

    def query_signal_summary(self) -> dict:
        """Count of bullish/bearish/neutral articles."""
        result = self.conn.execute("""
            MATCH (a:Article)
            RETURN a.direction AS direction, COUNT(a) AS count
            ORDER BY count DESC
        """)
        summary = {}
        while result.has_next():
            row = result.get_next()
            if row[0]:
                summary[row[0]] = row[1]
        return summary

    def query_high_urgency_events(self, limit: int = 10) -> list[dict]:
        """Events flagged critical or high urgency, most recent first."""
        result = self.conn.execute(f"""
            MATCH (a:Article)-[:HAS_EVENT]->(e:Event)
            WHERE e.urgency IN ['critical', 'high']
            RETURN a.title AS article, a.source AS source,
                   e.type AS type, e.description AS description,
                   e.urgency AS urgency, e.geopolitical_significance AS geo_sig,
                   a.published AS published, a.importance_score AS importance
            ORDER BY a.importance_score DESC, a.published DESC
            LIMIT {limit}
        """)
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({
                "article":     row[0],
                "source":      row[1],
                "type":        row[2],
                "description": row[3],
                "urgency":     row[4],
                "geo_sig":     row[5],
                "published":   row[6],
                "importance":  row[7],
            })
        return rows

    def query_risk_flags(self) -> dict:
        """Count articles with hormuz_risk, opec_event, sanctions_event flags."""
        result = self.conn.execute("""
            MATCH (a:Article)
            RETURN
                SUM(CASE WHEN a.hormuz_risk    = true THEN 1 ELSE 0 END) AS hormuz,
                SUM(CASE WHEN a.opec_event     = true THEN 1 ELSE 0 END) AS opec,
                SUM(CASE WHEN a.sanctions_event = true THEN 1 ELSE 0 END) AS sanctions,
                SUM(CASE WHEN a.is_breaking    = true THEN 1 ELSE 0 END) AS breaking
        """)
        if result.has_next():
            row = result.get_next()
            return {"hormuz": row[0], "opec": row[1], "sanctions": row[2], "breaking": row[3]}
        return {"hormuz": 0, "opec": 0, "sanctions": 0, "breaking": 0}

    def close(self):
        pass   # KuzuDB closes automatically