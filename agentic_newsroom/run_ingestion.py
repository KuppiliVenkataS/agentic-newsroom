"""
Main ingestion runner.

Run manually:
    python run_ingestion.py

Or via cron (every 12 hours):
    0 6,18 * * * /path/to/venv/bin/python /path/to/agentic_newsroom/run_ingestion.py >> /Volumes/Mac_extension/projects/OilNewsDB/agentic_newsroom/logs/cron.log 2>&1

Safe to run multiple times — dedup prevents repeated articles.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import STORAGE_ROOT, DEDUP_DB, LOG_DIR, RAW_DIR
from ingestion.dedup import DedupRegistry
from ingestion.rss_fetcher import fetch_all_feeds
from ingestion.eia_fetcher import fetch_eia_data
from ingestion.yfinance_fetcher import fetch_live_prices
from ingestion.gdelt_fetcher import fetch_gdelt_data
from ingestion.archive_writer import save_run
from ingestion.audit_logger import write_audit
from ingestion.article_fetcher import enrich_articles_with_body
from processing.cleaner import prepare_article
from processing.extractor import process_articles
from processing.processed_writer import save_processed
from config.settings import SKIP_EXTRACTION, SKIP_INGESTION, MAX_TOTAL_RSS
from vectordb.store import VectorStore
from graph.knowledge_graph import KnowledgeGraph
from prediction.predictor import generate_prediction
from prediction.scorer import score_last_prediction, read_accuracy_summary
from report.generator import generate_report

# ── Logging ───────────────────────────────────────────────────────────────────
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
    now        = datetime.now(timezone.utc)
    started_at = now.isoformat()
    run_id     = now.strftime("%Y-%m-%d_%H-%M-%S")
    errors: list[str] = []

    logger.info(f"=== Ingestion run started: {run_id} ===")
    logger.info(f"Storage root: {STORAGE_ROOT}")

    # ── 1. Dedup registry ──────────────────────────────────────────────────
    dedup = DedupRegistry(DEDUP_DB)

    # ── 2-5. Ingestion (RSS, EIA, GDELT, archive) ────────────────────────
    articles: list[dict]    = []
    market_data: list[dict] = []
    archive_path            = ""

    if SKIP_INGESTION:
        logger.info("Ingestion skipped (SKIP_INGESTION=true) — loading last raw archive")
        import json, glob
        raw_files = sorted(glob.glob(str(RAW_DIR / "**" / "*_run.json"), recursive=True))
        if raw_files:
            last = raw_files[-1]
            logger.info(f"  Loading: {last}")
            data        = json.load(open(last))
            articles    = data.get("articles", [])
            market_data = data.get("market_data", [])
            logger.info(f"  Loaded {len(articles)} articles, {len(market_data)} data points")
        else:
            logger.warning("  No raw archive found — nothing to process")
    else:
        try:
            articles = fetch_all_feeds(dedup)
            if len(articles) > MAX_TOTAL_RSS:
                logger.info(f"RSS capped: {len(articles)} → {MAX_TOTAL_RSS} articles")
                articles = articles[:MAX_TOTAL_RSS]
            logger.info(f"RSS total new articles: {len(articles)}")
        except Exception as exc:
            msg = f"RSS fetch failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        try:
            market_data = fetch_eia_data()
            logger.info(f"EIA data points: {len(market_data)}")
        except Exception as exc:
            msg = f"EIA fetch failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        # ── 3b. Live prices from OilPriceAPI ──────────────────────────────
        try:
            live_prices = fetch_live_prices()
            market_data.extend(live_prices)
            logger.info(f"Live prices fetched: {len(live_prices)}")
        except Exception as exc:
            msg = f"Live price fetch failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        try:
            gdelt_records = fetch_gdelt_data()
            logger.info(f"GDELT records: {len(gdelt_records)}")
            articles.extend(gdelt_records)
        except Exception as exc:
            msg = f"GDELT fetch failed: {exc}"
            logger.error(msg)
            errors.append(msg)

        try:
            saved        = save_run(articles, market_data, run_id)
            archive_path = str(saved)
        except Exception as exc:
            msg = f"Archive write failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ── 5b. Full article body fetch ────────────────────────────────────────
    if not SKIP_INGESTION:
        try:
            rss_articles = [a for a in articles if a.get("type") == "rss_article"]
            logger.info(f"Fetching full article bodies for {len(rss_articles)} RSS articles...")
            articles_with_body = enrich_articles_with_body(rss_articles, delay=0.5)
            # Merge back — replace RSS articles with body-enriched versions
            non_rss = [a for a in articles if a.get("type") != "rss_article"]
            articles = articles_with_body + non_rss
            fetched_full = sum(1 for a in articles_with_body if len(a.get("body","")) >= 200)
            logger.info(f"Full body fetch: {fetched_full}/{len(rss_articles)} articles got full text")
        except Exception as exc:
            msg = f"Article body fetch failed: {exc}"
            logger.error(msg)
            errors.append(msg)

    # ── 6. Clean and chunk articles ────────────────────────────────────────
    prepared: list[dict] = []
    try:
        # Strip any cached chunks from archive before re-cleaning
        for a in articles:
            a.pop("chunks", None)
            a.pop("cleaned_text", None)
            a.pop("chunk_count", None)
        prepared = [prepare_article(a) for a in articles]
        logger.info(f"Prepared {len(prepared)} articles for extraction")
    except Exception as exc:
        msg = f"Cleaning/chunking failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 7. LLM entity extraction ───────────────────────────────────────────
    enriched: list[dict] = []
    if SKIP_EXTRACTION:
        logger.info("Extraction skipped (SKIP_EXTRACTION=true)")
        enriched = prepared
    else:
        try:
            # Only extract RSS articles — GDELT has no body and doesn't need it
            # Prioritise: watchlist matches first, then all remaining RSS
            from config.settings import USER_WATCHLIST
            def _is_watchlist(a: dict) -> bool:
                text = (a.get("title","") + " " + a.get("summary","")).lower()
                return any(kw.lower() in text for kw in USER_WATCHLIST)

            rss_only   = [a for a in prepared if a.get("type") == "rss_article"]
            non_rss    = [a for a in prepared if a.get("type") != "rss_article"]

            priority   = [a for a in rss_only if _is_watchlist(a)]
            remainder  = [a for a in rss_only if not _is_watchlist(a)]

            # Cap: extract all watchlist matches + up to 50 others
            to_extract = priority + remainder[:50]
            logger.info(
                f"Extraction scope: {len(to_extract)} articles "
                f"({len(priority)} watchlist, {len(remainder[:50])} other RSS) "
                f"— skipping {len(non_rss)} GDELT + {len(remainder[50:])} low-priority RSS"
            )

            enriched_extracted = process_articles(to_extract)
            enriched = enriched_extracted + non_rss + remainder[50:]
            logger.info(f"Extraction complete: {len(enriched_extracted)} articles enriched")
        except Exception as exc:
            msg = f"Extraction failed: {exc}"
            logger.error(msg)
            errors.append(msg)
            enriched = prepared

    # ── 8. Save processed output ───────────────────────────────────────────
    try:
        save_processed(enriched, run_id)
    except Exception as exc:
        msg = f"Processed write failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 9. Embed into vector DB ───────────────────────────────────────────
    try:
        store       = VectorStore()
        added       = store.add_articles(enriched)
        logger.info(f"Vector DB: {added} chunks added, {store.count()} total")
    except Exception as exc:
        msg = f"Vector DB failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 11. Score previous prediction ────────────────────────────────────
    try:
        score_record = score_last_prediction(market_data, run_id)
        if score_record:
            accuracy = read_accuracy_summary()
            logger.info(
                f"Accuracy to date: {accuracy['correct']}/{accuracy['correct']+accuracy['wrong']} "
                f"scored ({accuracy['accuracy'] or 'n/a'})"
            )
    except Exception as exc:
        msg = f"Scorer failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 12. Generate prediction ──────────────────────────────────────────
    prediction = {}
    try:
        kg         = KnowledgeGraph()
        added      = kg.add_articles(enriched)
        logger.info(f"Knowledge graph: {added} articles added")
        prediction = generate_prediction(kg, market_data=market_data, articles=articles)
        logger.info(f"Prediction: {prediction['direction']} ({prediction['confidence']}) score={prediction['score']}")
    except Exception as exc:
        msg = f"Prediction failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 12b. Save prediction into archive ─────────────────────────────────
    if prediction and archive_path:
        try:
            import json as _json
            with open(archive_path, "r", encoding="utf-8") as f:
                archive = _json.load(f)
            archive["prediction"] = prediction
            with open(archive_path, "w", encoding="utf-8") as f:
                _json.dump(archive, f, ensure_ascii=False, indent=2)
            logger.info("Prediction saved into archive for future scoring")
        except Exception as exc:
            logger.warning(f"Could not save prediction to archive: {exc}")

    # ── 13. Alert check — high urgency triggers immediate report ──────────
    geo_level  = prediction.get("signals", {}).get("geopolitical", {}).get("level", "minimal")
    is_alert   = geo_level in ("critical", "high") and prediction.get("confidence") != "low"
    if is_alert:
        logger.warning(
            f"⚠ ALERT: Geopolitical risk level={geo_level} — "
            f"high-urgency report triggered outside normal schedule"
        )

    # ── 14. Generate report ───────────────────────────────────────────────
    try:
        if prediction:
            report_path = generate_report(
                prediction, kg,
                enriched_articles=enriched,
                is_alert=is_alert,
            )
            logger.info(f"Report saved: {report_path}")
        else:
            logger.warning("Skipping report — no prediction available")
    except Exception as exc:
        msg = f"Report generation failed: {exc}"
        logger.error(msg)
        errors.append(msg)

    # ── 13. Write audit log ────────────────────────────────────────────────
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

    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    run()