"""
Entity and event extractor using Ollama (local, free).

Runs llama3.1:8b locally via Ollama's REST API.
No API key needed. No cost per call.

Ollama must be running: `ollama serve`
Model must be pulled:   `ollama pull llama3.1:8b`

For each article chunk, extracts:
- Entities: organisations, people, locations, commodities
- Events: price movements, supply/demand signals, geopolitical events
- Sentiment: bullish / bearish / neutral on oil price
- Key figures: any numerical values mentioned (prices, volumes, dates)
"""

import json
import logging
import time
from datetime import datetime

import httpx

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are an expert financial analyst specialising in oil markets.

Extract structured information from the following news text. Return ONLY valid JSON, no explanation, no markdown backticks.

Text:
{text}

Return this exact JSON structure:
{{
  "entities": {{
    "organisations": [],
    "people": [],
    "locations": [],
    "commodities": []
  }},
  "events": [
    {{
      "type": "",
      "description": "",
      "date_mentioned": ""
    }}
  ],
  "price_signals": {{
    "direction": "",
    "confidence": "",
    "reasoning": ""
  }},
  "key_figures": [
    {{
      "value": "",
      "unit": "",
      "context": ""
    }}
  ],
  "sentiment": ""
}}

Rules:
- direction must be one of: "bullish", "bearish", "neutral", "unclear"
- confidence must be one of: "high", "medium", "low"
- sentiment must be one of: "positive", "negative", "neutral"
- event type must be one of: "price_move", "supply_change", "demand_change", "geopolitical", "policy", "weather", "other"
- If a field has no data, use empty list [] or empty string ""
- Return only the JSON object, nothing else, no markdown
"""


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temp for consistent structured output
            "num_predict": 800,
        }
    }
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120   # local inference can be slow on first call
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _parse_json(raw: str) -> dict:
    """Strip any accidental markdown fences and parse JSON."""
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue
    return json.loads(raw)


def extract_entities(chunks: list[str]) -> list[dict]:
    """
    Run extraction on each chunk.
    Returns one result dict per chunk.
    Failed chunks get an error entry rather than crashing the run.
    """
    results = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        try:
            raw    = _call_ollama(EXTRACTION_PROMPT.format(text=chunk))
            parsed = _parse_json(raw)
            parsed["chunk_index"] = i
            parsed["status"]      = "ok"
            results.append(parsed)

        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse failed on chunk {i}: {e}")
            results.append({"chunk_index": i, "status": "parse_error", "error": str(e)})

        except httpx.ConnectError:
            logger.error("  Ollama not reachable. Is `ollama serve` running?")
            results.append({"chunk_index": i, "status": "error", "error": "ollama_not_running"})
            break   # no point continuing if Ollama is down

        except Exception as e:
            logger.warning(f"  Extraction failed on chunk {i}: {e}")
            results.append({"chunk_index": i, "status": "error", "error": str(e)})

        # Small pause between calls — not strictly needed for local but
        # keeps the Mac Mini from thermal throttling on large batches
        time.sleep(0.2)

    return results


def process_articles(articles: list[dict]) -> list[dict]:
    """
    Run entity extraction on all articles that have chunks.
    Returns enriched article dicts with extraction results attached.
    """
    enriched = []
    total    = len(articles)

    for idx, article in enumerate(articles):
        chunks = article.get("chunks", [])

        if not chunks:
            article["extraction"] = []
            enriched.append(article)
            continue

        logger.info(f"  Extracting [{idx+1}/{total}]: {article.get('title', article.get('url', ''))[:60]}")

        extraction = extract_entities(chunks)

        article["extraction"]   = extraction
        article["extracted_at"] = datetime.utcnow().isoformat()
        enriched.append(article)

    return enriched