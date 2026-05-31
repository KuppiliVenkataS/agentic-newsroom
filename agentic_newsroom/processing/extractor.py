"""
Entity and event extractor using Claude API (primary) or Ollama (fallback).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import httpx

from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, ANTHROPIC_API_KEY, USE_CLAUDE_EXTRACTION

logger = logging.getLogger(__name__)

MAX_WORKERS = 2   # 2 workers × 0.5s chunk delay keeps well within rate limits

EXTRACTION_PROMPT_FULL = """You are an expert financial analyst specialising in oil markets.

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
      "date_mentioned": "",
      "urgency": "",
      "geopolitical_significance": ""
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
  "sentiment": "",
  "importance_score": 0.0,
  "importance_reason": "",
  "is_breaking": false,
  "hormuz_risk": false,
  "opec_event": false,
  "sanctions_event": false,
  "supply_disruption": false,
  "logistics_disruption": false
}}

Rules:
- direction: "bullish", "bearish", "neutral", or "unclear"
- confidence: "high", "medium", or "low"
- sentiment: "positive", "negative", or "neutral"
- event type: "price_move", "supply_change", "demand_change", "geopolitical", "sanctions", "military_strike", "opec_decision", "policy", "weather", or "other"
- urgency: "critical" (active strikes/Hormuz closure), "high" (sanctions/ceasefire collapse), "medium" (OPEC meeting/diplomacy), "low" (analyst notes/routine data)
- geopolitical_significance: "extreme", "high", "medium", "low", or "none"
- importance_score 0.0-1.0: how directly does this affect physical oil supply or demand in 48 hours?
  0.9+ = active attack on oil infrastructure or Hormuz closure
  0.7-0.9 = major sanctions, ceasefire collapse, confirmed supply disruption
  0.5-0.7 = OPEC decision, significant diplomatic development
  0.3-0.5 = geopolitical risk that could escalate
  0.1-0.3 = analyst notes, routine data, indirect policy
- importance_reason: one sentence naming the supply/demand mechanism
- is_breaking: true if event happened today or in last 24 hours
- hormuz_risk: true if Hormuz closure or Iranian naval threat mentioned
- opec_event: true if OPEC/OPEC+ production decision or meeting
- sanctions_event: true if new oil-related sanctions described
- supply_disruption: true for ANY physical interruption to oil/gas production or
  refining, from ANY cause — storm/hurricane shut-in, freeze-off, earthquake or
  tsunami damage, refinery fire/outage, field accident, pipeline rupture, labour
  strike. This is theme-agnostic: it is NOT limited to the Middle East.
- logistics_disruption: true for ANY interruption to the TRANSPORT of oil/products —
  low river water halting barges (e.g. Rhine, Mississippi), canal/strait blockage,
  port closure, tanker shortage, freight/shipping breakdown. Also theme-agnostic.
- Empty fields: use [] or ""
- Return only the JSON object, no markdown
"""

# Concise prompt for Ollama — same fields, minimal tokens
EXTRACTION_PROMPT_FAST = """You are an oil markets analyst. Extract structured data from this news text. Return ONLY valid JSON.

Text:
{text}

JSON structure:
{{
  "entities": {{"organisations": [], "people": [], "locations": [], "commodities": []}},
  "events": [{{"type": "", "description": "", "date_mentioned": "", "urgency": "", "geopolitical_significance": ""}}],
  "price_signals": {{"direction": "", "confidence": "", "reasoning": ""}},
  "key_figures": [{{"value": "", "unit": "", "context": ""}}],
  "sentiment": "",
  "importance_score": 0.0,
  "importance_reason": "",
  "is_breaking": false,
  "hormuz_risk": false,
  "opec_event": false,
  "sanctions_event": false,
  "supply_disruption": false,
  "logistics_disruption": false
}}

direction: bullish/bearish/neutral/unclear. confidence: high/medium/low. sentiment: positive/negative/neutral.
urgency: critical(strikes/Hormuz)/high(sanctions/ceasefire)/medium(OPEC/diplomacy)/low(analyst/routine).
importance_score: 0.9+=active oil infrastructure attack, 0.7+=major sanctions/disruption, 0.5+=OPEC decision, 0.3+=escalation risk, 0.1+=analyst note.
importance_reason: one sentence on supply/demand mechanism.
is_breaking: true if last 24h. hormuz_risk/opec_event/sanctions_event: true if applicable.
supply_disruption: true for ANY physical production/refining interruption from ANY cause (storm, freeze, quake/tsunami, refinery outage, field accident, pipeline rupture) — not just Middle East.
logistics_disruption: true for ANY oil/product TRANSPORT interruption (low river water/Rhine barges, canal/strait/port closure, tanker shortage) — not just Middle East.
Return only JSON, no markdown.
"""


def _call_llm(prompt: str, retries: int = 3) -> str:
    """
    Use Claude API if key available AND USE_CLAUDE_EXTRACTION=true.
    Falls back to Ollama otherwise.
    """
    if ANTHROPIC_API_KEY and USE_CLAUDE_EXTRACTION:
        for attempt in range(retries):
            try:
                response = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5",
                        "max_tokens": 1000,
                        "temperature": 0.1,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=30
                )
                if response.status_code == 429:
                    wait = 2 ** attempt   # 1s, 2s, 4s
                    logger.warning(f"  Rate limited — waiting {wait}s before retry {attempt+1}/{retries}")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()["content"][0]["text"].strip()
            except httpx.HTTPStatusError as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Claude API failed after retries")
    else:
        return _call_ollama(prompt)


def _call_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 800,
        }
    }
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
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
    Uses concise prompt for Ollama, full prompt for Claude API.
    Caps at 2 chunks per article when using Ollama to limit inference time.
    """
    using_claude = bool(ANTHROPIC_API_KEY and USE_CLAUDE_EXTRACTION)
    prompt_template = EXTRACTION_PROMPT_FULL if using_claude else EXTRACTION_PROMPT_FAST

    # Ollama: cap at first 2 chunks — covers key facts, avoids 3+ hour runtimes
    if not using_claude:
        chunks = chunks[:2]

    results = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        try:
            raw    = _call_llm(prompt_template.format(text=chunk))
            parsed = _parse_json(raw)
            parsed["chunk_index"] = i
            parsed["status"]      = "ok"
            results.append(parsed)

        except json.JSONDecodeError as e:
            logger.warning(f"  JSON parse failed on chunk {i}: {e}")
            results.append({"chunk_index": i, "status": "parse_error", "error": str(e)})

        except httpx.ConnectError:
            if not using_claude:
                logger.error("  Ollama not reachable. Is `ollama serve` running?")
                results.append({"chunk_index": i, "status": "error", "error": "ollama_not_running"})
                break
            results.append({"chunk_index": i, "status": "error", "error": "connection_error"})

        except Exception as e:
            logger.warning(f"  Extraction failed on chunk {i}: {e}")
            results.append({"chunk_index": i, "status": "error", "error": str(e)})

        if not using_claude:
            time.sleep(0.2)
        else:
            time.sleep(0.5)

    return results


def _process_single_article(args: tuple) -> dict:
    """Process one article — designed for use in ThreadPoolExecutor."""
    idx, total, article = args
    chunks = article.get("chunks", [])

    if not chunks:
        return {**article, "extraction": [], "extracted_at": datetime.utcnow().isoformat()}

    title = article.get("title", article.get("url", ""))[:60]
    logger.info(f"  Extracting [{idx+1}/{total}]: {title}")

    extraction = extract_entities(chunks)
    return {
        **article,
        "extraction":   extraction,
        "extracted_at": datetime.utcnow().isoformat(),
    }


def process_articles(articles: list[dict]) -> list[dict]:
    """
    Run entity extraction on all articles that have chunks.
    Parallel with Claude API, sequential with Ollama.
    """
    total = len(articles)

    if ANTHROPIC_API_KEY and USE_CLAUDE_EXTRACTION:
        logger.info(f"Extracting {total} articles via Claude API ({MAX_WORKERS} parallel workers)...")
        enriched = [None] * total
        args = [(i, total, article) for i, article in enumerate(articles)]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_single_article, arg): arg[0] for arg in args}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    enriched[idx] = future.result()
                except Exception as e:
                    logger.warning(f"  Article {idx} failed: {e}")
                    enriched[idx] = {
                        **articles[idx],
                        "extraction":   [],
                        "extracted_at": datetime.utcnow().isoformat(),
                    }
        return enriched
    else:
        logger.info(f"Extracting {total} articles via Ollama (sequential)...")
        return [_process_single_article((i, total, a)) for i, a in enumerate(articles)]