"""
Text cleaning and chunking.

Cleans raw article text and splits it into chunks small enough
to embed and send to the LLM for entity extraction.

Chunk size is deliberately conservative (400 tokens ~ 300 words)
to stay well within Claude's context and keep extraction focused.
"""

import re


MAX_CHUNK_CHARS = 1500   # ~400 tokens, safe for extraction prompts
OVERLAP_CHARS   = 150    # overlap so entities at chunk boundaries aren't missed


def clean_text(text: str) -> str:
    """Remove HTML tags, excess whitespace, and non-printable characters."""
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks.
    Tries to split on sentence boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_CHARS

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at sentence boundary within last 200 chars
        boundary = text.rfind(". ", start + MAX_CHUNK_CHARS - 200, end)
        if boundary != -1:
            end = boundary + 1   # include the period

        chunks.append(text[start:end].strip())
        start = end - OVERLAP_CHARS   # step back for overlap

    return [c for c in chunks if c]


def prepare_article(article: dict) -> dict:
    """
    Takes a raw article dict and returns it with cleaned text and chunks.
    Uses full body text if available (from article_fetcher),
    falls back to title + summary for paywalled or failed fetches.
    """
    if article.get("type") == "gdelt_gkg":
        return {
            **article,
            "cleaned_text": "",
            "chunks":       [],
            "chunk_count":  0,
            "text_source":  "none",
        }

    body    = article.get("body", "")
    title   = article.get("title", "")
    summary = article.get("summary", "")

    if len(body) >= 200:
        # Full body available — prepend title so it's always in first chunk
        raw_text    = title + ". " + body
        text_source = "full_body"
    else:
        # Fallback to title + summary
        raw_text    = " ".join(filter(None, [title, summary]))
        text_source = "summary_only"

    cleaned = clean_text(raw_text)
    chunks  = chunk_text(cleaned)

    return {
        **article,
        "cleaned_text": cleaned,
        "chunks":       chunks,
        "chunk_count":  len(chunks),
        "text_source":  text_source,
    }