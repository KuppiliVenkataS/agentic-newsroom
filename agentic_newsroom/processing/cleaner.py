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
    Takes a raw article dict from the archive and returns it
    with cleaned text and chunks added.
    """
    # Combine title and summary as the text body for RSS articles
    if article.get("type") == "gdelt_gkg":
        return {
            **article,
            "cleaned_text": "",
            "chunks":       [],
            "chunk_count":  0,
        }
    else:
        raw_text = " ".join(filter(None, [
            article.get("title", ""),
            article.get("summary", ""),
        ]))

    cleaned = clean_text(raw_text)
    chunks  = chunk_text(cleaned)

    return {
        **article,
        "cleaned_text": cleaned,
        "chunks":       chunks,
        "chunk_count":  len(chunks),
    }