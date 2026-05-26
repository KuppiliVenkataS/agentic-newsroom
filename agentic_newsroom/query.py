"""
RAG query tool — command line interface.

Ask questions about oil market news stored in the vector DB.
The system retrieves relevant chunks and generates an answer via Ollama.

Usage:
    python query.py "what did OPEC say about production cuts this week?"
    python query.py "any Iran Hormuz news today?"
    python query.py "what is the latest WTI price trend?"
    python query.py  (no argument — interactive mode)
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

import httpx
from vectordb.store import VectorStore
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

QUERY_PROMPT = """You are an expert oil market analyst. Answer the question below using ONLY the news excerpts provided.
Be concise and direct. If the excerpts do not contain enough information to answer, say so clearly.
Do not invent facts.

Question: {question}

Relevant news excerpts:
{context}

Answer:"""


def _call_ollama(prompt: str) -> str:
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 500,
        }
    }
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _format_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        source    = r["metadata"].get("source", "unknown")
        title     = r["metadata"].get("title", "")
        published = r["metadata"].get("published", "")
        chunk     = r["chunk"][:300]
        lines.append(f"[{i}] Source: {source} | {published}\n    {title}\n    {chunk}")
    return "\n\n".join(lines)


def query(question: str, n_results: int = 8) -> dict:
    """
    Run a RAG query against the vector DB.
    Returns answer and source chunks.
    """
    store   = VectorStore()
    results = store.search(question, n_results=n_results)

    if not results:
        return {
            "question": question,
            "answer":   "No relevant articles found in the database.",
            "sources":  [],
        }

    context = _format_context(results)
    prompt  = QUERY_PROMPT.format(question=question, context=context)
    answer  = _call_ollama(prompt)

    sources = [
        {
            "title":     r["metadata"].get("title", ""),
            "source":    r["metadata"].get("source", ""),
            "published": r["metadata"].get("published", ""),
            "score":     round(r["score"], 3),
            "url":       r["metadata"].get("url", ""),
        }
        for r in results
    ]

    return {
        "question":   question,
        "answer":     answer,
        "sources":    sources,
        "queried_at": datetime.now(timezone.utc).isoformat(),
    }


def interactive_mode():
    store = VectorStore()
    print(f"\nOil Market RAG Query — {store.count()} chunks in database")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("\nSearching...")
        result = query(question)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])} articles):")
        for s in result["sources"]:
            print(f"  [{s['score']}] {s['source']} — {s['title'][:70]} ({s['published'][:10]})")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        print(f"\nSearching: {question}\n")
        result = query(question)
        print(f"Answer:\n{result['answer']}\n")
        print(f"Sources:")
        for s in result["sources"]:
            print(f"  [{s['score']}] {s['source']} — {s['title'][:70]} ({s['published'][:10]})")
    else:
        # Interactive mode
        interactive_mode()