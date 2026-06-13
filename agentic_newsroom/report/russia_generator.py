"""
Russia oil market report generator.
Produces an Analyst Note focused on Russian crude, shadow fleet,
sanctions, and related supply/logistics themes.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config.settings import (
    ANTHROPIC_API_KEY,
    REPORT_DIR,
    RUSSIA_WATCHLIST,
)

logger = logging.getLogger(__name__)

RUSSIA_REPORT_PROMPT = """You are a senior oil market analyst specialising in Russian crude, sanctions, and shadow fleet logistics.

Write an Analyst Note for refinery SLT (senior leadership team). Audience: refinery owners and chief trading officers. They want crisp, judgment-driven analysis — not a news summary.

Today: {date}
Brent reference: {brent}
Urals discount to Brent (if available): {urals_discount}

Key events and signals from the last 12 hours:
{events}

Most relevant articles:
{articles}

Write the Analyst Note covering:
1. Urals crude — current discount, direction, and what is driving it
2. Shadow fleet / dark fleet — any incidents, detentions, insurance developments
3. Sanctions and price cap — any new enforcement, evasion, or policy developments
4. Russian export routes — Novorossiysk, Baltic ports, Arctic — any disruptions or changes
5. Ukraine Strikes on Russian Oil Infrastructure — PRIORITY SECTION
List every confirmed or reported strike on Russian refineries, oil depots, pipelines,
or port terminals in this cycle. For each: facility name, location, reported damage,
estimated capacity affected in bpd if available, and whether confirmed or unconfirmed.
If no strikes this cycle, state explicitly: No strikes reported this cycle.
This section must never be skipped — SLT reads it first.
6. What this means for a refinery buying crude — feedstock cost implications, alternative sourcing

Format:
- Analyst Note header with date and time
- One paragraph per theme above (skip if no material news)
- Close with a 2-3 sentence bottom line: what does the adviser need to act on or watch today?
- Tone: direct, no filler, no manufactured urgency
- Length: 400-600 words
- Do NOT include a Promoter Brief section
"""


def generate_russia_report(
    events: list[dict],
    articles: list[dict],
    brent: float | None = None,
    urals_discount: float | None = None,
) -> str:
    """Generate the Russia-focused Analyst Note via Claude Sonnet."""

    date_str = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    # Format events
    events_text = ""
    for e in events[:15]:
        desc = e.get("description", "")
        etype = e.get("type", "")
        urgency = e.get("urgency", "")
        if desc:
            events_text += f"- [{etype}/{urgency}] {desc}\n"
    if not events_text:
        events_text = "No high-importance Russia-specific events this cycle."

    # Format articles
    articles_text = ""
    for a in articles[:8]:
        title = a.get("title", "")
        source = a.get("source", "")
        summary = a.get("summary", "")[:200]
        if title:
            articles_text += f"- [{source}] {title}: {summary}\n"
    if not articles_text:
        articles_text = "No Russia-specific articles this cycle."

    prompt = RUSSIA_REPORT_PROMPT.format(
        date=date_str,
        brent=f"${brent:.2f}" if brent else "not available",
        urals_discount=f"${urals_discount:.2f}/bbl" if urals_discount else "adviser to confirm from Argus/Platts",
        events=events_text,
        articles=articles_text,
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
                "model": "claude-sonnet-4-6",
                "max_tokens": 1500,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        report_text = response.json()["content"][0]["text"].strip()

    except Exception as e:
        logger.error(f"Russia report generation failed: {e}")
        report_text = f"Russia report generation failed: {e}"

    # Save report
    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    report_path = report_dir / f"{timestamp}_russia_report.md"
    report_path.write_text(report_text)
    logger.info(f"Russia report saved: {report_path}")

    # Auto-convert to docx if pandoc is available
    try:
        import subprocess, shutil

        # cron runs with minimal PATH — search common install locations explicitly
        pandoc_cmd = shutil.which("pandoc") or next(
            (p for p in [
                "/usr/local/bin/pandoc",
                "/opt/homebrew/bin/pandoc",
                "/usr/bin/pandoc",
                "/home/linuxbrew/.linuxbrew/bin/pandoc",
            ] if Path(p).exists()),
            None
        )

        if not pandoc_cmd:
            logger.info("Pandoc not found in PATH or known locations — skipping DOCX conversion")
        else:
            docx_path = report_path.with_suffix(".docx")
            result = subprocess.run(
                [pandoc_cmd, str(report_path), "-o", str(docx_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                logger.info(f"DOCX saved: {docx_path}")
            else:
                logger.warning(f"Pandoc failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"DOCX conversion error: {e}")


    return report_text
