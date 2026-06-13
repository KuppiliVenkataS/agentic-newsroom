"""
Weekly oil market summary generator (standalone).

Separate from the daily generator.py on purpose: the weekly has a different
structure (retrospective, ~370 words, six fixed sections) and a different input
window (last 7 days of processed cycles, not one cycle).

Reads the past 7 days of processed-cycle JSON files from PROCESSED_DIR, dedupes
articles across cycles, applies the SAME cycle-health guard (send nothing if the
week's data is empty/stale), and writes a markdown report to REPORT_DIR.

Audience: SLT, via the adviser. Like the daily, it leaves assessed-price slots
for the adviser and is DRAFT until the adviser reviews. Email/delivery is handled
separately (deferred) — this module only GENERATES and SAVES.

Invoked by run_weekly.py (Friday evening cron).
"""

import glob
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.settings import PROCESSED_DIR, REPORT_DIR, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

WEEKLY_WINDOW_DAYS = 7

# Reuse the daily generator's LLM caller and domain knowledge so the weekly
# speaks the same house voice and applies the same oil-market corrections.
try:
    from report.generator import _call_llm, OIL_MARKET_KNOWLEDGE
except Exception:  # pragma: no cover - import path fallback
    from generator import _call_llm, OIL_MARKET_KNOWLEDGE


WEEKLY_PROMPT = """You are the chief trading adviser writing the WEEKLY oil market
summary for the senior leadership team (SLT). This is a retrospective of the
week's events and what they mean for the refinery's crude cost and risk — NOT a
day-by-day log. It is read by busy executives via their adviser.

{domain_knowledge}

This week's material (deduplicated across the last 7 days of cycles):
{week_digest}

FIRST, decide the week's SINGLE organising read — the one thread that ties the
week together (e.g. "a peace trade that reversed", "premium draining despite
loud headlines", "supply story quietly overtaking the conflict story"). Every
section below must serve that one read, so the whole summary speaks in one
coherent voice rather than listing disconnected events. Where the week had
several drivers, state how they RELATE — did they reinforce, offset, or did one
overtake another across the week — and which dominated by Friday.

WRITE THE WEEKLY IN EXACTLY THIS STRUCTURE (~370 words total, prose, no bullets):

Headline: one line that IS the week's conclusion for a refiner.

**Crude & cost.** What crude did this week and what it means for our feedstock
cost. Direction and drivers, not a price target.

**The other side of the trade.** The counter-risk the daily headlines underplay
(e.g. waiting supply, structural oversupply, an unwinding premium). The asymmetry.

**Physical risks.** Any live supply/logistics disruptions (Hormuz, storms, Rhine,
refinery/field outages, freight). What's real vs. noise this week.

**Demand & flows.** The demand-side and trade-flow signals (China, refining
margins, import/export shifts) — even if not acute.

**What it means for us.** The refiner's-eye takeaway: how the week shifts our
crude-cost outlook and posture. Inform, do not instruct a specific hedge.

**Watch next week.** 3-4 conditional signposts to monitor.

RULES:
- ~370 words. Tight. Executive audience — no padding, no jargon dumps.
- Sharp, opinionated adviser voice on direction and meaning; never a hard price
  call or trade instruction (directional hinting only — the SLT decides).
- Every section serves the one organising read; events are RELATED to each
  other, not listed in isolation. The coherence is the point.
- Frame actor-driven moves as MARKET BEHAVIOUR (sentiment that may retrace vs.
  substance that lasts). NEVER characterise a named real person's competence,
  motives, or depth — read the market's reaction, not the person.
- Do NOT invent assessed prices. The adviser adds Argus/Platts numbers at review.
- This is a DRAFT until the adviser reviews it.
"""


def _load_week_cycles(now: datetime = None) -> list[dict]:
    """Load processed-cycle JSONs from the last WEEKLY_WINDOW_DAYS.
    Handles both flat (PROCESSED_DIR/*_processed.json) and dated-subfolder
    layouts (PROCESSED_DIR/YYYY/MM/*_processed.json) via recursive glob."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WEEKLY_WINDOW_DAYS)

    paths = glob.glob(str(PROCESSED_DIR / "**" / "*_processed.json"), recursive=True)
    paths += glob.glob(str(PROCESSED_DIR / "*_processed.json"))
    paths = sorted(set(paths))

    cycles = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"  Skipping unreadable processed file {p}: {e}")
            continue
        # Filter by processed_at within the window
        ts = data.get("processed_at")
        try:
            dt = datetime.fromisoformat(ts) if ts else None
            if dt and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            dt = None
        if dt is None or dt >= cutoff:
            cycles.append(data)
    logger.info(f"Weekly: loaded {len(cycles)} cycles from last {WEEKLY_WINDOW_DAYS} days")
    return cycles


def _dedupe_articles(cycles: list[dict]) -> list[dict]:
    """Concatenate articles across cycles, deduped by URL (fallback title)."""
    seen = set()
    out = []
    for c in cycles:
        for a in c.get("articles", []) or []:
            key = (a.get("url") or a.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(a)
    logger.info(f"Weekly: {len(out)} unique articles across the week")
    return out


def _build_week_digest(articles: list[dict], cap: int = 60) -> str:
    """Build a compact digest of the week's most important articles for the prompt.
    Sorts by max extraction importance so the prompt leads with what mattered."""
    scored = []
    for a in articles:
        best = 0.0
        for c in a.get("extraction", []) or []:
            if c.get("status") == "ok":
                best = max(best, float(c.get("importance_score", 0.0)))
        scored.append((best, a))
    scored.sort(key=lambda t: t[0], reverse=True)

    lines = []
    for score, a in scored[:cap]:
        title = (a.get("title") or "").strip()
        src = a.get("source", "")
        if title:
            lines.append(f"- [{src}] {title} (importance {score:.2f})")
    return "\n".join(lines) if lines else "(no material articles this week)"


# Assessed-price slots — same block the daily uses, for adviser to fill.
_ASSESSED_BLOCK = (
    "\n> **⚠ ADVISER ACTION — confirm assessed prices before sending.** "
    "Any reference figures are pre-review, not live.\n>\n"
    "> | Assessed price | Adviser to enter (Argus/Platts) |\n"
    "> |---|---|\n"
    "> | Dated Brent | __________ |\n"
    "> | Ebob gasoline | __________ |\n"
    "> | Gasoil/diesel crack | __________ |\n"
    ">\n> *Replace blanks with latest assessed numbers, then forward to SLT.*\n\n"
)

_DRAFT_BANNER = (
    "> 🟡 **DRAFT — adviser review required before forwarding to SLT.** "
    "Directional hinting only; not investment advice.\n\n"
)


def generate_weekly(now: datetime = None) -> Path | None:
    """Generate and save the weekly summary. Returns the path, or None if the
    week's data failed the health guard (nothing generated)."""
    now = now or datetime.now(timezone.utc)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cycles = _load_week_cycles(now)
    articles = _dedupe_articles(cycles)

    # ── Guard: don't publish a weekly on an empty/stale week ────────────────
    try:
        from guard import check_cycle_health, log_verdict
        verdict = check_cycle_health({"articles": articles}, now=now)
        log_verdict(verdict)
        if not verdict["publish"]:
            logger.error("Weekly: week's data failed health guard — no weekly generated.")
            return None
    except Exception as e:
        logger.warning(f"Weekly: guard unavailable ({e}); proceeding.")

    prompt = WEEKLY_PROMPT.format(
        domain_knowledge=OIL_MARKET_KNOWLEDGE,
        week_digest=_build_week_digest(articles),
    )

    logger.info("Weekly: generating via LLM...")
    report_text = _call_llm(prompt)

    # Subject from first non-empty line
    subject_line = ""
    for line in report_text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            subject_line = s[:120]
            break
    if not subject_line:
        subject_line = f"Weekly Oil Outlook — week to {now.strftime('%d/%m/%Y')}"
    _safe_subject = subject_line.replace("\\", "\\\\").replace('"', '\\"')

    header = (
        f"---\n"
        f"generated_at: {now.isoformat()}\n"
        f"report_type: weekly\n"
        f"window_days: {WEEKLY_WINDOW_DAYS}\n"
        f"cycles_used: {len(cycles)}\n"
        f"articles_used: {len(articles)}\n"
        f'subject: "{_safe_subject}"\n'
        f"---\n\n"
    )

    filename = now.strftime("WEEKLY_%Y-%m-%d_%H-%M-%S_report.md")
    filepath = REPORT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + _DRAFT_BANNER + _ASSESSED_BLOCK + report_text + "\n")

    logger.info(f"Weekly report saved: {filepath}")


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
            docx_path = filepath.with_suffix(".docx")
            result = subprocess.run(
                [pandoc_cmd, str(filepath), "-o", str(docx_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                logger.info(f"DOCX saved: {docx_path}")
            else:
                logger.warning(f"Pandoc failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"DOCX conversion error: {e}")


        
    return filepath