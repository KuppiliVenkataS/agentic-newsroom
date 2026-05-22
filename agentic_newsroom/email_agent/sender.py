
import csv
import glob
import logging
import smtplib
"""
Email agent.

Sends the latest oil market report as a .docx attachment via any SMTP provider.

Recipients come from two sources (merged, deduplicated):
1. FIXED_RECIPIENTS list in config/settings.py
2. recipients.csv in this folder

Run manually:
    python email_agent/send_now.py

SMTP settings go in .env:
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=465
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=your_app_password
    SMTP_FROM=you@gmail.com

Common provider settings:
    Gmail:   smtp.gmail.com, port 465, use app password
    Outlook: smtp.office365.com, port 587
    Zoho:    smtp.zoho.com, port 465
    Any host: check your provider's SMTP docs
"""

import csv
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from config.settings import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM,
    FIXED_RECIPIENTS,
    REPORT_DIR,
    PORTAL_URL,
    PORTAL_PASSWORD,
)

logger = logging.getLogger(__name__)

RECIPIENTS_CSV = Path(__file__).parent / "recipients.csv"

EMAIL_SUBJECT = "Oil Market Briefing — {date} | {direction} ({confidence})"

EMAIL_BODY = """Dear {name},

The latest Oil Market Briefing Report is ready for {datetime}.

Key Signal: {direction} with {confidence} confidence (score: {score})
WTI: ${wti} | Brent: ${brent}

View and download your report here:
{portal_url}

Portal password: {portal_password}

This report is generated automatically from live news feeds, EIA market data,
and GDELT global news analysis.

Regards,
Agentic Newsroom
"""


def _load_recipients() -> list[dict]:
    """
    Merge recipients from CSV file and FIXED_RECIPIENTS in settings.
    Returns deduplicated list of {name, email} dicts.
    """
    seen   = set()
    result = []

    for r in FIXED_RECIPIENTS:
        email = r.get("email", "").strip().lower()
        if email and email not in seen:
            seen.add(email)
            result.append(r)

    if RECIPIENTS_CSV.exists():
        with open(RECIPIENTS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                email = row.get("email", "").strip().lower()
                name  = row.get("name", "").strip()
                if email and email not in seen and "@example.com" not in email:
                    seen.add(email)
                    result.append({"name": name, "email": email})

    logger.info(f"Recipients loaded: {len(result)}")
    return result


def _find_latest_report() -> tuple:
    """Find the most recent .md and .docx report files."""
    md_files   = sorted(glob.glob(str(REPORT_DIR / "*_report.md")))
    docx_files = sorted(glob.glob(str(REPORT_DIR / "*_report.docx")))
    md_path    = Path(md_files[-1])   if md_files   else None
    docx_path  = Path(docx_files[-1]) if docx_files else None
    return md_path, docx_path


def _build_message(recipient: dict, prediction: dict,
                   docx_path, md_path) -> MIMEMultipart:
    signals    = prediction.get("signals", {})
    eia        = signals.get("eia", {})
    direction  = prediction.get("direction", "neutral").upper()
    confidence = prediction.get("confidence", "low")
    score      = prediction.get("score", 0.0)
    wti        = eia.get("wti_latest", {}).get("value", "N/A")
    brent      = eia.get("brent_latest", {}).get("value", "N/A")
    now        = datetime.now(timezone.utc)

    msg = MIMEMultipart()
    msg["From"]    = SMTP_FROM
    msg["To"]      = recipient["email"]
    msg["Subject"] = EMAIL_SUBJECT.format(
        date      = now.strftime("%Y-%m-%d"),
        direction = direction,
        confidence= confidence,
    )

    body = EMAIL_BODY.format(
        name            = recipient.get("name", ""),
        datetime        = now.strftime("%Y-%m-%d %H:%M UTC"),
        direction       = direction,
        confidence      = confidence,
        score           = score,
        wti             = wti,
        brent           = brent,
        portal_url      = PORTAL_URL,
        portal_password = PORTAL_PASSWORD,
    )
    msg.attach(MIMEText(body, "plain"))

    return msg


def send_report(prediction: dict) -> dict:
    """
    Send the latest report to all recipients via SMTP.
    Returns summary dict with sent/failed counts.
    """
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        logger.error("SMTP credentials not set — check .env")
        return {"sent": 0, "failed": 0, "error": "credentials_missing"}

    recipients          = _load_recipients()
    md_path, docx_path  = _find_latest_report()

    if not recipients:
        logger.warning("No recipients found.")
        return {"sent": 0, "failed": 0, "error": "no_recipients"}

    if not md_path and not docx_path:
        logger.error("No report file found to attach.")
        return {"sent": 0, "failed": 0, "error": "no_report_file"}

    sent   = 0
    failed = 0
    errors = []

    try:
        # Port 465 uses SSL, port 587 uses TLS/STARTTLS
        if SMTP_PORT == 465:
            server_ctx = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        else:
            server_ctx = smtplib.SMTP(SMTP_HOST, SMTP_PORT)

        with server_ctx as server:
            if SMTP_PORT == 587:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            logger.info(f"SMTP authenticated: {SMTP_HOST}:{SMTP_PORT}")

            for recipient in recipients:
                try:
                    msg = _build_message(recipient, prediction, docx_path, md_path)
                    server.send_message(msg)
                    logger.info(f"  Sent to: {recipient['email']}")
                    sent += 1
                except Exception as e:
                    logger.warning(f"  Failed: {recipient['email']}: {e}")
                    errors.append({"email": recipient["email"], "error": str(e)})
                    failed += 1

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed — check credentials")
        return {"sent": 0, "failed": len(recipients), "error": "auth_failed"}
    except Exception as e:
        logger.error(f"SMTP connection failed: {e}")
        return {"sent": 0, "failed": len(recipients), "error": str(e)}

    logger.info(f"Email agent done: {sent} sent, {failed} failed")
    return {"sent": sent, "failed": failed, "errors": errors}