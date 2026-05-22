"""
Oil Market Report Portal.

Simple Flask web app that serves reports behind a login page.
Clients visit the URL, enter the password, and download reports.

Run locally:
    python portal/app.py

Access:
    http://localhost:8080  (local)
    http://YOUR_MAC_IP:8080  (from client browser on same network)
    https://your-app.onrender.com  (after cloud deploy)

For cloud deploy (Render/Railway):
    - Push this repo to GitHub
    - Connect to Render, set start command: python portal/app.py
    - Set environment variables from .env in Render dashboard
"""

import os
import glob
from datetime import datetime
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template_string, request,
    redirect, url_for, session, send_file, abort
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import REPORT_DIR, PORTAL_PASSWORD, PORTAL_SECRET_KEY

app = Flask(__name__)
app.secret_key = PORTAL_SECRET_KEY

# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Templates ─────────────────────────────────────────────────────────────────

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Oil Market Reports — Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Georgia, serif; background: #f5f5f0;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; margin: 0; }
        .box { background: white; padding: 48px; border-radius: 4px;
               box-shadow: 0 2px 12px rgba(0,0,0,0.1); width: 320px; }
        h2 { margin: 0 0 8px; color: #1a1a1a; font-size: 1.4em; }
        p  { margin: 0 0 24px; color: #666; font-size: 0.9em; }
        input { width: 100%; padding: 10px; border: 1px solid #ddd;
                border-radius: 3px; font-size: 1em; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background: #1a1a1a; color: white;
                 border: none; border-radius: 3px; font-size: 1em;
                 cursor: pointer; margin-top: 12px; }
        button:hover { background: #333; }
        .error { color: #c0392b; font-size: 0.85em; margin-top: 8px; }
    </style>
</head>
<body>
<div class="box">
    <h2>Oil Market Reports</h2>
    <p>Enter your access password to view reports.</p>
    <form method="post">
        <input type="password" name="password" placeholder="Password" autofocus>
        <button type="submit">Access Reports</button>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </form>
</div>
</body>
</html>
"""

REPORTS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Oil Market Reports</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Georgia, serif; background: #f5f5f0;
               margin: 0; padding: 32px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #1a1a1a; font-size: 1.6em; margin-bottom: 4px; }
        .subtitle { color: #666; font-size: 0.9em; margin-bottom: 32px; }
        .report { background: white; padding: 20px 24px; margin-bottom: 12px;
                  border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                  display: flex; justify-content: space-between; align-items: center; }
        .report-info h3 { margin: 0 0 4px; font-size: 1em; color: #1a1a1a; }
        .report-info p  { margin: 0; font-size: 0.85em; color: #888; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
                 font-size: 0.78em; font-weight: bold; margin-right: 6px; }
        .bullish { background: #e8f5e9; color: #2e7d32; }
        .bearish { background: #fce4ec; color: #c62828; }
        .neutral { background: #f3f3f3; color: #555; }
        .download { padding: 8px 18px; background: #1a1a1a; color: white;
                    text-decoration: none; border-radius: 3px; font-size: 0.85em; }
        .download:hover { background: #333; }
        .logout { float: right; color: #888; font-size: 0.85em;
                  text-decoration: none; margin-top: 4px; }
        .logout:hover { color: #333; }
        .empty { color: #888; text-align: center; padding: 48px; }
    </style>
</head>
<body>
<div class="container">
    <a href="/logout" class="logout">Logout</a>
    <h1>Oil Market Reports</h1>
    <p class="subtitle">Generated every 12 hours from live market data.</p>

    {% if reports %}
        {% for r in reports %}
        <div class="report">
            <div class="report-info">
                <h3>
                    <span class="badge {{ r.direction }}">{{ r.direction|upper }}</span>
                    {{ r.date }}
                </h3>
                <p>{{ r.confidence }} confidence &nbsp;·&nbsp; Score: {{ r.score }}
                   &nbsp;·&nbsp; WTI ${{ r.wti }} &nbsp;·&nbsp; Brent ${{ r.brent }}</p>
            </div>
            <a class="download" href="/download/{{ r.filename }}">Download</a>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty">No reports available yet.</div>
    {% endif %}
</div>
</body>
</html>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_report_meta(filepath: Path) -> dict:
    """Read YAML front matter from a report .md file."""
    meta = {
        "filename": filepath.name,
        "date":     filepath.stem.replace("_report", "").replace("_", " "),
        "direction":"neutral",
        "confidence":"low",
        "score":    "N/A",
        "wti":      "N/A",
        "brent":    "N/A",
    }
    try:
        lines = filepath.read_text().splitlines()
        for line in lines[1:10]:
            if line.startswith("direction:"):
                meta["direction"] = line.split(":", 1)[1].strip()
            elif line.startswith("confidence:"):
                meta["confidence"] = line.split(":", 1)[1].strip()
            elif line.startswith("score:"):
                meta["score"] = line.split(":", 1)[1].strip()
            elif line.startswith("wti:"):
                meta["wti"] = line.split(":", 1)[1].strip().split()[0]
            elif line.startswith("brent:"):
                meta["brent"] = line.split(":", 1)[1].strip().split()[0]
    except Exception:
        pass
    return meta


def _list_reports() -> list[dict]:
    """List all reports sorted newest first."""
    # Prefer .docx, fall back to .md
    docx_files = sorted(
        glob.glob(str(REPORT_DIR / "*_report.docx")), reverse=True
    )
    md_files   = sorted(
        glob.glob(str(REPORT_DIR / "*_report.md")), reverse=True
    )

    seen    = set()
    reports = []

    for f in docx_files:
        stem = Path(f).stem
        seen.add(stem)
        md_match = REPORT_DIR / f"{stem}.md"
        meta = _parse_report_meta(md_match) if md_match.exists() else {}
        meta["filename"] = Path(f).name
        reports.append(meta)

    for f in md_files:
        stem = Path(f).stem
        if stem not in seen:
            meta = _parse_report_meta(Path(f))
            reports.append(meta)

    return reports


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@login_required
def index():
    reports = _list_reports()
    return render_template_string(REPORTS_HTML, reports=reports)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == PORTAL_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Incorrect password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/download/<filename>")
@login_required
def download(filename):
    # Security: only serve files from REPORT_DIR, no path traversal
    safe_path = REPORT_DIR / Path(filename).name
    if not safe_path.exists():
        abort(404)
    return send_file(safe_path, as_attachment=True)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"Portal running at http://localhost:{port}")
    print(f"Reports served from: {REPORT_DIR}")
    app.run(host="0.0.0.0", port=port, debug=debug)