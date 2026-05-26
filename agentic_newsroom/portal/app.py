"""
Oil Market Report Portal — with per-user login and forgot password.
"""

import os
import glob
import json as _json
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from functools import wraps

from flask import (
    Flask, render_template_string, request,
    redirect, url_for, session, send_file, abort, jsonify
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    REPORT_DIR, PORTAL_SECRET_KEY, PORTAL_URL,
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
)
from portal.users import (
    init_db, authenticate, create_reset_token,
    validate_reset_token, reset_password
)

app = Flask(__name__)
app.secret_key = PORTAL_SECRET_KEY
init_db()

# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Templates ─────────────────────────────────────────────────────────────────

BASE_STYLE = """
<style>
    body { font-family: Georgia, serif; background: #f5f5f0; margin: 0; padding: 32px; }
    .container { max-width: 800px; margin: 0 auto; }
    .box { background: white; padding: 48px; border-radius: 4px;
           box-shadow: 0 2px 12px rgba(0,0,0,0.1); width: 340px;
           margin: 80px auto; }
    h1, h2 { color: #1a1a1a; }
    input[type=email], input[type=password], input[type=text] {
        width: 100%; padding: 10px; border: 1px solid #ddd;
        border-radius: 3px; font-size: 1em; box-sizing: border-box; margin-top: 4px; }
    label { font-size: 0.85em; color: #555; display: block; margin-top: 14px; }
    button, .btn { width: 100%; padding: 10px; background: #1a1a1a; color: white;
                   border: none; border-radius: 3px; font-size: 1em;
                   cursor: pointer; margin-top: 16px; text-align: center;
                   text-decoration: none; display: block; }
    button:hover, .btn:hover { background: #333; }
    .link { text-align: center; margin-top: 14px; font-size: 0.85em; color: #888; }
    .link a { color: #555; }
    .error { color: #c0392b; font-size: 0.85em; margin-top: 10px; }
    .success { color: #27ae60; font-size: 0.85em; margin-top: 10px; }
    .nav a { color: #888; text-decoration: none; margin-right: 16px; font-size: 0.85em; }
    .nav a:hover { color: #333; }
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
                text-decoration: none; border-radius: 3px; font-size: 0.85em; width: auto; }
    .download:hover { background: #333; }
    .empty { color: #888; text-align: center; padding: 48px; }
</style>
"""

LOGIN_HTML = """<!DOCTYPE html><html><head><title>Oil Market Reports — Login</title>
<meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """
</head><body>
<div class="box">
    <h2>Oil Market Reports</h2>
    <p style="color:#666;font-size:0.9em;margin:0 0 20px">Sign in to access reports.</p>
    <form method="post">
        <label>Email</label>
        <input type="email" name="email" autofocus required>
        <label>Password</label>
        <input type="password" name="password" required>
        <button type="submit">Sign In</button>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
    </form>
    <div class="link"><a href="/forgot-password">Forgot password?</a></div>
</div>
</body></html>"""

FORGOT_HTML = """<!DOCTYPE html><html><head><title>Reset Password</title>
<meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """
</head><body>
<div class="box">
    <h2>Reset Password</h2>
    <p style="color:#666;font-size:0.9em;margin:0 0 20px">Enter your email and we'll send a reset link.</p>
    <form method="post">
        <label>Email</label>
        <input type="email" name="email" autofocus required>
        <button type="submit">Send Reset Link</button>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        {% if success %}<div class="success">{{ success }}</div>{% endif %}
    </form>
    <div class="link"><a href="/login">Back to Sign In</a></div>
</div>
</body></html>"""

RESET_HTML = """<!DOCTYPE html><html><head><title>Set New Password</title>
<meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """
</head><body>
<div class="box">
    <h2>Set New Password</h2>
    <form method="post">
        <label>New Password</label>
        <input type="password" name="password" required minlength="8">
        <label>Confirm Password</label>
        <input type="password" name="password2" required minlength="8">
        <button type="submit">Reset Password</button>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
        {% if success %}<div class="success">{{ success }}</div>{% endif %}
    </form>
</div>
</body></html>"""

REPORTS_HTML = """<!DOCTYPE html><html><head><title>Oil Market Reports</title>
<meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """
</head><body>
<div class="container">
    <div class="nav" style="margin-bottom:24px">
        <a href="/chat">Query Intelligence →</a>
        <a href="/logout" style="float:right">Logout ({{ user }})</a>
    </div>
    <h1>Oil Market Reports</h1>
    <p class="subtitle">Generated twice daily from live market data.</p>
    {% if reports %}
        {% for r in reports %}
        <div class="report">
            <div class="report-info">
                <h3><span class="badge {{ r.direction }}">{{ r.direction|upper }}</span>{{ r.date }}</h3>
                <p>{{ r.confidence }} confidence &nbsp;·&nbsp; WTI ${{ r.wti }} &nbsp;·&nbsp; Brent ${{ r.brent }}</p>
            </div>
            <a class="download" href="/download/{{ r.filename }}">Download</a>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty">No reports available yet.</div>
    {% endif %}
</div>
</body></html>"""

CHAT_HTML = """<!DOCTYPE html><html><head><title>Query Intelligence</title>
<meta name="viewport" content="width=device-width, initial-scale=1">""" + BASE_STYLE + """
<style>
    .search-box { display:flex; gap:8px; margin-bottom:16px; }
    .search-box input { flex:1; }
    .search-box button { width:auto; margin-top:0; padding:10px 20px; }
    .suggestion { display:inline-block; background:#f0f0f0; padding:4px 12px;
                  border-radius:12px; font-size:0.82em; color:#555;
                  cursor:pointer; margin:4px; }
    .suggestion:hover { background:#e0e0e0; }
    .answer-box { background:white; padding:24px; border-radius:4px;
                  box-shadow:0 1px 4px rgba(0,0,0,0.08); margin-top:16px; }
    .answer-text { line-height:1.7; white-space:pre-wrap; }
    .source { font-size:0.82em; color:#888; padding:4px 0; border-bottom:1px solid #f0f0f0; }
    .source a { color:#555; text-decoration:none; }
</style>
</head><body>
<div class="container">
    <div class="nav" style="margin-bottom:24px">
        <a href="/">← Reports</a>
        <a href="/logout" style="float:right">Logout</a>
    </div>
    <h1>Query Oil Market Intelligence</h1>
    <p class="subtitle">Ask questions about news in the database.</p>
    <div style="margin-bottom:16px">
        <span class="suggestion" onclick="ask('What is the latest on Iran and Hormuz strait?')">Iran & Hormuz</span>
        <span class="suggestion" onclick="ask('What did OPEC say about production cuts?')">OPEC cuts</span>
        <span class="suggestion" onclick="ask('What is Trump saying about oil and energy?')">Trump on energy</span>
        <span class="suggestion" onclick="ask('Latest Russia Ukraine impact on oil?')">Russia Ukraine</span>
        <span class="suggestion" onclick="ask('Brent WTI price trend this week?')">Price trend</span>
        <span class="suggestion" onclick="ask('Any tanker or shipping incidents?')">Tanker news</span>
    </div>
    <div class="search-box">
        <input type="text" id="q" placeholder="e.g. What is the latest on Iran sanctions?" onkeydown="if(event.key==='Enter')ask(document.getElementById('q').value)">
        <button onclick="ask(document.getElementById('q').value)">Ask</button>
    </div>
    <div id="result"></div>
</div>
<script>
function ask(q){
    if(!q.trim()) return;
    document.getElementById('q').value = q;
    document.getElementById('result').innerHTML = '<div class="answer-box"><p style="color:#888;font-style:italic">Searching and generating answer...</p></div>';
    fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})
    .then(r=>r.json()).then(data=>{
        if(data.error){document.getElementById('result').innerHTML='<div class="answer-box">'+data.error+'</div>';return;}
        let src=data.sources.map(s=>'<div class="source">['+s.score+'] <strong>'+s.source+'</strong> — '+(s.url?'<a href="'+s.url+'" target="_blank">'+(s.title||s.url).slice(0,80)+'</a>':(s.title||'').slice(0,80))+' ('+( s.published||'').slice(0,10)+')</div>').join('');
        document.getElementById('result').innerHTML='<div class="answer-box"><h3 style="margin:0 0 12px;color:#555">Answer</h3><div class="answer-text">'+data.answer+'</div><div style="margin-top:16px"><strong style="font-size:0.85em;color:#888">Sources</strong>'+src+'</div></div>';
    });
}
</script>
</body></html>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _send_reset_email(email: str, token: str):
    reset_url = f"{PORTAL_URL}/reset-password/{token}"
    msg = MIMEMultipart()
    msg["From"]    = SMTP_FROM
    msg["To"]      = email
    msg["Subject"] = "Oil Market Reports — Password Reset"
    body = f"""You requested a password reset for Oil Market Reports.

Click the link below to set a new password (valid for 1 hour):
{reset_url}

If you did not request this, ignore this email.
"""
    msg.attach(MIMEText(body, "plain"))
    if SMTP_PORT == 465:
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
    else:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
    server.login(SMTP_USER, SMTP_PASSWORD)
    server.send_message(msg)
    server.quit()


def _parse_report_meta(filepath: Path) -> dict:
    meta = {"filename": filepath.name,
            "date": filepath.stem.replace("_report","").replace("_"," "),
            "direction":"neutral","confidence":"low","score":"N/A","wti":"N/A","brent":"N/A"}
    try:
        for line in filepath.read_text().splitlines()[1:10]:
            for key in ("direction","confidence","score","wti","brent"):
                if line.startswith(f"{key}:"):
                    meta[key] = line.split(":",1)[1].strip().split()[0]
    except Exception:
        pass
    return meta


def _list_reports() -> list[dict]:
    docx = sorted(glob.glob(str(REPORT_DIR/"*_report.docx")), reverse=True)
    mds  = sorted(glob.glob(str(REPORT_DIR/"*_report.md")),   reverse=True)
    seen, reports = set(), []
    for f in docx:
        stem = Path(f).stem
        seen.add(stem)
        md   = REPORT_DIR/f"{stem}.md"
        meta = _parse_report_meta(md) if md.exists() else {}
        meta["filename"] = Path(f).name
        reports.append(meta)
    for f in mds:
        stem = Path(f).stem
        if stem not in seen:
            reports.append(_parse_report_meta(Path(f)))
    return reports


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@login_required
def index():
    user = session.get("user", {})
    return render_template_string(REPORTS_HTML, reports=_list_reports(),
                                  user=user.get("email",""))


@app.route("/login", methods=["GET","POST"])
def login():
    if session.get("user"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email","").strip()
        password = request.form.get("password","")
        user     = authenticate(email, password)
        if user:
            session["user"] = user
            return redirect(url_for("index"))
        error = "Incorrect email or password."
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET","POST"])
def forgot_password():
    error, success = None, None
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        token = create_reset_token(email)
        # Always show success to prevent email enumeration
        if token:
            try:
                _send_reset_email(email, token)
            except Exception as e:
                error = f"Could not send email: {e}"
        if not error:
            success = "If that email is registered, a reset link has been sent."
    return render_template_string(FORGOT_HTML, error=error, success=success)


@app.route("/reset-password/<token>", methods=["GET","POST"])
def reset_password_page(token):
    email   = validate_reset_token(token)
    error, success = None, None
    if not email:
        error = "This reset link is invalid or has expired."
    elif request.method == "POST":
        pw  = request.form.get("password","")
        pw2 = request.form.get("password2","")
        if len(pw) < 8:
            error = "Password must be at least 8 characters."
        elif pw != pw2:
            error = "Passwords do not match."
        else:
            if reset_password(token, pw):
                success = "Password reset successfully. You can now sign in."
            else:
                error = "Reset failed — link may have expired."
    return render_template_string(RESET_HTML, error=error, success=success)


@app.route("/download/<filename>")
@login_required
def download(filename):
    safe = REPORT_DIR / Path(filename).name
    if not safe.exists():
        abort(404)
    return send_file(safe, as_attachment=True)


@app.route("/chat")
@login_required
def chat():
    return render_template_string(CHAT_HTML)


@app.route("/query", methods=["POST"])
@login_required
def query_rag():
    from query import query as run_query
    data     = request.get_json()
    question = (data or {}).get("question","").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        return jsonify(run_query(question, n_results=8))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 8080))
    debug = os.getenv("FLASK_DEBUG","false").lower() == "true"
    print(f"Portal running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)