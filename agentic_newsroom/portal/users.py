"""
User management for the portal.

Stores users in SQLite with hashed passwords.
Supports: create, authenticate, forgot password (email token), change password.

DB location: STORAGE_ROOT/portal_users.db
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config.settings import STORAGE_ROOT

USERS_DB = STORAGE_ROOT /"data" / "portal_users.db"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 260000
    ).hex()


def _get_conn():
    USERS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(USERS_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT UNIQUE NOT NULL,
            name       TEXT,
            password   TEXT NOT NULL,
            salt       TEXT NOT NULL,
            role       TEXT DEFAULT 'subscriber',
            active     INTEGER DEFAULT 1,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reset_tokens (
            token      TEXT PRIMARY KEY,
            email      TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def create_user(email: str, password: str, name: str = "", role: str = "subscriber") -> dict:
    """Create a new user. Returns user dict or raises on duplicate email."""
    salt     = secrets.token_hex(16)
    hashed   = _hash_password(password, salt)
    now      = datetime.now(timezone.utc).isoformat()
    conn     = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email, name, password, salt, role, created_at) VALUES (?,?,?,?,?,?)",
            (email.lower().strip(), name, hashed, salt, role, now)
        )
        conn.commit()
        return {"email": email, "name": name, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError(f"Email already exists: {email}")
    finally:
        conn.close()


def authenticate(email: str, password: str) -> dict | None:
    """Returns user dict if credentials are valid, None otherwise."""
    conn = _get_conn()
    row  = conn.execute(
        "SELECT * FROM users WHERE email=? AND active=1",
        (email.lower().strip(),)
    ).fetchone()
    conn.close()

    if not row:
        return None

    expected = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected, row["password"]):
        return None

    return {"email": row["email"], "name": row["name"], "role": row["role"]}


def list_users() -> list[dict]:
    conn  = _get_conn()
    rows  = conn.execute("SELECT email, name, role, active, created_at FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def deactivate_user(email: str):
    conn = _get_conn()
    conn.execute("UPDATE users SET active=0 WHERE email=?", (email.lower(),))
    conn.commit()
    conn.close()


def create_reset_token(email: str) -> str | None:
    """Create a password reset token. Returns token or None if email not found."""
    conn = _get_conn()
    row  = conn.execute("SELECT email FROM users WHERE email=? AND active=1", (email.lower(),)).fetchone()
    if not row:
        conn.close()
        return None

    token      = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    conn.execute(
        "INSERT INTO reset_tokens (token, email, expires_at) VALUES (?,?,?)",
        (token, email.lower(), expires_at)
    )
    conn.commit()
    conn.close()
    return token


def validate_reset_token(token: str) -> str | None:
    """Returns email if token is valid and unused, None otherwise."""
    conn = _get_conn()
    row  = conn.execute(
        "SELECT * FROM reset_tokens WHERE token=? AND used=0",
        (token,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    expires = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires:
        return None

    return row["email"]


def reset_password(token: str, new_password: str) -> bool:
    """Reset password using a valid token. Returns True on success."""
    email = validate_reset_token(token)
    if not email:
        return False

    salt   = secrets.token_hex(16)
    hashed = _hash_password(new_password, salt)
    conn   = _get_conn()
    conn.execute("UPDATE users SET password=?, salt=? WHERE email=?", (hashed, salt, email))
    conn.execute("UPDATE reset_tokens SET used=1 WHERE token=?", (token,))
    conn.commit()
    conn.close()
    return True


def change_password(email: str, new_password: str):
    salt   = secrets.token_hex(16)
    hashed = _hash_password(new_password, salt)
    conn   = _get_conn()
    conn.execute("UPDATE users SET password=?, salt=? WHERE email=?", (hashed, salt, email.lower()))
    conn.commit()
    conn.close()