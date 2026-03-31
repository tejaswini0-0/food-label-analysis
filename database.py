"""
database.py — NutriScan Supabase Integration
=============================================
Provides a unified DB interface that wraps both the local JSON store
(auth.py) and Supabase cloud storage.

SETUP (5 minutes):
  1. Create a free project at https://supabase.com
  2. Run the SQL in SCHEMA.sql (below) in the Supabase SQL editor
  3. Add to .streamlit/secrets.toml:
       SUPABASE_URL = "https://xxxx.supabase.co"
       SUPABASE_KEY = "your-anon-public-key"

SCHEMA (run once in Supabase SQL editor):
-------------------------------------------
create table users (
    id uuid primary key default gen_random_uuid(),
    username text unique not null,
    display_name text,
    pw_hash text not null,
    salt text not null,
    created_at timestamptz default now(),
    profile jsonb default '{}'
);

create table scans (
    id uuid primary key default gen_random_uuid(),
    username text not null references users(username) on delete cascade,
    product text,
    category text,
    score int,
    grade text,
    brand text,
    notes text,
    nutrients jsonb,
    saved_at timestamptz default now()
);

create index scans_username_idx on scans(username);
create index scans_saved_at_idx on scans(saved_at desc);
-------------------------------------------

If SUPABASE_URL / SUPABASE_KEY are not set, all calls fall back to
the local JSON store in auth.py transparently.
"""

import json
from typing import Optional, Dict, List, Any
import urllib.request
import urllib.error


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
def _supabase_config() -> tuple[Optional[str], Optional[str]]:
    """Return (url, key) or (None, None) if not configured."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
        if url and key:
            return url.rstrip("/"), key
    except Exception:
        pass
    # Fall back to environment variables
    import os
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if url and key:
        return url.rstrip("/"), key
    return None, None


def supabase_enabled() -> bool:
    url, key = _supabase_config()
    return bool(url and key)


# ─────────────────────────────────────────────
# LOW-LEVEL HTTP  (no SDK needed)
# ─────────────────────────────────────────────
def _sb_request(method: str, path: str, body: Optional[dict] = None,
                params: Optional[str] = None) -> Optional[Any]:
    """
    Make a Supabase PostgREST API request.
    Returns parsed JSON list on success, None on failure.
    Uses print() for errors — safe outside Streamlit UI context.
    """
    url, key = _supabase_config()
    if not url:
        return None

    full_url = f"{url}/rest/v1/{path}"
    if params:
        full_url += "?" + params

    # Supabase requires Content-Type even on DELETE/GET for some versions
    headers = {
        "apikey":        key,
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }

    # Ensure nutrients dict is JSON-serialisable (no float subclasses etc.)
    if body and "nutrients" in body and isinstance(body["nutrients"], dict):
        body = dict(body)
        body["nutrients"] = json.loads(json.dumps(body["nutrients"], default=str))

    data = json.dumps(body, default=str).encode() if body else None
    req  = urllib.request.Request(full_url, data=data, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else []
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"[NutriScan] Supabase {method} {path} → HTTP {e.code}: {err_body[:300]}")
        return None
    except urllib.error.URLError as e:
        print(f"[NutriScan] Supabase connection error: {e.reason}")
        return None
    except Exception as e:
        print(f"[NutriScan] Supabase unexpected error: {e}")
        return None


# ─────────────────────────────────────────────
# USER OPERATIONS
# ─────────────────────────────────────────────
def sb_get_user(username: str) -> Optional[Dict]:
    """Fetch user record by username."""
    result = _sb_request("GET", "users", params=f"username=eq.{username}&limit=1")
    if result and len(result) > 0:
        return result[0]
    return None


def sb_create_user(username: str, display_name: str,
                   pw_hash: str, salt: str, profile: dict) -> bool:
    """Insert a new user. Returns True on success."""
    result = _sb_request("POST", "users", body={
        "username":     username,
        "display_name": display_name,
        "pw_hash":      pw_hash,
        "salt":         salt,
        "profile":      profile,
    })
    return result is not None


def sb_update_profile(username: str, profile: dict) -> bool:
    """Update user profile JSON."""
    result = _sb_request("PATCH", "users",
                         body={"profile": profile},
                         params=f"username=eq.{username}")
    return result is not None


# ─────────────────────────────────────────────
# SCAN OPERATIONS
# ─────────────────────────────────────────────
def sb_save_scan(username: str, scan: dict) -> bool:
    """Insert a new scan row. Returns True on success."""
    nutrients = scan.get("nutrients", {})
    # Ensure all nutrient values are plain Python floats, not numpy types
    clean_nutrients = {k: float(v) if v is not None else None
                       for k, v in nutrients.items()}
    result = _sb_request("POST", "scans", body={
        "username":  username,
        "product":   scan.get("product", "Unknown"),
        "category":  scan.get("category", ""),
        "score":     int(scan.get("score", 0)),
        "grade":     scan.get("grade", "F"),
        "brand":     scan.get("brand") or None,
        "notes":     scan.get("notes", "") or None,
        "nutrients": clean_nutrients,
    })
    if result is None:
        print(f"[NutriScan] Failed to save scan for {username}")
    return result is not None


def sb_load_scans(username: str, limit: int = 200) -> List[Dict]:
    """Fetch all scans for a user, newest first."""
    result = _sb_request(
        "GET", "scans",
        params=f"username=eq.{username}&order=saved_at.desc&limit={limit}"
    )
    if not result:
        return []
    # Normalise to match local JSON format
    scans = []
    for row in result:
        scans.append({
            "product":   row.get("product", "Unknown"),
            "category":  row.get("category", ""),
            "score":     row.get("score", 0),
            "grade":     row.get("grade", "F"),
            "brand":     row.get("brand"),
            "notes":     row.get("notes", ""),
            "nutrients": row.get("nutrients", {}),
            "saved_at":  (row.get("saved_at") or "")[:10],
            "_sb_id":    row.get("id"),
        })
    return scans


def sb_delete_scan(scan_sb_id: str) -> bool:
    """Delete a scan by its Supabase UUID."""
    result = _sb_request("DELETE", "scans", params=f"id=eq.{scan_sb_id}")
    return result is not None


def sb_clear_scans(username: str) -> bool:
    """Delete all scans for a user."""
    result = _sb_request("DELETE", "scans", params=f"username=eq.{username}")
    return result is not None


def sb_get_stats(username: str) -> Dict:
    """Aggregate stats for a user (count, avg score, best grade)."""
    scans = sb_load_scans(username, limit=500)
    if not scans:
        return {"total": 0, "avg_score": 0, "best_score": 0}
    scores = [s["score"] for s in scans]
    return {
        "total":      len(scans),
        "avg_score":  round(sum(scores) / len(scores), 1),
        "best_score": max(scores),
    }


# ─────────────────────────────────────────────
# UNIFIED INTERFACE  (used by auth.py wrappers)
# ─────────────────────────────────────────────
def save_scan_db(username: str, scan: dict):
    """
    Dual-write: always local JSON + Supabase if configured.
    Logs success/failure to terminal for debugging.
    """
    from auth import save_user_scan
    save_user_scan(username, scan)
    print(f"[NutriScan] Saved scan locally for {username}: {scan.get('product','?')}")

    if supabase_enabled():
        ok = sb_save_scan(username, scan)
        if ok:
            print(f"[NutriScan] ✓ Supabase scan saved for {username}")
        else:
            print(f"[NutriScan] ✗ Supabase scan FAILED for {username} — check terminal for HTTP error above")
    else:
        print("[NutriScan] Supabase not configured — scan saved locally only")


def load_scans_db(username: str) -> List[Dict]:
    """
    Load from Supabase if available (cloud-first), else local JSON.
    """
    if supabase_enabled():
        scans = sb_load_scans(username)
        if scans:
            return scans
    from auth import load_user_scans
    return load_user_scans(username)


def register_user_db(username: str, password: str, display_name: str) -> tuple[bool, str]:
    """
    Register user in both local JSON and Supabase.
    """
    from auth import register_user
    ok, msg = register_user(username, password, display_name)
    if not ok:
        return False, msg

    if supabase_enabled():
        # Read back the hashes that auth.py just wrote
        from auth import _load_users
        users = _load_users()
        u = users.get(username.strip().lower(), {})
        sb_create_user(
            username    = username.strip().lower(),
            display_name= display_name or username.title(),
            pw_hash     = u.get("pw_hash", ""),
            salt        = u.get("salt", ""),
            profile     = u.get("profile", {}),
        )
    return True, msg


def update_profile_db(username: str, profile: dict):
    """Sync profile to both stores."""
    from auth import update_profile
    update_profile(username, profile)
    if supabase_enabled():
        sb_update_profile(username, profile)