"""
auth.py — NutriScan Local Authentication & Scan History
=========================================================
Provides:
  - Local file-based user store (users.json) — swap for Supabase in Phase 2
  - Hashed password storage (bcrypt via hashlib PBKDF2)
  - login / register / logout flow
  - Per-user persistent scan history (scans/<username>.json)
  - Dashboard: category breakdown, score trends, full scan log

Phase 2 note:
  Replace _load_users / _save_users / load_user_scans / save_user_scan
  with Supabase client calls. The rest of the UI stays identical.
"""

import streamlit as st
import json
import os
import hashlib
import secrets
import datetime
from typing import Optional, Dict, List

# ── Storage paths (local, Phase 1) ───────────────────────────────────────────
_DATA_DIR   = os.path.join(os.path.dirname(__file__), ".nutriscan_data")
_USERS_FILE = os.path.join(_DATA_DIR, "users.json")
_SCANS_DIR  = os.path.join(_DATA_DIR, "scans")

os.makedirs(_SCANS_DIR, exist_ok=True)

# ── Colour palette (must match app.py P dict) ────────────────────────────────
P = {
    "bg":       "#0d0e14",
    "surface":  "#16181f",
    "surface2": "#1d1f2a",
    "border":   "#272a38",
    "accent":   "#f0a500",
    "good":     "#5cb88a",
    "warn":     "#e8884a",
    "bad":      "#d95555",
    "text":     "#eceef8",
    "muted":    "#666b8a",
}


# ─────────────────────────────────────────────
# PASSWORD HASHING  (PBKDF2-HMAC-SHA256)
# ─────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Returns (hashed_hex, salt_hex). Pass existing salt to verify."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return h.hex(), salt

def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, stored_hash)


# ─────────────────────────────────────────────
# USER STORE  (local JSON — replace with DB in Phase 2)
# ─────────────────────────────────────────────

def _load_users() -> Dict:
    if not os.path.exists(_USERS_FILE):
        return {}
    try:
        with open(_USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: Dict):
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def register_user(username: str, password: str, display_name: str = "") -> tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not re_username(username):
        return False, "Username can only contain letters, numbers, and underscores."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    if username in users:
        return False, "Username already taken. Please choose another."
    pw_hash, salt = _hash_password(password)
    users[username] = {
        "display_name": display_name.strip() or username.title(),
        "pw_hash": pw_hash,
        "salt":    salt,
        "created": datetime.datetime.now().isoformat(),
        "profile": {
            "weight_kg": 0., "height_cm": 0., "age": 0,
            "sex": "Male", "activity": "Moderately Active", "goal": "Balanced",
        },
    }
    _save_users(users)
    return True, "Account created successfully."

def login_user(username: str, password: str) -> tuple[bool, str, Optional[Dict]]:
    """Returns (success, message, user_dict_or_None)."""
    username = username.strip().lower()
    users = _load_users()
    if username not in users:
        return False, "No account found with that username.", None
    u = users[username]
    if not _verify_password(password, u["pw_hash"], u["salt"]):
        return False, "Incorrect password.", None
    return True, "Logged in.", u

def update_profile(username: str, profile: Dict):
    users = _load_users()
    if username in users:
        users[username]["profile"] = profile
        _save_users(users)

import re as _re
def re_username(s: str) -> bool:
    return bool(_re.match(r"^[a-z0-9_]{3,30}$", s))


# ─────────────────────────────────────────────
# SCAN HISTORY  (per-user JSON file)
# Phase 2: replace with Supabase table queries
# ─────────────────────────────────────────────

def _scans_path(username: str) -> str:
    return os.path.join(_SCANS_DIR, f"{username}.json")

def load_user_scans(username: str) -> List[Dict]:
    path = _scans_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_user_scan(username: str, scan: Dict):
    """Prepend new scan; keep last 200."""
    scans = load_user_scans(username)
    scan["saved_at"] = datetime.datetime.now().isoformat()
    scans.insert(0, scan)
    if len(scans) > 200:
        scans = scans[:200]
    with open(_scans_path(username), "w") as f:
        json.dump(scans, f, indent=2)

def delete_scan(username: str, scan_index: int):
    scans = load_user_scans(username)
    if 0 <= scan_index < len(scans):
        scans.pop(scan_index)
        with open(_scans_path(username), "w") as f:
            json.dump(scans, f, indent=2)

def clear_all_scans(username: str):
    path = _scans_path(username)
    if os.path.exists(path):
        os.remove(path)


# ─────────────────────────────────────────────
# AUTH CSS  (injected once)
# ─────────────────────────────────────────────

AUTH_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

.auth-wrap {{
    max-width: 420px;
    margin: 3rem auto;
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 16px;
    padding: 2.5rem 2.75rem;
    box-shadow: 0 8px 48px rgba(0,0,0,.45);
}}
.auth-logo {{
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: {P['accent']};
    text-align: center;
    margin-bottom: .2rem;
}}
.auth-sub {{
    color: {P['muted']};
    font-size: .84rem;
    text-align: center;
    margin-bottom: 1.75rem;
}}
.auth-divider {{
    border: none;
    border-top: 1px solid {P['border']};
    margin: 1.25rem 0;
}}

/* Dashboard cards */
.dash-stat {{
    background: {P['surface2']};
    border: 1px solid {P['border']};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.dash-stat-val {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: {P['accent']};
    line-height: 1;
}}
.dash-stat-lbl {{
    color: {P['muted']};
    font-size: .74rem;
    margin-top: .2rem;
}}

.scan-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .6rem .8rem;
    border-radius: 8px;
    background: {P['surface2']};
    border: 1px solid {P['border']};
    margin-bottom: .4rem;
    font-size: .85rem;
}}
.scan-row-left  {{ display: flex; flex-direction: column; gap: .1rem; }}
.scan-row-name  {{ font-weight: 600; color: {P['text']}; }}
.scan-row-meta  {{ color: {P['muted']}; font-size: .74rem; }}
.scan-row-grade {{
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    min-width: 36px;
    text-align: center;
}}

.ns-card-auth {{
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
}}
</style>
"""


# ─────────────────────────────────────────────
# LOGIN / REGISTER UI
# ─────────────────────────────────────────────

def render_auth_page() -> bool:
    """
    Renders the login/register UI.
    Returns True if user is now authenticated (session has 'auth_user').
    """
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    if st.session_state.get("auth_user"):
        return True

    st.markdown("""
    <div style="max-width:420px;margin:3rem auto 1rem;">
        <div class="auth-logo">NutriScan</div>
        <div class="auth-sub">Sign in to track your scans across sessions</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        uname = st.text_input("Username", placeholder="your_username", key="li_user")
        pw    = st.text_input("Password", type="password", placeholder="••••••••", key="li_pw")
        if st.button("Sign In", use_container_width=True, key="li_btn"):
            ok, msg, user_dict = login_user(uname, pw)
            if ok:
                st.session_state["auth_user"]    = uname.strip().lower()
                st.session_state["auth_display"]  = user_dict["display_name"]
                st.session_state["user_profile"]  = user_dict.get("profile", {
                    "weight_kg":0.,"height_cm":0.,"age":0,
                    "sex":"Male","activity":"Moderately Active","goal":"Balanced",
                })
                st.rerun()
            else:
                st.error(msg)

    with tab_reg:
        r_display = st.text_input("Display Name", placeholder="Jane Doe", key="rg_disp")
        r_uname   = st.text_input("Username", placeholder="jane_doe  (letters / numbers / _)", key="rg_user")
        r_pw      = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="rg_pw")
        r_pw2     = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="rg_pw2")
        if st.button("Create Account", use_container_width=True, key="rg_btn"):
            if r_pw != r_pw2:
                st.error("Passwords don't match.")
            else:
                ok, msg = register_user(r_uname, r_pw, r_display)
                if ok:
                    st.success(msg + "  You can now sign in.")
                else:
                    st.error(msg)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue without signing in", use_container_width=True, key="skip_auth"):
        st.session_state["auth_user"]    = None
        st.session_state["auth_display"] = "Guest"
        st.rerun()

    return False


# ─────────────────────────────────────────────
# SIDEBAR AUTH WIDGET
# ─────────────────────────────────────────────

def render_auth_sidebar():
    """
    Call inside render_sidebar() to show user info + logout.
    Returns current username (or None for guest).
    """
    user  = st.session_state.get("auth_user")
    disp  = st.session_state.get("auth_display", "Guest")

    if user:
        st.markdown(f"""
        <div style='padding:.5rem .2rem .3rem;'>
            <div style='font-size:.78rem;color:{P["muted"]};margin-bottom:.1rem;'>Signed in as</div>
            <div style='font-weight:600;font-size:.92rem;color:{P["text"]};'>{disp}</div>
            <div style='font-size:.72rem;color:{P["muted"]};'>@{user}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", key="signout_btn"):
            for k in ["auth_user","auth_display"]:
                st.session_state.pop(k, None)
            st.rerun()
    else:
        st.markdown(f"<div style='color:{P['muted']};font-size:.8rem;padding:.4rem .2rem;'>👤 Guest mode — scans not saved</div>",
                    unsafe_allow_html=True)
        if st.button("Sign In", key="signin_sidebar"):
            # Clear guest flag so auth wall shows again
            st.session_state.pop("auth_user", None)
            st.session_state.pop("auth_display", None)
            st.rerun()

    return user


# ─────────────────────────────────────────────
# SCAN SAVER  (called from app.py after analysis)
# ─────────────────────────────────────────────

def persist_scan(
    username: Optional[str],
    product_name: str,
    category: str,
    score: int,
    grade: str,
    nutrients: Dict,
    brand: Optional[str] = None,
    notes: str = "",
):
    """
    Saves a scan to the user's persistent history.
    No-op for guests (username is None).
    """
    if not username:
        return
    scan = {
        "product":   product_name or "Unknown Product",
        "category":  category,
        "score":     score,
        "grade":     grade,
        "brand":     brand,
        "notes":     notes,
        "nutrients": {k: v.get("value") for k, v in nutrients.items() if isinstance(v, dict)},
    }
    save_user_scan(username, scan)


# ─────────────────────────────────────────────
# DASHBOARD TAB
# ─────────────────────────────────────────────

def render_dashboard():
    """
    Full scan history dashboard.
    Call as a Streamlit tab from app.py.
    """
    import plotly.graph_objects as go
    import pandas as pd

    user = st.session_state.get("auth_user")
    disp = st.session_state.get("auth_display", "Guest")

    if not user:
        st.markdown(f"""
        <div style='text-align:center;padding:4rem 2rem;color:{P["muted"]};'>
            <div style='font-size:3rem;margin-bottom:1rem;'>🔒</div>
            <p style='font-size:1rem;font-weight:600;color:{P["text"]};'>Sign in to access your dashboard</p>
            <p style='font-size:.85rem;'>Your scan history is saved per account across all sessions.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    scans = load_user_scans(user)

    # ── Header ──
    st.markdown(f"""
    <div style='padding:.5rem 0 1.25rem;'>
        <span style='font-family:"Playfair Display",serif;font-size:1.6rem;color:{P["text"]};'>
            Welcome back, <span style='color:{P["accent"]};'>{disp}</span>
        </span>
        <p style='color:{P["muted"]};font-size:.85rem;margin:.25rem 0 0;'>
        Your personalised scan history across all sessions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not scans:
        st.markdown(f"""
        <div style='text-align:center;padding:3rem 1rem;color:{P["muted"]};'>
            <div style='font-size:2.5rem;'>📋</div>
            <p style='margin:.6rem 0;'>No scans saved yet.</p>
            <p style='font-size:.82rem;'>Head to <b style='color:{P["text"]};'>Scan Label</b> to analyse your first product.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Summary stats ──
    total_scans = len(scans)
    avg_score   = round(sum(s["score"] for s in scans) / total_scans, 1)
    best        = max(scans, key=lambda s: s["score"])
    # Grade streak: consecutive scans from most recent with grade A or B
    streak = 0
    for s in scans:
        if s.get("grade","F") in ("A","B"): streak += 1
        else: break
    # Avg grade letter
    if   avg_score >= 85: avg_grade = "A"
    elif avg_score >= 70: avg_grade = "B"
    elif avg_score >= 55: avg_grade = "C"
    elif avg_score >= 40: avg_grade = "D"
    else:                 avg_grade = "F"

    sc = st.columns(4)
    for col, (val, lbl) in zip(sc, [
        (total_scans,               "Total Scans"),
        (f"{avg_grade}  {avg_score}","Avg Score"),
        (f"🔥 {streak}",            "Grade A/B Streak"),
        (f"{best['score']}/100",    f"Best: {best['product'][:14]}"),
    ]):
        col.markdown(f"""
        <div class="dash-stat">
            <div class="dash-stat-val">{val}</div>
            <div class="dash-stat-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──
    cc1, cc2 = st.columns([1, 1], gap="large")

    with cc1:
        st.markdown(f'<div class="ns-card-auth"><div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;color:{P["text"]};margin-bottom:.9rem;">📂 Scans by Category</div>', unsafe_allow_html=True)
        cat_counts: Dict[str, int] = {}
        for s in scans:
            cat = s.get("category","Other")
            # Strip emoji prefix for cleaner labels
            cat_short = cat.split(" ",1)[-1].split("/")[0].strip() if " " in cat else cat
            cat_counts[cat_short] = cat_counts.get(cat_short, 0) + 1

        if cat_counts:
            labels = list(cat_counts.keys())
            vals   = list(cat_counts.values())
            colors = [P["accent"], P["good"], P["warn"], P["bad"],
                      "#6ab3f5","#c084fc","#f472b6","#34d399","#fb923c","#a3e635","#22d3ee"]
            fig = go.Figure(go.Pie(
                labels=labels, values=vals, hole=.55,
                marker={"colors": colors[:len(labels)],
                        "line": {"color": P["surface"], "width": 2}},
                textfont={"color": P["text"], "size": 11},
                hovertemplate="<b>%{label}</b>: %{value} scan(s)<extra></extra>",
            ))
            fig.update_layout(
                height=260, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                legend=dict(font={"color":P["muted"],"size":10},bgcolor="rgba(0,0,0,0)",
                            orientation="v",x=1.02,y=.5,xanchor="left"),
                annotations=[dict(text=f"<b>{total_scans}</b><br>scans",x=.5,y=.5,
                                  font_size=14,showarrow=False,font={"color":P["text"],"family":"Playfair Display"})],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with cc2:
        st.markdown(f'<div class="ns-card-auth"><div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;color:{P["text"]};margin-bottom:.9rem;">📈 Score Trend (last 20)</div>', unsafe_allow_html=True)
        recent = list(reversed(scans[:20]))  # oldest→newest for trend
        if len(recent) >= 2:
            scores = [s["score"] for s in recent]
            labels = [s["product"][:12] for s in recent]
            bar_colors = [P["good"] if sc>=70 else (P["warn"] if sc>=45 else P["bad"]) for sc in scores]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(scores))), y=scores,
                mode="lines+markers",
                line=dict(color=P["accent"], width=2),
                marker=dict(color=bar_colors, size=8, line=dict(color=P["surface"],width=1.5)),
                hovertemplate="<b>%{text}</b><br>Score: %{y}<extra></extra>",
                text=labels,
            ))
            fig2.add_hline(y=70, line_dash="dot", line_color=P["good"], opacity=.4,
                           annotation_text="Good (70)", annotation_font_color=P["good"],
                           annotation_font_size=10)
            fig2.add_hline(y=45, line_dash="dot", line_color=P["warn"], opacity=.4,
                           annotation_text="Avg (45)", annotation_font_color=P["warn"],
                           annotation_font_size=10)
            fig2.update_layout(
                height=260, margin=dict(l=10,r=10,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showticklabels=False, gridcolor=P["border"], zeroline=False),
                yaxis=dict(range=[0,105], gridcolor=P["border"], color=P["muted"]),
                font={"color":P["text"]}, showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Scan at least 2 products to see a trend.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Category filter + full scan log ──
    st.markdown(f'<div class="ns-card-auth">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-family:\'Playfair Display\',serif;font-size:1.1rem;color:{P["text"]};margin-bottom:.9rem;">🗂️ Scan Log</div>', unsafe_allow_html=True)

    # Filter controls
    fc1, fc2, fc3 = st.columns([2,2,1])
    all_cats  = sorted(set(s.get("category","") for s in scans))
    grade_opts = ["All","A","B","C","D","F"]
    with fc1:
        sel_cat_f = st.selectbox("Filter by Category", ["All"] + all_cats, key="dash_cat_f")
    with fc2:
        sel_grade = st.selectbox("Filter by Grade", grade_opts, key="dash_grade_f")
    with fc3:
        sort_by = st.selectbox("Sort", ["Newest","Score ↓","Score ↑"], key="dash_sort")

    # Apply filters
    filtered = scans[:]
    if sel_cat_f != "All":
        filtered = [s for s in filtered if s.get("category","") == sel_cat_f]
    if sel_grade != "All":
        filtered = [s for s in filtered if s.get("grade","") == sel_grade]
    if sort_by == "Score ↓":
        filtered = sorted(filtered, key=lambda s: -s["score"])
    elif sort_by == "Score ↑":
        filtered = sorted(filtered, key=lambda s: s["score"])

    if not filtered:
        st.info("No scans match this filter.")
    else:
        # Paginate: 15 per page
        PAGE_SIZE = 15
        total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
        if "dash_page" not in st.session_state:
            st.session_state.dash_page = 0
        pg = st.session_state.dash_page
        page_scans = filtered[pg*PAGE_SIZE : (pg+1)*PAGE_SIZE]

        for i, s in enumerate(page_scans):
            real_idx = scans.index(s) if s in scans else -1
            gc = P["good"] if s["score"]>=70 else (P["warn"] if s["score"]>=45 else P["bad"])
            cal_str = f'{s["nutrients"].get("calories","—")} kcal' if isinstance(s.get("nutrients",{}).get("calories"),float) else "—"

            col_main, col_del = st.columns([10,1])
            with col_main:
                _muted   = P["muted"]
                _note    = s.get("notes", "")[:40]
                _cat     = s.get("category","")[:22]
                _brand   = ("· " + s["brand"]) if s.get("brand") else ""
                _date    = s.get("saved_at","")[:10]
                _grade   = s.get("grade","?")
                _notes_s = f"<span style='color:{_muted};font-style:italic;'> · {_note}</span>" if _note else ""
                _html    = (
                    f"<div class='scan-row'>"
                    f"<div class='scan-row-left'>"
                    f"<span class='scan-row-name'>{s['product']}</span>"
                    f"<span class='scan-row-meta'>{_cat} {_brand} · {_date} · {cal_str}{_notes_s}</span>"
                    f"</div>"
                    f"<span class='scan-row-grade' style='color:{gc};'>{_grade}"
                    f"<span style='display:block;font-size:.7rem;font-weight:400;'>{s['score']}/100</span>"
                    f"</span></div>"
                )
                st.markdown(_html, unsafe_allow_html=True)
            with col_del:
                if real_idx >= 0 and st.button("🗑", key=f"del_{i}_{pg}", help="Delete this scan"):
                    delete_scan(user, real_idx)
                    st.rerun()

        # Pagination controls
        p1, p2, p3 = st.columns([2,3,2])
        with p1:
            if pg > 0 and st.button("← Previous", key="pg_prev"):
                st.session_state.dash_page -= 1; st.rerun()
        with p2:
            st.markdown(f'<p style="text-align:center;color:{P["muted"]};font-size:.8rem;margin:.5rem 0;">Page {pg+1} of {total_pages} · {len(filtered)} scan(s)</p>', unsafe_allow_html=True)
        with p3:
            if pg < total_pages-1 and st.button("Next →", key="pg_next"):
                st.session_state.dash_page += 1; st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Danger zone ──
    with st.expander("⚠️ Danger Zone"):
        st.markdown(f"<p style='color:{P['muted']};font-size:.83rem;'>This will permanently delete all your saved scans. This cannot be undone.</p>", unsafe_allow_html=True)
        if st.button("🗑 Delete All My Scans", key="nuke_scans"):
            clear_all_scans(user)
            st.success("All scans deleted.")
            st.rerun()