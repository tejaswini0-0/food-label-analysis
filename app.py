"""
NutriScan — Intelligent Nutrition Label Analyzer  v1.2
=======================================================
Changes in v1.1:
  - Redesigned health scoring: 5-dimension system (Sugar/Sodium/Fat Quality/Protein/Fiber),
    each 0-20, summed to 100. Category-aware weights. Letter grade A-F.
  - Macronutrient donut: always shows all 3 macros (protein, carbs, fat). Shows 0 if missing.
  - Colour palette: warm amber / sage / coral — no neon green
  - BMI gauge centred correctly with proper domain config
  - Manual fallback form: appears automatically on low/medium OCR confidence
  - Food category selector (chips, drink, ice cream, namkeen, etc.) before analysis
  - Phase 2 stub comments for cloud DB (brand + label name storage)
"""

import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import re
import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import cv2

# Auth module (local file-based; swap internals for Supabase in Phase 2)
from auth import (
    render_auth_page, render_auth_sidebar,
    persist_scan, render_dashboard,
)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NutriScan",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# PALETTE & GLOBAL CSS
# Warm editorial palette: amber / sage green / coral
# ─────────────────────────────────────────────
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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
:root{{
  --bg:{P['bg']};--surface:{P['surface']};--surface2:{P['surface2']};
  --border:{P['border']};--accent:{P['accent']};--good:{P['good']};
  --warn:{P['warn']};--bad:{P['bad']};--text:{P['text']};--muted:{P['muted']};
  --r:12px;
}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{background:var(--surface)!important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] *{{color:var(--text)!important;}}

.ns-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:1.5rem 1.75rem;margin-bottom:1.25rem;}}
.card-amber{{background:linear-gradient(135deg,rgba(240,165,0,.07),rgba(240,165,0,.02));border:1px solid rgba(240,165,0,.22);border-radius:var(--r);padding:1.3rem 1.5rem;margin-bottom:1rem;}}
.card-good{{background:linear-gradient(135deg,rgba(92,184,138,.07),rgba(92,184,138,.02));border:1px solid rgba(92,184,138,.22);border-radius:var(--r);padding:1.3rem 1.5rem;margin-bottom:1rem;}}
.card-warn{{background:linear-gradient(135deg,rgba(232,136,74,.07),rgba(232,136,74,.02));border:1px solid rgba(232,136,74,.22);border-radius:var(--r);padding:1.3rem 1.5rem;margin-bottom:1rem;}}
.card-bad{{background:linear-gradient(135deg,rgba(217,85,85,.07),rgba(217,85,85,.02));border:1px solid rgba(217,85,85,.22);border-radius:var(--r);padding:1.3rem 1.5rem;margin-bottom:1rem;}}

.ns-hero{{text-align:center;padding:1.75rem 1rem 1.25rem;}}
.ns-hero h1{{font-family:'Playfair Display',serif;font-size:2.75rem;letter-spacing:-.01em;color:var(--text);margin:0 0 .3rem;line-height:1.1;}}
.ns-hero h1 span{{color:var(--accent);}}
.ns-hero p{{color:var(--muted);font-size:.98rem;margin:0;}}

.sec-hdr{{font-family:'Playfair Display',serif;font-size:1.2rem;color:var(--text);margin:0 0 .9rem;display:flex;align-items:center;gap:.4rem;}}
.nut-row{{display:flex;justify-content:space-between;align-items:center;padding:.48rem 0;border-bottom:1px solid var(--border);font-size:.9rem;}}
.nut-row:last-child{{border-bottom:none;}}
.nut-name{{color:var(--muted);}}
.nut-val{{font-weight:600;color:var(--text);}}

.tag{{display:inline-block;padding:.17rem .58rem;border-radius:999px;font-size:.74rem;font-weight:600;margin:.15rem .1rem;}}
.tg{{background:rgba(92,184,138,.12);color:var(--good);border:1px solid rgba(92,184,138,.28);}}
.tw{{background:rgba(232,136,74,.10);color:var(--warn);border:1px solid rgba(232,136,74,.28);}}
.tb{{background:rgba(217,85,85,.10);color:var(--bad);border:1px solid rgba(217,85,85,.28);}}
.tn{{background:rgba(102,107,138,.10);color:var(--muted);border:1px solid rgba(102,107,138,.28);}}
.ta{{background:rgba(240,165,0,.10);color:var(--accent);border:1px solid rgba(240,165,0,.28);}}

.grade-ring{{display:block;width:68px;height:68px;border-radius:50%;line-height:68px;text-align:center;font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;margin:0 auto .5rem;}}

div.stButton>button{{background:var(--accent);color:#0d0e14;border:none;border-radius:8px;font-weight:600;font-size:.92rem;padding:.58rem 1.4rem;width:100%;transition:opacity .18s;}}
div.stButton>button:hover{{opacity:.82;}}
div[data-testid="metric-container"]{{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:.7rem .95rem;}}
div[data-testid="metric-container"] label{{color:var(--muted)!important;font-size:.77rem!important;}}
div[data-testid="stAlert"]{{background:var(--surface2)!important;border-radius:10px!important;border-left-width:3px!important;}}
hr{{border-color:var(--border);margin:1.25rem 0;}}

.fallback-box{{background:linear-gradient(135deg,rgba(232,136,74,.09),rgba(217,85,85,.04));border:1px solid rgba(232,136,74,.32);border-radius:var(--r);padding:1.1rem 1.4rem;margin-bottom:1rem;}}

/* Hide sidebar and its toggle entirely */
[data-testid="stSidebar"]{{display:none!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}

/* Top control bar */
.top-bar{{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:.75rem 1.25rem;
  margin-bottom:1rem;
  display:flex;
  align-items:center;
  gap:1rem;
}}
.top-bar-logo{{
  font-family:'Playfair Display',serif;
  font-size:1.3rem;
  color:var(--accent);
  white-space:nowrap;
  font-weight:700;
}}
.history-pill{{
  display:flex;align-items:center;gap:.5rem;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;padding:.3rem .7rem;font-size:.78rem;
  cursor:default;
}}
.history-pill .grade{{font-weight:700;font-size:.9rem;}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DRV = {
    "calories": 2000, "total_fat": 78, "saturated_fat": 20,
    "carbohydrates": 275, "fiber": 28, "sugar": 50,
    "protein": 50, "sodium": 2300, "cholesterol": 300,
}

GOAL_PROFILES = {
    "Balanced":         {"protein": .20, "carbs": .50, "fat": .30},
    "High Protein":     {"protein": .35, "carbs": .40, "fat": .25},
    "Low Carb / Keto":  {"protein": .30, "carbs": .05, "fat": .65},
    "Weight Loss":      {"protein": .30, "carbs": .40, "fat": .30},
    "Endurance / Carb": {"protein": .15, "carbs": .60, "fat": .25},
}

BMI_CATS = [
    (0,    18.5, "Underweight",      "#6ab3f5"),
    (18.5, 25.0, "Healthy Weight",   "#5cb88a"),
    (25.0, 30.0, "Overweight",       "#e8884a"),
    (30.0, 35.0, "Obese (Class I)",  "#d95555"),
    (35.0, 999,  "Obese (Class II+)","#b03030"),
]

# Category config: concern = penalised 1.5×, benefit = rewarded 1.3×
# cal_bench / sodium_bench = typical per-serving for that food type
FOOD_CATS = {
    "🍟 Chips / Crisps":    {"id":"chips",   "concern":["total_fat","saturated_fat","sodium"],  "benefit":["fiber","protein"],  "cal_bench":150,"sodium_bench":200},
    "🥤 Drink / Beverage":  {"id":"drink",   "concern":["sugar","calories"],                    "benefit":["protein"],          "cal_bench":120,"sodium_bench":50},
    "🍦 Ice Cream":         {"id":"icecream","concern":["sugar","saturated_fat","calories"],    "benefit":["protein","calcium"], "cal_bench":200,"sodium_bench":80},
    "🧀 Cheese / Dairy":    {"id":"cheese",  "concern":["saturated_fat","sodium","calories"],   "benefit":["protein","calcium"], "cal_bench":100,"sodium_bench":200},
    "🥜 Namkeen / Snack":   {"id":"namkeen", "concern":["sodium","total_fat","calories"],       "benefit":["protein","fiber"],   "cal_bench":130,"sodium_bench":300},
    "🍫 Chocolate / Candy": {"id":"choc",    "concern":["sugar","saturated_fat","calories"],    "benefit":["fiber","protein"],   "cal_bench":150,"sodium_bench":50},
    "🥣 Cereal / Granola":  {"id":"cereal",  "concern":["sugar","sodium"],                     "benefit":["fiber","protein"],   "cal_bench":200,"sodium_bench":150},
    "🍞 Bread / Bakery":    {"id":"bread",   "concern":["sodium","sugar","total_fat"],          "benefit":["fiber","protein"],   "cal_bench":130,"sodium_bench":200},
    "🍝 Ready Meal":        {"id":"meal",    "concern":["sodium","saturated_fat","calories"],   "benefit":["protein","fiber"],   "cal_bench":400,"sodium_bench":600},
    "💊 Supplement / Bar":  {"id":"supp",    "concern":["sugar","sodium"],                     "benefit":["protein","fiber"],   "cal_bench":200,"sodium_bench":100},
    "🏷️ Other / Unknown":   {"id":"other",   "concern":["sodium","sugar","saturated_fat"],      "benefit":["protein","fiber"],   "cal_bench":200,"sodium_bench":200},
}

MACRO_C = {"Protein": P["accent"], "Carbohydrates": "#6ab3f5", "Fat": P["bad"]}


# ─────────────────────────────────────────────
# OCR & PREPROCESSING
# ─────────────────────────────────────────────
# ROOT CAUSE for coloured labels (like this FDA label):
#   Rows like "Total Carbohydrate 37g" sit on a bright PINK background,
#   "Dietary Fiber 4g" on PURPLE, "Total Sugars 12g" on GREEN, etc.
#   Standard adaptive thresholding converts these coloured rows to dark
#   blobs that erase the text entirely.
#
# SOLUTION — 6 strategies tried in parallel, best result wins:
#   1. hsv_value    — extract V (brightness) from HSV; coloured backgrounds
#                     are bright, dark text stays dark regardless of hue.
#                     ★ BEST for FDA-style coloured nutrition labels
#   2. lab_lum      — CLAHE on LAB L-channel + Otsu; good for tinted labels
#   3. adaptive     — classic CLAHE + adaptive threshold; uneven lighting
#   4. otsu         — Gaussian blur + Otsu; clean high-contrast labels
#   5. invert       — inverted adaptive; white-on-dark backgrounds
#   6. raw          — no preprocessing; always runs as final safety net
#
# GROQ VISION FALLBACK (if EasyOCR confidence is still low):
#   Sends the image as base64 to Groq's free llama-4-scout-17b-16e-instruct
#   vision model, asking it to return structured JSON of all nutrients.
#   Requires GROQ_API_KEY in Streamlit secrets or environment variables.
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_reader():
    return easyocr.Reader(["en"], gpu=False)


# ── Upscale helper ────────────────────────────────────────────────────────────
def _upscale_rgb(arr: np.ndarray, min_dim: int = 1400) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) < min_dim:
        s   = min_dim / max(h, w)
        arr = cv2.resize(arr, (int(w*s), int(h*s)), interpolation=cv2.INTER_LANCZOS4)
    return arr


# ── Strategy 1: HSV Value channel (★ best for coloured backgrounds) ───────────
def _strategy_hsv_value(arr: np.ndarray) -> np.ndarray:
    """
    Converts to HSV and uses the Value channel only.
    Coloured backgrounds (pink, purple, green, orange) all have HIGH Value
    (they are bright). Dark text has LOW Value regardless of hue.
    After CLAHE + Otsu, text is crisp black on white whatever the band colour.
    """
    hsv  = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v    = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_eq  = clahe.apply(v)
    v_eq  = cv2.fastNlMeansDenoising(v_eq, h=8)
    _, binary = cv2.threshold(v_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ── Strategy 2: LAB luminance ─────────────────────────────────────────────────
def _strategy_lab_lum(arr: np.ndarray) -> np.ndarray:
    """
    CLAHE on the L channel of LAB colour space, then Otsu.
    Works well when backgrounds have moderate saturation / tinted paper.
    """
    lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab2  = cv2.merge([l_eq, a, b])
    gray  = cv2.cvtColor(cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2GRAY)
    gray  = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray  = cv2.fastNlMeansDenoising(gray, h=8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ── Strategy 3: Adaptive threshold ────────────────────────────────────────────
def _strategy_adaptive(arr: np.ndarray) -> np.ndarray:
    """CLAHE + denoise + adaptive threshold. Best for uneven lighting."""
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    eq    = cv2.fastNlMeansDenoising(eq, h=12)
    binary = cv2.adaptiveThreshold(eq, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 29, 9)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ── Strategy 4: Otsu on grayscale ─────────────────────────────────────────────
def _strategy_otsu(arr: np.ndarray) -> np.ndarray:
    """Gaussian blur + Otsu. Best for clean, high-contrast labels."""
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ── Strategy 5: Invert + adaptive ─────────────────────────────────────────────
def _strategy_invert(arr: np.ndarray) -> np.ndarray:
    """Inverted adaptive. For white-on-dark / light-on-dark labels."""
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    inv   = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq    = clahe.apply(inv)
    binary = cv2.adaptiveThreshold(eq, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 29, 9)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


# ── OCR scorer ────────────────────────────────────────────────────────────────
def _ocr_score(results: List[Tuple]) -> float:
    """
    Score = (total confident chars) × avg_confidence.
    Bonus multiplier if key nutrition words are present — this biases
    selection toward results that actually extracted the right content.
    """
    if not results:
        return 0.
    total_chars = sum(len(t) for _, t, _ in results)
    avg_conf    = sum(c for _, _, c in results) / len(results)
    base        = total_chars * avg_conf
    # Keyword bonus: each of these found adds 15% to score
    all_text = " ".join(t.lower() for _, t, _ in results)
    keywords = ["sodium","carbohydrate","protein","calorie","fat","fiber","sugar"]
    bonus    = sum(0.15 for kw in keywords if kw in all_text)
    return base * (1 + bonus)


# ── Main OCR runner ───────────────────────────────────────────────────────────
# Keywords that indicate a good nutrition label parse
_GOOD_KW = ["sodium","carbohydrate","protein","calorie","fat","fiber","sugar"]
_GOOD_KW_THRESHOLD = 4   # if this many keywords found → accept immediately

def run_ocr(pil_img: Image.Image) -> List[Tuple]:
    """
    Speed-optimised multi-strategy OCR.
    Strategy: run hsv_value first (best for coloured labels).
    If it finds ≥4 nutrition keywords, accept immediately (fast path).
    Otherwise try up to 2 more strategies and pick the best.
    Greedy decoder used throughout (3-5× faster than beamsearch).
    """
    reader  = load_reader()
    arr     = _upscale_rgb(np.array(pil_img.convert("RGB")))

    def _run(name, fn):
        try:
            processed = fn(arr)
            results   = reader.readtext(processed, detail=1, paragraph=False)
            return name, results, _ocr_score(results)
        except Exception:
            return name, [], 0.

    best_results: List[Tuple] = []
    best_score   = 0.
    best_name    = "none"

    def _update(name, results, sc):
        nonlocal best_results, best_score, best_name
        if sc > best_score:
            best_score, best_results, best_name = sc, results, name

    def _is_good_enough(results):
        txt = " ".join(t.lower() for _, t, _ in results)
        return sum(1 for kw in _GOOD_KW if kw in txt) >= _GOOD_KW_THRESHOLD

    # ── Fast path: try HSV first ──────────────────────────────────────────────
    name, results, sc = _run("hsv_value", _strategy_hsv_value)
    _update(name, results, sc)
    if _is_good_enough(results):
        st.session_state["ocr_strategy"] = best_name
        st.session_state["ocr_score"]    = round(best_score, 1)
        return best_results

    # ── Slow path: try 2 more strategies ─────────────────────────────────────
    for name, fn in [("lab_lum", _strategy_lab_lum), ("otsu", _strategy_otsu)]:
        n, r, sc = _run(name, fn)
        _update(n, r, sc)
        if _is_good_enough(r):
            break

    # ── Final fallback: raw image ─────────────────────────────────────────────
    if not best_results:
        n, r, sc = _run("raw", lambda a: a)
        _update(n, r, sc)

    st.session_state["ocr_strategy"] = best_name
    st.session_state["ocr_score"]    = round(best_score, 1)
    return best_results


# ─────────────────────────────────────────────
# GROQ VISION FALLBACK
# ─────────────────────────────────────────────
# Uses Groq's free llama-4-scout-17b-16e-instruct vision model to extract
# all nutrition values from the image when EasyOCR confidence is low.
#
# Setup (one-time):
#   Option A — Streamlit secrets:  add  GROQ_API_KEY = "gsk_..."  to
#              .streamlit/secrets.toml  or the Streamlit Cloud dashboard.
#   Option B — Environment variable:  export GROQ_API_KEY="gsk_..."
#   Get a free key at: https://console.groq.com  (no credit card needed)
# ─────────────────────────────────────────────

_GROQ_SYSTEM = """You are a nutrition label parser.
The user will send an image of a nutrition facts label.
Extract EVERY nutrient you can read and return ONLY a valid JSON object.
No markdown, no explanation, no code fences — just the raw JSON.

Required JSON format:
{
  "calories": <number or null>,
  "total_fat": <number or null>,
  "saturated_fat": <number or null>,
  "trans_fat": <number or null>,
  "cholesterol": <number or null>,
  "sodium": <number or null>,
  "carbohydrates": <number or null>,
  "fiber": <number or null>,
  "sugar": <number or null>,
  "added_sugar": <number or null>,
  "protein": <number or null>,
  "vitamin_d": <number or null>,
  "calcium": <number or null>,
  "iron": <number or null>,
  "potassium": <number or null>,
  "serving_size_g": <number or null>,
  "servings_per_container": <number or null>,
  "product_name": "<string or null>"
}
Use null for any nutrient not visible on the label.
All fat/carb/fiber/sugar/protein values must be in grams (g).
Cholesterol, sodium, calcium, iron, potassium in milligrams (mg).
Vitamin D in micrograms (mcg). Calories in kcal."""


def _get_groq_key() -> Optional[str]:
    """Try Streamlit secrets first, then environment variable."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key: return key
    except Exception:
        pass
    import os
    return os.environ.get("GROQ_API_KEY")


def groq_extract(pil_img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Send image to Groq vision API, parse the JSON response.
    Returns a nutrient dict on success, None on failure.
    Free tier: 30 requests/minute, 14,400/day — plenty for a nutrition app.
    """
    import base64, json, urllib.request, urllib.error

    api_key = _get_groq_key()
    if not api_key:
        return None

    # Encode image as base64
    try:
        import io
        # Resize to max 1024px on longest side to stay within Groq's token limits
        img = pil_img.copy()
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": _GROQ_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": "Extract all nutrition facts from this label."},
                ],
            },
        ],
    }

    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
        raw = body["choices"][0]["message"]["content"].strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return None


def groq_to_parsed_data(groq_json: Dict, original_data: Dict) -> Dict:
    """
    Convert Groq's flat JSON into the app's internal nutrient dict format,
    only filling in fields that EasyOCR missed.
    """
    unit_map = {
        "calories": "kcal", "total_fat": "g", "saturated_fat": "g",
        "trans_fat": "g", "cholesterol": "mg", "sodium": "mg",
        "carbohydrates": "g", "fiber": "g", "sugar": "g",
        "added_sugar": "g", "protein": "g", "vitamin_d": "mcg",
        "calcium": "mg", "iron": "mg", "potassium": "mg",
        "serving_size": "g",
    }
    micro = {"vitamin_d", "calcium", "iron", "potassium"}

    for key, unit in unit_map.items():
        val = groq_json.get(key) or groq_json.get(f"{key}_g") or groq_json.get(f"{key}_mg")
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if val < 0 or val >= 1e5:
            continue
        bucket  = "micronutrients" if key in micro else "nutrients"
        # Only fill gaps — don't overwrite EasyOCR finds
        if key not in original_data[bucket]:
            original_data[bucket][key] = {"value": val, "unit": unit, "source": "groq"}

    # serving_size special case
    ssv = groq_json.get("serving_size_g")
    if ssv and "serving_size" not in original_data["nutrients"]:
        try:
            original_data["nutrients"]["serving_size"] = {"value": float(ssv), "unit": "g", "source": "groq"}
        except Exception:
            pass

    if groq_json.get("product_name") and not original_data.get("product_name"):
        original_data["product_name"] = str(groq_json["product_name"]).title()

    # Re-evaluate confidence after Groq fill
    core  = {"calories", "protein", "carbohydrates", "total_fat", "sodium"}
    found = core & original_data["nutrients"].keys()
    original_data["missing_nutrients"] = list(core - found)
    original_data["parse_confidence"]  = (
        "high"   if len(found) >= 4 else
        "medium" if len(found) >= 2 else "low"
    )
    return original_data


# ─────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────
_FIX = str.maketrans({"O":"0","o":"0","l":"1","I":"1","|":"1","S":"5","B":"8","Z":"2"})

def ocr_num(t: str) -> Optional[float]:
    c = re.sub(r"[^0-9.]", "", t.translate(_FIX))
    p = c.split(".")
    if len(p) > 2: c = p[0] + "." + "".join(p[1:])
    try:
        v = float(c)
        return v if 0 <= v < 1e5 else None
    except ValueError:
        return None

def extr(text: str, pats: List[str]) -> Tuple[Optional[float], str]:
    for pat in pats:
        m = re.search(pat, text, re.I | re.M)
        if m:
            gd = m.groupdict()
            val = ocr_num(gd.get("val",""))
            unit = (gd.get("unit") or "").strip().lower()
            if val is not None:
                if not unit:
                    unit = "mg" if "mg" in pat.lower() else "g"
                return val, unit
    return None, ""

NPATS: Dict[str, List[str]] = {
    "serving_size":           [r"serving\s*size[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|ml|oz)",
                                r"serving\s*size[\s:]+\d+\s+\w+\s*\((?P<val>[\d.]+)\s*(?P<unit>g|ml)\)"],
    "servings_per_container": [r"(?:about\s*)?(?P<val>[\d.]+)\s*servings?\s*per\s*container",
                                r"servings?\s*per\s*container[\s:]+(?P<val>[\d.]+)"],
    "calories":               [r"(?<!\w)calories[\s:]+(?P<val>\d{2,4})(?!\d)",
                                r"energy[\s:]+(?P<val>\d{2,4})\s*(?:kcal|cal)",
                                r"^(?P<val>\d{2,4})\s*(?:calories|cal)$"],
    "total_fat":              [r"total\s*fat[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"fat,?\s*total[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "saturated_fat":          [r"saturated\s*fat[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"sat\.?\s*fat[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "trans_fat":              [r"trans\s*fat[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "cholesterol":            [r"cholesterol[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g)"],
    "sodium":                 [r"sodium[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g)",
                                r"salt[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g)"],
    "carbohydrates":          [r"total\s*carbohydrate[s]?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"carbohydrate[s]?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"(?:^|\n|\s)carbs?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "fiber":                  [r"dietary\s*fiber[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"(?:total\s*)?fi(?:ber|bre)[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "sugar":                  [r"total\s*sugars?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)",
                                r"sugars?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "added_sugar":            [r"added\s*sugars?[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "protein":                [r"protein[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|mg)"],
    "vitamin_d":              [r"vitamin\s*d[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mcg|µg|iu|mg|%)"],
    "calcium":                [r"calcium[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g|%)"],
    "iron":                   [r"iron[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g|%)"],
    "potassium":              [r"potassium[\s:]+(?P<val>[\d.]+)\s*(?P<unit>mg|g|%)"],
}

DV_PATS = {
    "total_fat":     r"total\s*fat[^%\n]{0,25}?(\d+)\s*%",
    "saturated_fat": r"saturated\s*fat[^%\n]{0,25}?(\d+)\s*%",
    "cholesterol":   r"cholesterol[^%\n]{0,25}?(\d+)\s*%",
    "sodium":        r"sodium[^%\n]{0,25}?(\d+)\s*%",
    "carbohydrates": r"total\s*carbohydrate[^%\n]{0,25}?(\d+)\s*%",
    "fiber":         r"(?:dietary\s*)?fiber[^%\n]{0,25}?(\d+)\s*%",
    "protein":       r"protein[^%\n]{0,25}?(\d+)\s*%",
    "vitamin_d":     r"vitamin\s*d[^%\n]{0,25}?(\d+)\s*%",
    "calcium":       r"calcium[^%\n]{0,25}?(\d+)\s*%",
    "iron":          r"iron[^%\n]{0,25}?(\d+)\s*%",
    "potassium":     r"potassium[^%\n]{0,25}?(\d+)\s*%",
}
MICRO_KEYS = {"vitamin_d","calcium","iron","potassium"}

def _bbox_center_y(bbox) -> float:
    """Return vertical center of an OCR bounding box."""
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def _bbox_center_x(bbox) -> float:
    xs = [pt[0] for pt in bbox]
    return (min(xs) + max(xs)) / 2.0


def _group_rows(sorted_res: List[Tuple], row_tol_ratio: float = 0.012) -> List[List[Tuple]]:
    """
    Cluster OCR tokens into horizontal rows based on Y proximity.
    row_tol_ratio: fraction of image height used as row-gap tolerance.
    Returns list of rows, each row is a list of (bbox, text, conf) sorted by X.
    """
    if not sorted_res:
        return []
    img_height = max(_bbox_center_y(b) for b, t, c in sorted_res) or 1
    tol = max(8, img_height * row_tol_ratio)

    rows: List[List[Tuple]] = []
    current_row: List[Tuple] = [sorted_res[0]]
    current_y = _bbox_center_y(sorted_res[0][0])

    for item in sorted_res[1:]:
        cy = _bbox_center_y(item[0])
        if abs(cy - current_y) <= tol:
            current_row.append(item)
        else:
            rows.append(sorted(current_row, key=lambda x: _bbox_center_x(x[0])))
            current_row = [item]
            current_y   = cy
    if current_row:
        rows.append(sorted(current_row, key=lambda x: _bbox_center_x(x[0])))
    return rows


# ── Keyword → nutrient key map for positional matching ──
_KW_MAP = {
    "energy":         "calories",
    "calories":       "calories",
    "calorie":        "calories",
    "kcal":           "calories",
    "total fat":      "total_fat",
    "fat":            "total_fat",
    "saturated fat":  "saturated_fat",
    "sat. fat":       "saturated_fat",
    "sat fat":        "saturated_fat",
    "trans fat":      "trans_fat",
    "trans":          "trans_fat",
    "cholesterol":    "cholesterol",
    "sodium":         "sodium",
    "salt":           "sodium",
    "carbohydrate":   "carbohydrates",
    "carbohydrates":  "carbohydrates",
    "total carbohydrate": "carbohydrates",
    "carbs":          "carbohydrates",
    "dietary fibre":  "fiber",
    "dietary fiber":  "fiber",
    "fibre":          "fiber",
    "fiber":          "fiber",
    "total sugars":   "sugar",
    "sugars":         "sugar",
    "sugar":          "sugar",
    "added sugars":   "added_sugar",
    "protein":        "protein",
    "vitamin d":      "vitamin_d",
    "calcium":        "calcium",
    "iron":           "iron",
    "potassium":      "potassium",
}

_UNIT_MAP = {
    "calories": "kcal", "total_fat": "g", "saturated_fat": "g", "trans_fat": "g",
    "cholesterol": "mg", "sodium": "mg", "carbohydrates": "g", "fiber": "g",
    "sugar": "g", "added_sugar": "g", "protein": "g",
    "vitamin_d": "mcg", "calcium": "mg", "iron": "mg", "potassium": "mg",
}


def _positional_parse(rows: List[List[Tuple]], out: Dict) -> Dict:
    """
    Positional row-pair parser.

    Many Indian/Asian nutrition tables use a two-column layout:
      | Nutrient Name | Value per 100g | Value per serving |
    or a single-value layout:
      | Total Carbohydrates | 24.5g |

    Strategy:
      For each row, check if any token matches a known nutrient keyword.
      If yes, look for a numeric token in the SAME row (same-row value)
      or in the NEXT row (below-value layout common in Indian packs).
    """

    def find_nutrient_in_row(row: List[Tuple]) -> Optional[str]:
        row_text = " ".join(t.lower() for _, t, _ in row)
        # Longest-match first
        for kw in sorted(_KW_MAP.keys(), key=len, reverse=True):
            if kw in row_text:
                return _KW_MAP[kw]
        return None

    def find_numeric_in_row(row: List[Tuple]) -> Tuple[Optional[float], str]:
        """Return first plausible (value, unit) in row."""
        row_text = " ".join(t for _, t, _ in row)
        # Pattern: number optionally followed by unit
        m = re.search(r"(?<!\d)([\d]+(?:[.,]\d+)?)\s*(g|mg|mcg|µg|kcal|cal|%)?(?!\d)",
                      row_text, re.I)
        if m:
            raw = m.group(1).replace(",",".")
            v   = ocr_num(raw)
            u   = (m.group(2) or "").lower().strip()
            return v, u
        return None, ""

    n_rows = len(rows)
    for i, row in enumerate(rows):
        nut_key = find_nutrient_in_row(row)
        if nut_key is None:
            continue
        # Skip if already parsed with higher confidence
        bucket = "micronutrients" if nut_key in MICRO_KEYS else "nutrients"
        if nut_key in out[bucket]:
            continue

        # 1. Try same row first (most common for inline tables)
        val, unit = find_numeric_in_row(row)

        # 2. If not found, try the next row (value-below layout)
        if val is None and i + 1 < n_rows:
            val, unit = find_numeric_in_row(rows[i + 1])

        # 3. If still not found, try combining this row + next as one text
        if val is None and i + 1 < n_rows:
            combined_text = " ".join(t for _, t, _ in (row + rows[i+1]))
            m2 = re.search(r"(?<!\d)([\d]+(?:[.,]\d+)?)\s*(g|mg|mcg|µg|kcal|cal|%)?(?!\d)",
                           combined_text, re.I)
            if m2:
                val = ocr_num(m2.group(1).replace(",","."))
                unit = (m2.group(2) or "").lower().strip()

        if val is not None and val < 1e5:
            # Infer unit if not captured
            if not unit:
                unit = _UNIT_MAP.get(nut_key, "g")
            out[bucket][nut_key] = {"value": val, "unit": unit, "source": "positional"}

    return out


def parse_label(ocr_res: List[Tuple]) -> Dict[str, Any]:
    """
    Three-pass parser:
      Pass 1 — Regex on full text (existing, fast)
      Pass 2 — Positional row-pair matching (new; fixes sodium/carbs on tabular labels)
      Pass 3 — Regex on single joined line (line-wrap OCR fallback)
    Results from earlier passes are kept; later passes only fill gaps.
    """
    sorted_res = sorted(ocr_res, key=lambda r: _bbox_center_y(r[0]))
    texts, hi_conf = [], []
    for bbox, text, conf in sorted_res:
        texts.append(text)
        if conf > 0.4:
            hi_conf.append(text)

    full   = "\n".join(texts)
    full_l = full.lower()
    join_l = " ".join(texts).lower()

    out: Dict[str, Any] = {
        "raw_text": full, "nutrients": {}, "daily_values": {},
        "micronutrients": {}, "product_name": None, "brand": None,
        "parse_confidence": "low", "missing_nutrients": [],
    }

    # ── Pass 1: regex on newline-structured text ──
    for key, pats in NPATS.items():
        val, unit = extr(full_l, pats)
        if val is not None:
            bucket = "micronutrients" if key in MICRO_KEYS else "nutrients"
            out[bucket][key] = {"value": val, "unit": unit, "source": "regex"}

    # ── Pass 2: positional row-pair matching ──
    rows = _group_rows(sorted_res)
    out  = _positional_parse(rows, out)

    # ── Pass 3: regex on single joined line (line-wrap fallback) ──
    for key, pats in NPATS.items():
        bucket = "micronutrients" if key in MICRO_KEYS else "nutrients"
        if key not in out[bucket]:
            val, unit = extr(join_l, pats)
            if val is not None:
                out[bucket][key] = {"value": val, "unit": unit, "source": "joined"}

    # ── Daily values ──
    for key, pat in DV_PATS.items():
        m = re.search(pat, full_l, re.I)
        if m:
            dv = ocr_num(m.group(1))
            if dv is not None:
                out["daily_values"][key] = dv

    # ── Product name ──
    nf = full_l.find("nutrition facts")
    if nf > 0:
        lines = [l.strip() for l in full[:nf].split("\n") if l.strip() and len(l.strip()) > 2]
        if lines: out["product_name"] = lines[-1].title()
    elif hi_conf:
        for t in hi_conf[:5]:
            if not re.match(r"^\d", t) and len(t) > 3:
                out["product_name"] = t.strip().title(); break

    # ── Confidence ──
    core  = {"calories","protein","carbohydrates","total_fat","sodium"}
    found = core & out["nutrients"].keys()
    out["missing_nutrients"] = list(core - found)
    out["parse_confidence"]  = "high" if len(found) >= 4 else ("medium" if len(found) >= 2 else "low")
    return out

def apply_overrides(data: Dict, ov: Dict[str, Optional[float]]) -> Dict:
    umap = {"calories":"kcal","total_fat":"g","saturated_fat":"g","trans_fat":"g",
            "carbohydrates":"g","fiber":"g","sugar":"g","protein":"g","sodium":"mg"}
    for k, v in ov.items():
        if v is not None and v >= 0:
            bucket = "micronutrients" if k in MICRO_KEYS else "nutrients"
            data[bucket][k] = {"value": float(v), "unit": umap.get(k,"g"), "manual": True}
    core  = {"calories","protein","carbohydrates","total_fat","sodium"}
    found = core & data["nutrients"].keys()
    data["missing_nutrients"] = list(core - found)
    data["parse_confidence"]  = "high" if len(found) >= 4 else ("medium" if len(found) >= 2 else "low")
    return data


# ─────────────────────────────────────────────
# HEALTH SCORING  (5-dimension, category-aware)
# ─────────────────────────────────────────────
def _g(data: Dict, key: str) -> Optional[float]:
    n = data.get("nutrients", {}).get(key)
    return n["value"] if n else None

def grade(score: int) -> Tuple[str, str]:
    if score >= 85: return "A", P["good"]
    if score >= 70: return "B", "#70c98b"
    if score >= 55: return "C", P["warn"]
    if score >= 40: return "D", "#c87040"
    return "F", P["bad"]

def score_label(data: Dict, profile: Dict, cat: Dict) -> Dict[str, Any]:
    """
    5 dimensions, each 0-20:
      Sugar · Sodium · Fat Quality · Protein · Fiber
    Summed → 0-100 base. Category weights applied.
    Calorie-density bonus/penalty on top.
    """
    R = {"total":0,"dims":{},"components":[],"tags":[],"flags":[],"summary":"","grade":"F","grade_color":P["bad"]}
    concern = set(cat.get("concern",[]))
    benefit = set(cat.get("benefit",[]))

    def w(k): return 1.5 if k in concern else (1.3 if k in benefit else 1.0)

    cals = _g(data,"calories") or 0

    # ── Sugar (0-20) ──
    sugar = _g(data,"sugar")
    if sugar is not None:
        dp = sugar / DRV["sugar"] * 100
        if   sugar <= 5:  ds,rs,cs = 20, f"Very low sugar ({sugar:.0f}g)",              "good"; R["tags"].append({"l":"Low Sugar ✓","c":"tg"})
        elif sugar <= 12: ds,rs,cs = 15, f"Low-moderate sugar ({sugar:.0f}g)",           "good"
        elif sugar <= 20: ds,rs,cs = 10, f"Moderate sugar ({sugar:.0f}g = {dp:.0f}% DV)","warn"; R["tags"].append({"l":f"Sugar {sugar:.0f}g","c":"tw"})
        elif sugar <= 30: ds,rs,cs =  5, f"High sugar ({sugar:.0f}g = {dp:.0f}% DV)",   "bad";  R["tags"].append({"l":f"High Sugar {sugar:.0f}g","c":"tb"}); R["flags"].append("⚠️ Sugar exceeds 60% of daily recommended intake.")
        else:             ds,rs,cs =  0, f"Very high sugar ({sugar:.0f}g = {dp:.0f}% DV)","bad"; R["tags"].append({"l":f"Very High Sugar","c":"tb"}); R["flags"].append("🚨 Sugar exceeds full daily limit in one serving.")
    else:
        ds,rs,cs = 10,"Sugar not detected — assumed average","warn"

    ws = min(20, round(ds * w("sugar")))
    R["dims"]["Sugar"] = {"score":ws,"max":20,"reason":rs,"cls":cs}
    R["components"].append({"n":"Sugar","score":ws,"max":20,"reason":rs,"cls":cs})

    # ── Sodium (0-20) ──
    sodium = _g(data,"sodium")
    sb     = cat.get("sodium_bench",200)
    if sodium is not None:
        dp = sodium / DRV["sodium"] * 100
        if   sodium <= sb*.5:  ds,rs,cs = 20, f"Very low sodium ({sodium:.0f}mg)",            "good"; R["tags"].append({"l":"Low Sodium ✓","c":"tg"})
        elif sodium <= sb:     ds,rs,cs = 15, f"Low sodium ({sodium:.0f}mg = {dp:.0f}% DV)",  "good"
        elif sodium <= sb*2:   ds,rs,cs = 10, f"Moderate sodium ({sodium:.0f}mg = {dp:.0f}% DV)","warn"
        elif sodium <= sb*3.5: ds,rs,cs =  5, f"High sodium ({sodium:.0f}mg = {dp:.0f}% DV)", "bad"; R["tags"].append({"l":"High Sodium","c":"tb"}); R["flags"].append("⚠️ Sodium is high — linked to elevated blood pressure.")
        else:                  ds,rs,cs =  0, f"Very high sodium ({sodium:.0f}mg = {dp:.0f}% DV)","bad"; R["tags"].append({"l":"Very High Sodium","c":"tb"}); R["flags"].append("🚨 Sodium very high for this food category.")
    else:
        ds,rs,cs = 10,"Sodium not detected","warn"
    ws = min(20, round(ds * w("sodium")))
    R["dims"]["Sodium"] = {"score":ws,"max":20,"reason":rs,"cls":cs}
    R["components"].append({"n":"Sodium","score":ws,"max":20,"reason":rs,"cls":cs})

    # ── Fat Quality (0-20) ──
    sat = _g(data,"saturated_fat")
    trans = _g(data,"trans_fat")
    fs = 20; fr = "Good fat quality"; fc = "good"
    if trans is not None and trans > 0.1:  # ignore OCR noise / "0g" labels
        fs -= 12; R["flags"].append("🚨 Trans fat detected — strongly linked to cardiovascular disease.")
        R["tags"].append({"l":"Trans Fat ⚠️","c":"tb"})
    if sat is not None:
        dp = sat / DRV["saturated_fat"] * 100
        if   sat <= 2:  fr = f"Low saturated fat ({sat:.1f}g)"; R["tags"].append({"l":"Low Sat. Fat ✓","c":"tg"})
        elif sat <= 5:  fs -= 3;  fr = f"Moderate sat. fat ({sat:.1f}g)"
        elif sat <= 10: fs -= 8;  fr = f"High sat. fat ({sat:.1f}g = {dp:.0f}% DV)"; fc = "warn"; R["tags"].append({"l":"High Sat. Fat","c":"tw"}); R["flags"].append("⚠️ Saturated fat is high — associated with cardiovascular risk.")
        else:           fs -= 15; fr = f"Very high sat. fat ({sat:.1f}g = {dp:.0f}% DV)"; fc = "bad"; R["tags"].append({"l":"High Sat. Fat","c":"tb"}); R["flags"].append("⚠️ Saturated fat very high for a single serving.")
    fs = max(0, fs)
    fc = "good" if fs >= 15 else ("warn" if fs >= 8 else "bad")
    ws = min(20, round(fs * w("saturated_fat")))
    R["dims"]["Fat Quality"] = {"score":ws,"max":20,"reason":fr,"cls":fc}
    R["components"].append({"n":"Fat Quality","score":ws,"max":20,"reason":fr,"cls":fc})

    # ── Protein (0-20) ──
    prot = _g(data,"protein")
    if prot is not None:
        if   prot >= 20: ds,rs,cs = 20, f"Excellent protein ({prot:.0f}g)","good"; R["tags"].append({"l":f"High Protein {prot:.0f}g","c":"tg"})
        elif prot >= 12: ds,rs,cs = 16, f"Good protein ({prot:.0f}g)",     "good"; R["tags"].append({"l":f"Protein {prot:.0f}g","c":"tg"})
        elif prot >= 6:  ds,rs,cs = 11, f"Moderate protein ({prot:.0f}g)", "warn"
        elif prot >= 2:  ds,rs,cs =  6, f"Low protein ({prot:.0f}g)",      "warn"
        else:            ds,rs,cs =  2, f"Very low protein ({prot:.0f}g)",  "bad"
    else:
        ds,rs,cs = 8,"Protein not detected","warn"
    ws = min(20, round(ds * w("protein")))
    R["dims"]["Protein"] = {"score":ws,"max":20,"reason":rs,"cls":cs}
    R["components"].append({"n":"Protein","score":ws,"max":20,"reason":rs,"cls":cs})

    # ── Fiber (0-20) ──
    fiber = _g(data,"fiber")
    cat_id = cat.get("id","other")
    if fiber is not None:
        if   fiber >= 6:   ds,rs,cs = 20, f"Excellent fiber ({fiber:.1f}g)", "good"; R["tags"].append({"l":f"High Fiber {fiber:.0f}g","c":"tg"})
        elif fiber >= 3.5: ds,rs,cs = 15, f"Good fiber ({fiber:.1f}g)",      "good"; R["tags"].append({"l":"Good Fiber ✓","c":"tg"})
        elif fiber >= 1.5: ds,rs,cs =  9, f"Low-moderate fiber ({fiber:.1f}g)","warn"
        elif fiber >= .5:  ds,rs,cs =  4, f"Low fiber ({fiber:.1f}g)",         "warn"
        else:              ds,rs,cs =  0, f"Very low fiber ({fiber:.1f}g)",     "bad"
    elif cat_id == "drink":
        ds,rs,cs = 12,"Fiber N/A for beverages","good"
    else:
        ds,rs,cs = 5,"Fiber not detected — assumed low","warn"
    ws = min(20, round(ds * w("fiber")))
    R["dims"]["Fiber"] = {"score":ws,"max":20,"reason":rs,"cls":cs}
    R["components"].append({"n":"Fiber","score":ws,"max":20,"reason":rs,"cls":cs})

    # ── Base sum → 0-100 ──
    base = min(100, sum(d["score"] for d in R["dims"].values()))

    # ── Calorie-density bonus/penalty ──
    cb = cat.get("cal_bench",200)
    if cals > 0:
        if   cals > cb*2.5: base = max(0,base-8);  R["flags"].append(f"⚠️ Calorie-dense for this category ({cals:.0f} kcal/serving vs typical {cb} kcal)."); R["tags"].append({"l":f"{cals:.0f} kcal","c":"tb"})
        elif cals > cb*1.5: base = max(0,base-4);  R["tags"].append({"l":f"{cals:.0f} kcal","c":"tw"})
        else:               R["tags"].append({"l":f"{cals:.0f} kcal","c":"tn"})

    R["total"] = max(0, min(100, base))
    g_letter, g_color = grade(R["total"])
    R["grade"] = g_letter; R["grade_color"] = g_color

    s = R["total"]
    R["summary"] = (
        "Outstanding nutritional profile for this category." if s >= 85 else
        "Good choice — minor concerns noted." if s >= 70 else
        "Average quality. Consume in moderation." if s >= 55 else
        "Below average. Watch portion sizes." if s >= 40 else
        "Poor nutritional profile. High in multiple concerning nutrients."
    )
    return R


# ─────────────────────────────────────────────
# BMI
# ─────────────────────────────────────────────
def calc_bmi(wt, ht):
    if wt <= 0 or ht <= 0: return 0., "", ""
    bmi = wt / (ht/100)**2
    for lo,hi,lbl,col in BMI_CATS:
        if lo <= bmi < hi: return round(bmi,1), lbl, col
    return round(bmi,1), "Obese (Class II+)", "#b03030"

def bmr_tdee(wt, ht, age, sex, act):
    bmr = (10*wt + 6.25*ht - 5*age + 5) if sex=="Male" else (10*wt + 6.25*ht - 5*age - 161)
    m = {"Sedentary":1.2,"Lightly Active":1.375,"Moderately Active":1.55,"Very Active":1.725,"Extremely Active":1.9}.get(act,1.2)
    return round(bmr), round(bmr*m)


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def radar_fig(dims: Dict) -> go.Figure:
    cats = list(dims.keys())
    vals = [dims[c]["score"]/dims[c]["max"]*100 for c in cats]
    catsc = cats+[cats[0]]; valsc = vals+[vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=valsc,theta=catsc,fill="toself",
        fillcolor="rgba(240,165,0,.10)",line=dict(color=P["accent"],width=2),name="Score"))
    fig.add_trace(go.Scatterpolar(r=[100]*(len(cats)+1),theta=catsc,
        line=dict(color=P["border"],width=1,dash="dot"),fill=None,showlegend=False))
    fig.update_layout(
        polar=dict(bgcolor=P["surface2"],
                   radialaxis=dict(visible=True,range=[0,100],tickfont={"size":8,"color":P["muted"]},
                                   gridcolor=P["border"],linecolor=P["border"]),
                   angularaxis=dict(tickfont={"size":10,"color":P["text"]},gridcolor=P["border"])),
        showlegend=False,height=300,margin=dict(l=40,r=40,t=20,b=20),
        paper_bgcolor="rgba(0,0,0,0)",font={"color":P["text"]})
    return fig

def donut_fig(prot, carbs, fat) -> go.Figure:
    pk,ck,fk = prot*4, carbs*4, fat*9
    total = pk+ck+fk
    if total == 0:
        fig = go.Figure(go.Pie(labels=["Protein","Carbohydrates","Fat"],values=[1,1,1],hole=.65,
                               marker={"colors":[P["border"]]*3},textinfo="none",hoverinfo="skip"))
        fig.update_layout(height=240,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False,annotations=[dict(text="No macro<br>data",x=.5,y=.5,
                          font_size=13,showarrow=False,font={"color":P["muted"]})])
        return fig
    fig = go.Figure(go.Pie(
        labels=["Protein","Carbohydrates","Fat"],values=[pk,ck,fk],hole=.65,
        marker={"colors":[MACRO_C["Protein"],MACRO_C["Carbohydrates"],MACRO_C["Fat"]],
                "line":{"color":P["surface"],"width":2}},
        textfont={"color":P["text"],"size":11},
        hovertemplate="<b>%{label}</b><br>%{value:.0f} kcal (%{percent})<extra></extra>"))
    fig.update_layout(
        height=260,margin=dict(l=0,r=0,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",showlegend=True,
        legend=dict(font={"color":P["muted"],"size":11},bgcolor="rgba(0,0,0,0)",
                    orientation="v",x=1.02,y=.5,xanchor="left",yanchor="middle"),
        annotations=[dict(text=f"<b>{total:.0f}</b><br>kcal",x=.5,y=.5,font_size=15,
                          showarrow=False,font={"color":P["text"],"family":"Playfair Display"})])
    return fig

def dv_fig(data: Dict) -> go.Figure:
    dvs,nutr = data.get("daily_values",{}), data.get("nutrients",{})
    nm = {"total_fat":("Total Fat",DRV["total_fat"]),"saturated_fat":("Sat. Fat",DRV["saturated_fat"]),
          "sodium":("Sodium",DRV["sodium"]),"carbohydrates":("Carbohydrates",DRV["carbohydrates"]),
          "fiber":("Fiber",DRV["fiber"]),"protein":("Protein",DRV["protein"]),"sugar":("Sugar",DRV["sugar"])}
    rows=[]
    for k,(lbl,drv) in nm.items():
        pct=dvs.get(k)
        if pct is None:
            nd=nutr.get(k)
            if nd: pct=nd["value"]/drv*100
        if pct is not None:
            col=P["good"] if pct<20 else (P["warn"] if pct<50 else P["bad"])
            rows.append({"L":lbl,"p":min(pct,160),"c":col,"r":pct})
    if not rows: return go.Figure()
    df=pd.DataFrame(rows).sort_values("p",ascending=True)
    fig=go.Figure(go.Bar(x=df["p"],y=df["L"],orientation="h",
        marker={"color":df["c"].tolist(),"opacity":.82},
        text=[f"{p:.0f}%" for p in df["r"]],textposition="outside",
        textfont={"color":P["muted"],"size":11},
        hovertemplate="<b>%{y}</b>: %{x:.1f}% DV<extra></extra>"))
    fig.add_vline(x=100,line_dash="dash",line_color=P["muted"],line_width=1,opacity=.35)
    fig.update_layout(
        height=max(240,len(rows)*40),margin=dict(l=10,r=55,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"gridcolor":P["border"],"color":P["muted"],"range":[0,max(170,df["p"].max()+30)]},
        yaxis={"gridcolor":P["border"],"color":P["text"]},font={"color":P["text"]})
    return fig

def bmi_fig(bmi: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=bmi,
        number={"font":{"size":44,"color":color,"family":"Playfair Display"}},
        domain={"x":[0,1],"y":[0,1]},
        gauge={"axis":{"range":[10,45],"tickcolor":P["muted"],"tickfont":{"color":P["muted"],"size":9}},
               "bar":{"color":color,"thickness":.22},"bgcolor":P["surface2"],"borderwidth":0,
               "steps":[{"range":[10,18.5],"color":"rgba(106,179,245,.13)"},
                        {"range":[18.5,25],"color":"rgba(92,184,138,.13)"},
                        {"range":[25,30],"color":"rgba(232,136,74,.13)"},
                        {"range":[30,45],"color":"rgba(217,85,85,.13)"}]}))
    fig.update_layout(height=230,margin=dict(l=30,r=30,t=20,b=20),
                      paper_bgcolor="rgba(0,0,0,0)",font={"color":P["text"]})
    return fig


# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────
def recs_fn(data, sr, profile, cat, tdee=None):
    recs=[]
    nutr=data.get("nutrients",{})
    g2=lambda k:(nutr.get(k) or {}).get("value",0)
    cals,prot,carbs=g2("calories"),g2("protein"),g2("carbohydrates")
    fat,fiber,sugar,sodium=g2("total_fat"),g2("fiber"),g2("sugar"),g2("sodium")
    cat_id=cat.get("id","other")

    if tdee and cals>0:
        pct=cals/tdee*100
        if pct>35: recs.append({"i":"🔥","t":"Calorie-Dense Serving","b":f"~{pct:.0f}% of your daily energy ({tdee:.0f} kcal). Balance with lighter meals."})
    if prot<8 and profile.get("protein",.20)>=.25:
        recs.append({"i":"💪","t":"Low Protein for Your Goal","b":"Add eggs, legumes, Greek yogurt, or lean meat to reach your daily protein target."})
    if fiber<2 and cat_id not in ("drink",):
        recs.append({"i":"🌾","t":"Low Dietary Fiber","b":"Pair this with vegetables, legumes, or whole grains to boost fiber intake."})
    if sugar>20:
        recs.append({"i":"🍬","t":"High Sugar Alert","b":f"{sugar:.0f}g sugar. WHO recommends <50g added sugars/day — watch other sugary foods today."})
    if sodium>800:
        recs.append({"i":"🧂","t":"High Sodium","b":f"{sodium:.0f}mg = {sodium/DRV['sodium']*100:.0f}% of daily limit. Stay hydrated and limit other salty foods."})
    if cat_id=="drink" and sugar>10:
        recs.append({"i":"💧","t":"Consider a Water Swap","b":"Sugary drinks add calories without satiety. Try sparkling water with lemon or lime."})
    if cat_id in ("chips","namkeen") and sodium>300:
        recs.append({"i":"🥗","t":"Balance with Fresh Foods","b":"Salty snacks pair well with fresh fruits or veg to balance electrolytes and add fiber."})
    if sr["total"]>=75 and not recs:
        recs.append({"i":"✅","t":"Well-Balanced Choice","b":"Scores well across all dimensions. A great fit for a balanced diet!"})
    if not recs:
        recs.append({"i":"📊","t":"Consume in Moderation","b":"Mixed profile. Watch portions and complement with nutrient-dense whole foods."})
    return recs[:4]


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_session():
    # Note: auth_user / auth_display intentionally NOT pre-set here.
    # Their absence from session_state is what triggers the login wall.
    defs = {
        "history": [],
        "user_profile": {"weight_kg":0.,"height_cm":0.,"age":0,"sex":"Male","activity":"Moderately Active","goal":"Balanced"},
        "sel_cat": "🏷️ Other / Unknown",
        "parsed_data": None,
        "ocr_strategy": "—",
        "ocr_score": 0,
        "dash_page": 0,
    }
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k] = v

def save_history(name, score, grade_l, cat_lbl, data):
    st.session_state.history.insert(0, {
        "timestamp": datetime.datetime.now().strftime("%H:%M · %d %b"),
        "product": name or "Unknown",
        "score": score, "grade": grade_l, "category": cat_lbl,
        "calories": (data.get("nutrients",{}).get("calories") or {}).get("value","—"),
        "brand":  data.get("brand"),
        "notes":  data.get("notes",""),
    })
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]
    # Persist to user account if logged in
    persist_scan(
        username=st.session_state.get("auth_user"),
        product_name=name,
        category=cat_lbl,
        score=score,
        grade=grade_l,
        nutrients=data.get("nutrients",{}),
        brand=data.get("brand"),
        notes=data.get("notes",""),
    )


# ─────────────────────────────────────────────
# FALLBACK FORM
# ─────────────────────────────────────────────
def fallback_form(missing: List[str], existing_data: Dict) -> Dict[str, Optional[float]]:
    lmap = {"calories":("Calories","kcal"),"total_fat":("Total Fat","g"),
            "saturated_fat":("Sat. Fat","g"),"carbohydrates":("Carbohydrates","g"),
            "fiber":("Fiber","g"),"sugar":("Sugars","g"),
            "protein":("Protein","g"),"sodium":("Sodium","mg")}

    st.markdown("""
    <div class="fallback-box">
        <b style="color:#e8884a;">📝 Manual Entry Mode</b><br>
        <span style="color:#666b8a; font-size:.85rem;">
        OCR couldn't read all nutrients clearly. Type values directly from the label below.
        Leave any field blank to skip it.
        </span>
    </div>
    """, unsafe_allow_html=True)

    ov = {}
    miss_in_map = [k for k in missing if k in lmap]
    other_keys  = [k for k in lmap if k not in miss_in_map]

    if miss_in_map:
        st.markdown(f"<span style='color:{P['warn']}; font-size:.8rem; font-weight:600;'>Missing nutrients</span>",
                    unsafe_allow_html=True)
        cols = st.columns(min(len(miss_in_map), 4))
        for i, key in enumerate(miss_in_map):
            lbl, unit = lmap[key]
            with cols[i % 4]:
                v = st.number_input(f"{lbl} ({unit})", min_value=0., value=None,
                                    step=.5, key=f"man_{key}", placeholder="—")
                ov[key] = v

    with st.expander("✏️ Edit other detected values", expanded=False):
        cols2 = st.columns(4)
        for i, key in enumerate(other_keys):
            lbl, unit = lmap[key]
            existing = existing_data.get("nutrients",{}).get(key)
            default  = float(existing["value"]) if existing else None
            with cols2[i % 4]:
                v = st.number_input(f"{lbl} ({unit})", min_value=0.,
                                    value=default, step=.5, key=f"man_{key}", placeholder="—")
                ov[key] = v

    return ov


# ─────────────────────────────────────────────
# TOP BAR  (replaces sidebar — always visible)
# ─────────────────────────────────────────────
def render_topbar():
    p = st.session_state.user_profile

    # ── Row 1: Logo + Auth + Recent scans pills ──────────────────────────────
    c_logo, c_auth, c_hist = st.columns([2, 2, 6], gap="small")

    with c_logo:
        disp   = st.session_state.get("auth_display","Guest")
        _acc   = P["accent"]
        _muted = P["muted"]
        st.markdown(
            f"<div style='padding:.4rem 0;font-family:Playfair Display,serif;"
            f"font-size:1.4rem;color:{_acc};font-weight:700;'>🔬 NutriScan</div>"
            f"<div style='font-size:.72rem;color:{_muted};margin-top:-.2rem;'>👤 {disp}</div>",
            unsafe_allow_html=True
        )

    with c_auth:
        auth_user = st.session_state.get("auth_user")
        if auth_user:
            if st.button("Sign Out", key="signout_top"):
                st.session_state.pop("auth_user", None)
                st.session_state.pop("auth_display", None)
                st.rerun()
        else:
            if st.button("Sign In", key="signin_top"):
                st.session_state.pop("auth_user", None)
                st.session_state.pop("auth_display", None)
                st.rerun()

    with c_hist:
        hist = st.session_state.history
        if hist:
            pills = ""
            for e in hist[:5]:
                sc = P["good"] if e["score"]>=70 else (P["warn"] if e["score"]>=45 else P["bad"])
                _surf2 = P["surface2"]
                _bdr   = P["border"]
                _muted2= P["muted"]
                pills += (
                    f"<span style='display:inline-flex;align-items:center;gap:.3rem;"
                    f"background:{_surf2};border:1px solid {_bdr};"
                    f"border-radius:6px;padding:.2rem .55rem;font-size:.74rem;margin-right:.3rem;'>"
                    f"<span style='color:{sc};font-weight:700;'>{e['grade']}</span>"
                    f"<span style='color:{_muted2};'>{e['product'][:14]}</span>"
                    f"</span>"
                )
            st.markdown(
                f"<div style='padding:.35rem 0;overflow-x:auto;white-space:nowrap;'>{pills}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='padding:.5rem 0;font-size:.78rem;color:#666b8a;'>No scans yet — upload a label to get started</div>",
                unsafe_allow_html=True
            )

    _bdr2 = P["border"]
    st.markdown(f"<hr style='margin:.5rem 0;border-color:{_bdr2};'>", unsafe_allow_html=True)

    # ── Row 2: Profile inputs (compact, always visible) ──────────────────────
    pc1,pc2,pc3,pc4,pc5,pc6 = st.columns([1.2,1.2,0.8,0.9,2,2], gap="small")
    with pc1: p["weight_kg"] = st.number_input("Weight (kg)", 0., 300., p["weight_kg"], .5, key="wt")
    with pc2: p["height_cm"] = st.number_input("Height (cm)", 0., 250., p["height_cm"], .5, key="ht")
    with pc3: p["age"]       = st.number_input("Age", 0, 120, p["age"], 1, key="age")
    with pc4: p["sex"]       = st.selectbox("Sex", ["Male","Female"], key="sex", index=0 if p.get("sex","Male")=="Male" else 1)
    with pc5:
        act_opts = ["Sedentary","Lightly Active","Moderately Active","Very Active","Extremely Active"]
        p["activity"] = st.selectbox("Activity", act_opts, key="act",
                                     index=act_opts.index(p.get("activity","Moderately Active")))
    with pc6:
        goal_opts = list(GOAL_PROFILES.keys())
        p["goal"] = st.selectbox("Goal", goal_opts, key="goal",
                                 index=goal_opts.index(p.get("goal","Balanced")) if p.get("goal") in goal_opts else 0)

    # Auto-save profile to user account when logged in
    auth_user = st.session_state.get("auth_user")
    if auth_user:
        from auth import update_profile
        update_profile(auth_user, p)

    # Hidden AI key (no sidebar needed)
    if not _get_groq_key():
        with st.expander("⚙️ Settings", expanded=False):
            runtime_key = st.text_input("Enhancement key (optional)", type="password",
                                        key="groq_rt_key", label_visibility="visible")
            if runtime_key and runtime_key.startswith("gsk_"):
                import os; os.environ["GROQ_API_KEY"] = runtime_key
                st.rerun()

    st.markdown(f"<hr style='margin:.5rem 0 1rem;border-color:{_bdr2};'>", unsafe_allow_html=True)
    return p


# ─────────────────────────────────────────────
# ANALYSIS DISPLAY
# ─────────────────────────────────────────────
def render_analysis(data, sr, profile, user_profile, cat, cat_lbl):
    nutr=data.get("nutrients",{})
    tdee=None
    if user_profile["weight_kg"]>0 and user_profile["height_cm"]>0 and user_profile["age"]>0:
        _,tdee=bmr_tdee(user_profile["weight_kg"],user_profile["height_cm"],
                         user_profile["age"],user_profile["sex"],user_profile["activity"])

    prot_g  = (nutr.get("protein") or {}).get("value",0)
    carbs_g = (nutr.get("carbohydrates") or {}).get("value",0)
    fat_g   = (nutr.get("total_fat") or {}).get("value",0)
    total   = sr["total"]
    gc      = sr["grade_color"]
    gr      = sr["grade"]

    # ── Row 1: Score + Macro donut ──
    c1,c2=st.columns([1,1],gap="large")
    with c1:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">🏆 Health Score</div>',unsafe_allow_html=True)
        # Grade ring
        r,g2,b2 = int(gc[1:3],16),int(gc[3:5],16),int(gc[5:7],16)
        st.markdown(f"""
        <div style='text-align:center;padding:.4rem 0;'>
            <div class='grade-ring' style='background:rgba({r},{g2},{b2},.14);border:2.5px solid {gc};color:{gc};'>
                {gr}
            </div>
            <div style='font-family:"Playfair Display",serif;font-size:2.7rem;color:{gc};line-height:1;margin:.25rem 0 .1rem;'>{total}</div>
            <div style='color:{P["muted"]};font-size:.8rem;'>out of 100</div>
        </div>
        <div style='background:{P["surface2"]};border-radius:999px;height:7px;margin:.65rem 0 .4rem;overflow:hidden;'>
            <div style='background:{gc};width:{total}%;height:100%;border-radius:999px;'></div>
        </div>
        <p style='text-align:center;color:{P["muted"]};font-size:.84rem;margin-top:.25rem;'>{sr["summary"]}</p>
        """, unsafe_allow_html=True)
        tags_html="".join(f'<span class="tag {t["c"]}">{t["l"]}</span>' for t in sr["tags"])
        if tags_html: st.markdown(f'<div style="text-align:center;margin-top:.5rem;">{tags_html}</div>',unsafe_allow_html=True)
        if sr["flags"]:
            st.markdown("<br>",unsafe_allow_html=True)
            for fl in sr["flags"]:
                st.markdown(f'<div style="color:{P["warn"]};font-size:.8rem;margin:.22rem 0;line-height:1.4;">{fl}</div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">🍽️ Macronutrient Split</div>',unsafe_allow_html=True)
        st.plotly_chart(donut_fig(prot_g,carbs_g,fat_g),use_container_width=True,config={"displayModeBar":False})
        if prot_g+carbs_g+fat_g>0:
            mc=st.columns(3)
            for col,(nm,val,cc) in zip(mc,[("Protein",prot_g,MACRO_C["Protein"]),
                                           ("Carbs",carbs_g,MACRO_C["Carbohydrates"]),
                                           ("Fat",fat_g,MACRO_C["Fat"])]):
                col.markdown(f"""<div style='text-align:center;padding:.35rem 0;'>
                    <div style='font-size:1.05rem;font-weight:700;color:{cc};'>{val:.0f}g</div>
                    <div style='font-size:.72rem;color:{P["muted"]};'>{nm}</div></div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # ── Row 2: Radar + DV bars ──
    c3,c4=st.columns([1,1],gap="large")
    with c3:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">📐 Score Dimensions</div>',unsafe_allow_html=True)
        st.plotly_chart(radar_fig(sr["dims"]),use_container_width=True,config={"displayModeBar":False})
        cls_col={"good":P["good"],"warn":P["warn"],"bad":P["bad"],"neutral":P["muted"]}
        for comp in sr["components"]:
            c=cls_col.get(comp["cls"],P["muted"])
            bw=int(comp["score"]/comp["max"]*100)
            st.markdown(f"""<div class="nut-row">
                <span class="nut-name">{comp['n']}</span>
                <span style='display:flex;align-items:center;gap:.4rem;'>
                    <span style='width:55px;height:5px;background:{P["surface2"]};border-radius:3px;display:inline-block;overflow:hidden;'>
                        <span style='display:block;width:{bw}%;height:100%;background:{c};border-radius:3px;'></span>
                    </span>
                    <span class='nut-val' style='color:{c};min-width:34px;text-align:right;'>{comp['score']}/{comp['max']}</span>
                </span></div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">📊 % Daily Value</div>',unsafe_allow_html=True)
        dvf=dv_fig(data)
        if dvf.data: st.plotly_chart(dvf,use_container_width=True,config={"displayModeBar":False})
        else: st.info("% DV data not found. Clear photo or use manual entry for better results.")
        st.markdown("</div>",unsafe_allow_html=True)

    # ── Row 3: Full table ──
    st.markdown('<div class="ns-card">',unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">🔬 Full Nutrition Breakdown</div>',unsafe_allow_html=True)
    display_keys=[
        ("serving_size","Serving Size"),("servings_per_container","Servings / Container"),
        ("calories","Calories"),("total_fat","Total Fat"),("saturated_fat","  ↳ Saturated Fat"),
        ("trans_fat","  ↳ Trans Fat"),("cholesterol","Cholesterol"),("sodium","Sodium"),
        ("carbohydrates","Total Carbohydrates"),("fiber","  ↳ Dietary Fiber"),
        ("sugar","  ↳ Total Sugars"),("added_sugar","      ↳ Added Sugars"),("protein","Protein"),
    ]
    html=""
    for key,lbl in display_keys:
        nd=nutr.get(key)
        if nd and nd.get("value") is not None:
            dv=data.get("daily_values",{}).get(key)
            dvs=f'<span style="color:{P["muted"]};font-size:.78rem;">{dv:.0f}% DV</span>' if dv else ""
            mb=' <span class="tag ta" style="font-size:.68rem;">manual</span>' if nd.get("manual") else ""
            html+=f'<div class="nut-row"><span class="nut-name">{lbl}</span><span class="nut-val">{nd["value"]:.1f}{nd["unit"]} {dvs}{mb}</span></div>'
    for key,lbl in [("vitamin_d","Vitamin D"),("calcium","Calcium"),("iron","Iron"),("potassium","Potassium")]:
        md=data.get("micronutrients",{}).get(key)
        if md:
            dv=data.get("daily_values",{}).get(key)
            dvs=f'<span style="color:{P["muted"]};font-size:.78rem;">{dv:.0f}% DV</span>' if dv else ""
            html+=f'<div class="nut-row"><span class="nut-name">{lbl}</span><span class="nut-val">{md["value"]:.1f}{md["unit"]} {dvs}</span></div>'
    if html: st.markdown(html,unsafe_allow_html=True)
    else:    st.warning("Not enough data. Use manual entry for better results.")
    st.markdown("</div>",unsafe_allow_html=True)

    # ── Row 4: Recommendations ──
    recs=recs_fn(data,sr,profile,cat,tdee)
    if recs:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">💡 Personalised Insights</div>',unsafe_allow_html=True)
        rcols=st.columns(min(len(recs),4))
        for i,r in enumerate(recs):
            card_c="card-good" if "✅" in r["i"] else "card-warn"
            with rcols[i%len(rcols)]:
                st.markdown(f"""<div class="{card_c}" style='height:100%;'>
                    <div style='font-size:1.4rem;margin-bottom:.3rem;'>{r['i']}</div>
                    <div style='font-weight:600;font-size:.88rem;margin-bottom:.3rem;color:{P["text"]};'>{r['t']}</div>
                    <div style='font-size:.8rem;color:{P["muted"]};line-height:1.5;'>{r['b']}</div>
                </div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # ── OCR Debug ──
    with st.expander("🛠 Raw OCR & Debug",expanded=False):
        pc=data.get("parse_confidence","low")
        cls2="tg" if pc=="high" else ("tw" if pc=="medium" else "tb")
        strat=st.session_state.get("ocr_strategy","—")
        dc1,dc2=st.columns(2)
        dc1.markdown(f'<span class="tag {cls2}">Parse Confidence: {pc.title()}</span>',unsafe_allow_html=True)
        dc2.markdown(f'<span class="tag tn">OCR Strategy: {strat}</span>',unsafe_allow_html=True)
        miss=data.get("missing_nutrients",[])
        if miss: st.markdown(f'<span class="tag tw">Missing: {", ".join(miss)}</span>',unsafe_allow_html=True)
        # Show per-nutrient parse sources with colour-coded badges
        src_color = {"regex":"#6ab3f5","positional":P["good"],"joined":P["warn"],"groq":"#c084fc","manual":P["accent"]}
        src_rows=""
        for k,v in data.get("nutrients",{}).items():
            src=v.get("source","regex")
            sc2=src_color.get(src,P["muted"])
            src_rows+=f"<div class='nut-row'><span class='nut-name'>{k.replace('_',' ').title()}</span><span style='display:flex;align-items:center;gap:.5rem;'><span class='nut-val'>{v.get('value')} {v.get('unit')}</span><span style='font-size:.7rem;padding:.1rem .45rem;border-radius:4px;background:rgba(255,255,255,.05);border:1px solid {sc2};color:{sc2};'>{src}</span></span></div>"
        if src_rows: st.markdown(src_rows,unsafe_allow_html=True)
        st.markdown(f"<br><b style='color:{P['muted']};font-size:.8rem;'>Raw OCR Text</b>",unsafe_allow_html=True)
        st.code(data.get("raw_text",""),language="text")

    # ── Phase 2 cloud DB stub ──
    # When db.py is ready, uncomment:
    # if st.button("☁️ Save to Cloud", key="cloud_save"):
    #     from db import save_scan
    #     db_id = save_scan(
    #         product_name = data.get("product_name"),
    #         brand        = data.get("brand"),        # from brand image scan
    #         label_img_url= None,                     # S3/Supabase Storage URL
    #         category     = cat_lbl,
    #         score        = sr["total"],
    #         nutrients    = data["nutrients"],
    #     )
    #     st.success(f"Saved! ID: {db_id}")


# ─────────────────────────────────────────────
# BMI TAB
# ─────────────────────────────────────────────
def render_bmi(user_profile):
    w,h,age,sex,act = (user_profile["weight_kg"],user_profile["height_cm"],
                       user_profile["age"],user_profile["sex"],user_profile["activity"])
    if w<=0 or h<=0:
        st.info("Enter weight and height in the sidebar to see your BMI and energy targets.")
        return

    bmi,lbl,bcol = calc_bmi(w,h)
    bmr_v,tdee_v = bmr_tdee(w,h,max(age,1),sex,act)
    c1,c2=st.columns([1,1],gap="large")

    with c1:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">⚖️ BMI Score</div>',unsafe_allow_html=True)
        st.plotly_chart(bmi_fig(bmi,bcol),use_container_width=True,config={"displayModeBar":False})
        st.markdown(f'<p style="text-align:center;font-size:1rem;font-weight:600;color:{bcol};margin:-.4rem 0 .8rem;">{lbl}</p>',unsafe_allow_html=True)
        for lo,hi,cl,cc in BMI_CATS:
            mk=f'<span style="color:{cc};font-size:.75rem;">◀ you</span>' if lo<=bmi<hi else ""
            his=f"{hi:.0f}" if hi<999 else "+"
            st.markdown(f'<div class="nut-row"><span style="color:{cc};font-size:.85rem;">■ {cl} ({lo}–{his})</span>{mk}</div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ns-card">',unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">⚡ Energy Needs</div>',unsafe_allow_html=True)
        mc1,mc2=st.columns(2)
        mc1.metric("BMR",f"{bmr_v:,} kcal/day")
        mc2.metric(f"TDEE",f"{tdee_v:,} kcal/day")
        st.markdown("<br>",unsafe_allow_html=True)
        goal=user_profile["goal"]
        adj={"Weight Loss":("Deficit −500 kcal",tdee_v-500),"Balanced":("Maintenance",tdee_v),
             "High Protein":("Maintenance",tdee_v),"Low Carb / Keto":("Maintenance",tdee_v),
             "Endurance / Carb":("Surplus +150",tdee_v+150)}.get(goal,("Maintenance",tdee_v))
        desc,tc=adj
        st.markdown(f"""<div class="card-amber">
            <div style='font-size:.8rem;color:{P["muted"]};margin-bottom:.25rem;'>Target for <b style='color:{P["text"]};'>{goal}</b></div>
            <div style='font-size:1.55rem;font-weight:700;color:{P["accent"]};'>{max(0,tc):,} kcal</div>
            <div style='font-size:.76rem;color:{P["muted"]};margin-top:.12rem;'>{desc}</div>
        </div>""",unsafe_allow_html=True)
        ps=GOAL_PROFILES[goal]
        st.markdown(f"<br><span style='color:{P['muted']};font-size:.76rem;font-weight:600;letter-spacing:.05em;'>MACRO TARGETS</span>",unsafe_allow_html=True)
        for mn,pct,mc in [("Protein",ps["protein"],MACRO_C["Protein"]),("Carbohydrates",ps["carbs"],MACRO_C["Carbohydrates"]),("Fat",ps["fat"],MACRO_C["Fat"])]:
            grams=(tc*pct)/(4 if mn!="Fat" else 9)
            st.markdown(f'<div class="nut-row"><span style="color:{mc};font-size:.88rem;">■ {mn}</span><span class="nut-val">{grams:.0f}g · {pct*100:.0f}%</span></div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    init_session()

    # ── Auth wall: show only if user has never made a choice (login or skip) ──
    # auth_user key is absent from session_state until user logs in or clicks Skip.
    # After Skip: auth_user=None, auth_display="Guest"  → wall does NOT re-appear.
    # After Login: auth_user=<str>                       → wall does NOT re-appear.
    if "auth_user" not in st.session_state:
        authenticated = render_auth_page()
        if not authenticated:
            return   # stay on login screen until user acts

    user_profile = render_topbar()
    goal_profile = GOAL_PROFILES[user_profile["goal"]]

    tab_scan, tab_bmi, tab_dash = st.tabs(["🔬 Scan Label","⚖️ BMI & Energy","📊 My Dashboard"])

    # ── SCAN TAB ──────────────────────────────
    with tab_scan:
        up_col, res_col = st.columns([1,2],gap="large")

        with up_col:
            st.markdown('<div class="ns-card">',unsafe_allow_html=True)
            st.markdown('<div class="sec-hdr">📤 Upload Label</div>',unsafe_allow_html=True)

            # Step 1: Category
            st.markdown(f"<span style='color:{P['muted']};font-size:.8rem;font-weight:600;letter-spacing:.04em;'>STEP 1 — FOOD CATEGORY</span>",unsafe_allow_html=True)
            cat_opts = list(FOOD_CATS.keys())
            sel_cat  = st.selectbox("Category",cat_opts,
                                    index=cat_opts.index(st.session_state.sel_cat),
                                    key="cat_sel",label_visibility="collapsed")
            st.session_state.sel_cat = sel_cat
            cat_cfg  = FOOD_CATS[sel_cat]

            concern_s = ", ".join(c.replace("_"," ").title() for c in cat_cfg["concern"])
            benefit_s = ", ".join(c.replace("_"," ").title() for c in cat_cfg["benefit"])
            st.markdown(f"""
            <div style='background:{P["surface2"]};border-radius:8px;padding:.55rem .85rem;
                        margin:.35rem 0 .85rem;font-size:.78rem;color:{P["muted"]};'>
                ⚠️ Watch: <b style='color:{P["warn"]};'>{concern_s}</b><br>
                ✓ Rewarded: <b style='color:{P["good"]};'>{benefit_s}</b>
            </div>""",unsafe_allow_html=True)

            # Step 2: Upload
            st.markdown(f"<span style='color:{P['muted']};font-size:.8rem;font-weight:600;letter-spacing:.04em;'>STEP 2 — IMAGE</span>",unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload nutrition label image",type=["png","jpg","jpeg"],key="upl",label_visibility="collapsed")

            if uploaded:
                pil_img = Image.open(uploaded)
                dw=350; dr=dw/pil_img.width
                st.image(pil_img.resize((dw,int(pil_img.height*dr)),Image.Resampling.LANCZOS),
                         caption="Uploaded label",use_container_width=True)
                st.markdown(f"<div style='font-size:.76rem;color:{P['muted']};line-height:1.6;margin:.45rem 0;'>✦ Flat angle, even lighting, no glare<br>✦ Local processing — nothing leaves your device</div>",unsafe_allow_html=True)
                st.markdown(f"<span style='color:{P['muted']};font-size:.8rem;font-weight:600;letter-spacing:.04em;'>STEP 3 — PRODUCT NAME (optional)</span>",unsafe_allow_html=True)
                manual_name = st.text_input("Product name", placeholder="e.g. Lay's Classic Salted", key="prod_name_input", label_visibility="collapsed")
                scan_notes  = st.text_input("Notes (optional)", placeholder="e.g. bought at DMart, 200g pack", key="scan_notes_input", label_visibility="collapsed")
                analyse = st.button("🔍 Analyse Label",key="analyse_btn",use_container_width=True)
            else:
                st.markdown(f"<div style='text-align:center;padding:2.5rem 1rem;color:{P['muted']};'><div style='font-size:2.6rem;'>📷</div><p style='margin:.4rem 0 .2rem;'>Drop a nutrition label image above</p><p style='font-size:.76rem;'>Local processing only.</p></div>",unsafe_allow_html=True)
                analyse = False
            st.markdown("</div>",unsafe_allow_html=True)

        with res_col:
            if uploaded and analyse:
                # ── Step 1: EasyOCR ──────────────────────────────────────────
                with st.spinner("🔍 Running OCR…"):
                    try:
                        ocr_res = run_ocr(pil_img)
                    except Exception as e:
                        st.error(f"OCR engine failed: {e}")
                        ocr_res = []

                if not ocr_res:
                    st.error("OCR returned no text. Try a clearer image, or add your Groq API key for AI-powered extraction.")
                else:
                    data    = parse_label(ocr_res)
                    pc      = data.get("parse_confidence", "low")
                    missing = data.get("missing_nutrients", [])
                    strat   = st.session_state.get("ocr_strategy", "—")

                    # ── Step 2: Groq Vision Fallback ─────────────────────────────
                    # Triggers automatically if EasyOCR confidence is still low/medium
                    # or any core nutrient is missing. Requires GROQ_API_KEY.
                    groq_used = False
                    groq_available = bool(_get_groq_key())

                    if (pc in ("low", "medium") or missing) and groq_available:
                        with st.spinner("🤖 EasyOCR partial — trying Groq vision AI…"):
                            groq_json = groq_extract(pil_img)
                            if groq_json:
                                data     = groq_to_parsed_data(groq_json, data)
                                pc       = data.get("parse_confidence", "low")
                                missing  = data.get("missing_nutrients", [])
                                groq_used = True

                    # ── Confidence banner ──────────────────────────────────────────
                    groq_badge = ""  # internal detail, not shown to user
                    ocr_badge  = f" · OCR: {strat}"
                    prod_name  = data.get("product_name") or "Scanned Product"

                    if pc == "high":
                        st.success(f"✅ High confidence parse{groq_badge}{ocr_badge} · {prod_name}")
                    elif pc == "medium":
                        st.warning(f"⚠️ Partial parse{groq_badge} — fill in missing values below if needed.")
                    else:
                        if not groq_available:
                            st.error(
                                f"⚠️ Low OCR confidence ({strat} strategy). "
                                "**Add a free Groq API key** to enable AI-powered fallback — "
                                "get one at [console.groq.com](https://console.groq.com) and set "
                                "`GROQ_API_KEY` in `.streamlit/secrets.toml`."
                            )
                        else:
                            st.error("⚠️ Low confidence scan. Please fill in values manually below.")

                    st.session_state["parsed_data"] = data

                    # ── Step 3: Manual entry form (last resort) ────────────────────
                    if pc in ("low", "medium") or missing:
                        ov = fallback_form(missing, data)
                        if st.button("✅ Apply Manual Values & Score", key="apply_btn"):
                            data = apply_overrides(data, {k: v for k, v in ov.items() if v is not None})
                            st.session_state["parsed_data"] = data

                    # ── Render analysis ────────────────────────────────────────────
                    try:
                        # User-typed name takes priority over OCR-detected name
                        if manual_name.strip():
                            data["product_name"] = manual_name.strip().title()
                        if scan_notes.strip():
                            data["notes"] = scan_notes.strip()
                        sr   = score_label(data, goal_profile, cat_cfg)
                        name = data.get("product_name") or "Scanned Product"
                        save_history(name, sr["total"], sr["grade"], sel_cat, data)
                        render_analysis(data, sr, goal_profile, user_profile, cat_cfg, sel_cat)
                    except Exception as e:
                        st.error(f"Analysis rendering failed: {e}")
                        st.info("Tips: high-res photo, flat angle, good lighting.")

            elif not uploaded:
                last = st.session_state.history[0] if st.session_state.history else None
                if last:
                    lc = P["good"] if last["score"]>=70 else (P["warn"] if last["score"]>=45 else P["bad"])
                    st.markdown(f"""
                    <div class="ns-card" style="margin-top:.5rem;">
                        <div class="sec-hdr">🕐 Last Scanned</div>
                        <div style="display:flex;align-items:center;gap:1.2rem;padding:.4rem 0;">
                            <div style="font-family:'Playfair Display',serif;font-size:2.8rem;color:{lc};font-weight:700;min-width:52px;text-align:center;">{last["grade"]}</div>
                            <div>
                                <div style="font-size:1rem;font-weight:600;color:{P["text"]};">{last["product"]}</div>
                                <div style="font-size:.78rem;color:{P["muted"]};margin:.1rem 0;">{last.get("category","")} · {last["timestamp"]}</div>
                                <div style="font-size:.85rem;color:{lc};font-weight:600;">{last["score"]}/100</div>
                                {f'<div style="font-size:.72rem;color:{P["muted"]};font-style:italic;margin-top:.2rem;">{last["notes"][:50]}</div>' if last.get("notes") else ""}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='text-align:center;padding:5rem 2rem;color:#666b8a;'>
                    <div style='font-size:3.5rem;margin-bottom:1rem;'>🔬</div>
                    <p style='font-size:.98rem;'>Your analysis will appear here</p>
                    <p style='font-size:.8rem;'>Upload a label and click Analyse to get started</p>
                </div>""",unsafe_allow_html=True)

    # ── BMI TAB ───────────────────────────────
    with tab_bmi:
        render_bmi(user_profile)

    with tab_dash:
        render_dashboard()

    st.markdown("---")
    st.markdown(f"<p style='text-align:center;color:{P['border']};font-size:.7rem;'>NutriScan v1.2 · Multi-strategy OCR · Local & Offline · Not a substitute for medical/dietary advice</p>",unsafe_allow_html=True)


if __name__ == "__main__":
    main()