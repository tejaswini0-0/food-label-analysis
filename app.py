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
import numpy as np
import re
import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go

# Auth module (local file-based; swap internals for Supabase in Phase 2)
from auth import (
    render_auth_page, render_auth_sidebar,
    persist_scan, render_dashboard,
)
from database import (
    save_scan_db, load_scans_db, register_user_db,
    update_profile_db, supabase_enabled,
)
from ocr_engine import _get_groq_key

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

/* ── MOBILE RESPONSIVE ─────────────────────────────────────── */
@media (max-width: 768px) {{
  /* Stack all columns vertically */
  [data-testid="column"] {{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }}
  /* Smaller padding on cards */
  .ns-card {{
    padding: 1rem 1rem !important;
  }}
  /* Grade ring smaller on mobile */
  .grade-ring {{
    width: 52px !important;
    height: 52px !important;
    line-height: 52px !important;
    font-size: 1.5rem !important;
  }}
  /* Tabs scrollable */
  [data-testid="stTabs"] [role="tablist"] {{
    overflow-x: auto;
    flex-wrap: nowrap;
  }}
  /* File uploader full width */
  [data-testid="stFileUploader"] {{
    width: 100% !important;
  }}
  /* Topbar profile row wraps */
  [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap !important;
  }}
  /* Buttons full width */
  div.stButton > button {{
    padding: .65rem 1rem !important;
    font-size: .88rem !important;
  }}
  /* Make number inputs smaller */
  input[type="number"] {{
    font-size: .85rem !important;
  }}
  /* Nutrient rows wrap */
  .nut-row {{
    font-size: .82rem !important;
  }}
}}
@media (max-width: 480px) {{
  .sec-hdr {{ font-size: 1rem !important; }}
  .tag {{ font-size: .68rem !important; padding: .12rem .4rem !important; }}
}}

/* Touch-friendly tap targets */
div.stButton > button {{
  min-height: 44px;
}}
[data-testid="stFileUploader"] label {{
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}}
/* Smooth scroll for mobile */
html {{ scroll-behavior: smooth; }}
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
# OCR ENGINE  (see ocr_engine.py)
# ─────────────────────────────────────────────
from ocr_engine import (
    run_ocr, parse_label, apply_overrides,
    groq_extract, groq_to_parsed_data, _get_groq_key,
    ocr_num, MICRO_KEYS,
)

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
    if trans is not None and trans > 0.2:  # ignore OCR noise / "0g" labels
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
        number={"font":{"size":48,"color":color,"family":"Playfair Display"},
                "valueformat":".1f"},
        domain={"x":[0,1],"y":[0,1]},
        gauge={"axis":{"range":[10,45],"tickcolor":P["muted"],"tickfont":{"color":P["muted"],"size":9}},
               "bar":{"color":color,"thickness":.22},"bgcolor":P["surface2"],"borderwidth":0,
               "steps":[{"range":[10,18.5],"color":"rgba(106,179,245,.13)"},
                        {"range":[18.5,25],"color":"rgba(92,184,138,.13)"},
                        {"range":[25,30],"color":"rgba(232,136,74,.13)"},
                        {"range":[30,45],"color":"rgba(217,85,85,.13)"}]}))
    fig.update_layout(
        height=260,
        margin=dict(l=40,r=40,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color":P["text"]},
        autosize=True,
    )
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
    # (Logging happens in database.py - save_scan_db() prints to terminal)
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

    # Settings expander — Groq key + Supabase status
    with st.expander("⚙️ Settings", expanded=False):
        if not _get_groq_key():
            runtime_key = st.text_input("Groq API key (optional — faster AI)", type="password",
                                        key="groq_rt_key", label_visibility="visible")
            if runtime_key and runtime_key.startswith("gsk_"):
                import os; os.environ["GROQ_API_KEY"] = runtime_key
                st.rerun()
        else:
            st.markdown("<span style='color:#5cb88a;font-size:.8rem;'>✓ Groq key active</span>", unsafe_allow_html=True)
        _sb_on = supabase_enabled()
        _sb_color = "#5cb88a" if _sb_on else "#666b8a"
        _sb_label = "✓ Supabase connected — scans sync to cloud" if _sb_on else "○ Supabase not configured — using local storage"
        st.markdown(f"<span style='color:{_sb_color};font-size:.8rem;'>{_sb_label}</span>", unsafe_allow_html=True)

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
# AI ADVISOR TAB
# ─────────────────────────────────────────────
def _groq_chat(messages: list, system: str, max_tokens: int = 800) -> str:
    """
    Calls Groq's llama-3.1-8b-instant model for fast, free text generation.
    Requires GROQ_API_KEY in .streamlit/secrets.toml or as env var.
    Free tier: 30 req/min, 14,400/day — no credit card needed.
    Get key at: https://console.groq.com
    """
    key = _get_groq_key()
    if not key:
        return (
            "🔑 **AI features need a free Groq API key.**\n\n"
            "1. Go to [console.groq.com](https://console.groq.com) — sign up free\n"
            "2. Create an API key (starts with `gsk_`)\n"
            "3. Add to `.streamlit/secrets.toml`:\n"
            "```\nGROQ_API_KEY = \"gsk_your_key_here\"\n```\n"
            "4. Restart the app — AI features will work instantly."
        )
    
    try:
        from groq import Groq
    except ImportError:
        return "⚠️ Groq SDK not installed. Run: pip install groq>=0.4.0"
    
    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast, free, and currently available
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "⚠️ Groq API key is invalid. Update it in ⚙️ Settings."
        elif "429" in error_msg:
            return "⚠️ Rate limit hit. Wait 1 minute and try again."
        elif "decommissioned" in error_msg:
            return "⚠️ The model is no longer available. Try again later."
        else:
            return f"⚠️ AI error: {error_msg[:100]}"


def render_ai_advisor(user_profile: dict, goal_profile: dict):
    """AI Advisor tab — chat, meal suggestions, weekly summary, label explainer."""
    from ocr_engine import _get_groq_key

    has_key = bool(_get_groq_key())
    last_scan = st.session_state.history[0] if st.session_state.history else None

    # Build user context string for all prompts
    w, h, age, sex = (user_profile.get("weight_kg",0), user_profile.get("height_cm",0),
                      user_profile.get("age",0), user_profile.get("sex","Male"))
    goal = user_profile.get("goal","Balanced")
    ctx  = f"User: {sex}, age {age}, {w}kg, {h}cm, goal: {goal}."
    if last_scan:
        ctx += (f" Last scanned: {last_scan['product']} "
                f"(score {last_scan['score']}/100, grade {last_scan['grade']}, "
                f"category {last_scan.get('category','')}).")
    if st.session_state.history:
        avg = round(sum(e["score"] for e in st.session_state.history) / len(st.session_state.history))
        ctx += f" Average scan score: {avg}/100 across {len(st.session_state.history)} scans."

    if not has_key:
        st.markdown(
            "<div style='background:rgba(240,165,0,.07);border:1px solid rgba(240,165,0,.2);"
            "border-radius:10px;padding:.8rem 1rem;font-size:.85rem;color:#eceef8;margin-bottom:.75rem;'>"
            "🔑 <b>Add a free Groq API key</b> to unlock AI features.<br>"
            "<span style='color:#666b8a;font-size:.78rem;'>"
            "Get one free at <a href='https://console.groq.com' target='_blank' style='color:#f0a500;'>console.groq.com</a>"
            " → paste in ⚙️ Settings above. No credit card needed.</span></div>",
            unsafe_allow_html=True
        )

    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
        "💬 Chat", "🍽️ Meal Ideas", "📅 Weekly Summary", "🔍 Explain My Scan"
    ])

    # ── Tab 1: Free Chat ────────────────────────────────────────────────────
    with ai_tab1:
        st.markdown(f'<div class="sec-hdr">💬 Ask Anything About Nutrition</div>', unsafe_allow_html=True)

        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []

        # Display chat history
        for msg in st.session_state.ai_chat_history:
            role_color = P["accent"] if msg["role"] == "user" else P["good"]
            role_label = "You" if msg["role"] == "user" else "NutriScan AI"
            st.markdown(
                f"<div style='margin:.5rem 0;padding:.7rem 1rem;border-radius:10px;"
                f"background:{P['surface2']};border-left:3px solid {role_color};'>"
                f"<span style='font-size:.72rem;color:{role_color};font-weight:600;"
                f"letter-spacing:.05em;'>{role_label}</span>"
                f"<p style='margin:.25rem 0 0;font-size:.88rem;color:{P['text']};'>{msg['content']}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

        user_q = st.text_input("Ask about ingredients, nutrients, diet tips…",
                               placeholder="e.g. Is 37g carbs too much for dinner?",
                               key="ai_chat_input", label_visibility="collapsed")
        c1, c2 = st.columns([4, 1])
        with c2:
            send = st.button("Send →", key="ai_send", use_container_width=True)
        with c1:
            if st.button("Clear chat", key="ai_clear"):
                st.session_state.ai_chat_history = []
                st.rerun()

        if send and user_q.strip():
            st.session_state.ai_chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Thinking…"):
                system = (
                    "You are NutriScan AI, a concise, friendly nutrition expert. "
                    "Give practical, evidence-based advice. Keep responses under 150 words. "
                    "Never recommend medical treatment. Always be encouraging. "
                    + ctx
                )
                reply = _groq_chat(st.session_state.ai_chat_history, system)
            st.session_state.ai_chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # ── Tab 2: Meal Ideas ───────────────────────────────────────────────────
    with ai_tab2:
        st.markdown(f'<div class="sec-hdr">🍽️ Personalised Meal Ideas</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            meal_type = st.selectbox("Meal", ["Breakfast","Lunch","Dinner","Snack"], key="meal_type")
            cuisine   = st.selectbox("Cuisine preference", ["Any","Indian","Mediterranean","Asian","Western","Mexican"], key="meal_cuisine")
        with m2:
            avoid     = st.text_input("Avoid / allergies", placeholder="e.g. gluten, dairy, nuts", key="meal_avoid")
            cal_target= st.number_input("Calorie target (kcal)", 0, 2000, 500, 50, key="meal_cal")

        if st.button("✨ Generate Meal Ideas", key="meal_gen", use_container_width=True):
            with st.spinner("Crafting meal ideas…"):
                prompt = (
                    f"Suggest 3 {meal_type} meal ideas for someone with goal: {goal}. "
                    f"Cuisine: {cuisine}. Avoid: {avoid or 'nothing'}. "
                    f"Target ~{cal_target} kcal per meal. "
                    f"For each: name, ~calories, key macros, and why it fits the goal. "
                    f"Format as a numbered list. Keep it concise and practical."
                )
                system = "You are a friendly dietitian specialising in practical meal planning. " + ctx
                reply  = _groq_chat([{"role": "user", "content": prompt}], system, max_tokens=600)
            st.markdown(
                f"<div class='ns-card' style='margin-top:.5rem;'>{reply.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )

    # ── Tab 3: Weekly Summary ───────────────────────────────────────────────
    with ai_tab3:
        st.markdown(f'<div class="sec-hdr">📅 Weekly Health Summary</div>', unsafe_allow_html=True)

        history = st.session_state.history
        if len(history) < 2:
            st.info("Scan at least 2 products to generate a weekly summary.")
        else:
            recent = history[:min(20, len(history))]
            scan_list = chr(10).join("- "+e["product"]+" | Score "+str(e["score"])+"/100 | Grade "+e["grade"]+" | "+e.get("category","")+" | "+str(e.get("calories","?"))+" kcal" for e in recent)
            if st.button("📊 Generate My Health Summary", key="summary_gen", use_container_width=True):
                with st.spinner("Analysing your scan history…"):
                    prompt = (
                        "Here are the user's recent food scans:\n" + scan_list + "\n\n"
                        "Write a brief, encouraging weekly nutrition summary. Include:\n"
                        "1. Overall pattern (2 sentences)\n"
                        "2. Top 2 strengths\n"
                        "3. Top 2 areas to improve\n"
                        "4. One actionable tip for next week\n"
                        "Keep it under 200 words. Be positive and specific."
                    )
                    system = "You are a supportive nutrition coach reviewing a client's food diary. " + ctx
                    reply  = _groq_chat([{"role": "user", "content": prompt}], system, max_tokens=400)
                st.markdown(
                    f"<div class='ns-card'>{reply.replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )

            # Show score trend mini chart
            if len(recent) >= 3:
                import plotly.graph_objects as pgo
                scores  = [e["score"] for e in reversed(recent)]
                colours = [P["good"] if s>=70 else (P["warn"] if s>=45 else P["bad"]) for s in scores]
                fig = pgo.Figure(pgo.Bar(y=scores, marker_color=colours,
                                        hovertemplate="%{y}/100<extra></extra>"))
                fig.update_layout(
                    height=160, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showticklabels=False), yaxis=dict(range=[0,105], color=P["muted"]),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 4: Explain My Scan ──────────────────────────────────────────────
    with ai_tab4:
        st.markdown(f'<div class="sec-hdr">🔍 Explain My Last Scan</div>', unsafe_allow_html=True)

        if not last_scan:
            st.info("Scan a product first to get an AI explanation.")
        else:
            parsed = st.session_state.get("parsed_data")
            nutrients_txt = ""
            if parsed:
                for k, v in parsed.get("nutrients", {}).items():
                    if isinstance(v, dict):
                        nutrients_txt += f"{k.replace('_',' ').title()}: {v.get('value')} {v.get('unit')}  "

            explain_mode = st.radio("What do you want explained?",
                                    ["Plain English summary", "Is this healthy for me?",
                                     "What are the concerning ingredients?",
                                     "How does this fit my daily goal?"],
                                    key="explain_mode", horizontal=True)

            if st.button("🤖 Explain Now", key="explain_btn", use_container_width=True):
                with st.spinner("Analysing…"):
                    prompts = {
                        "Plain English summary":
                            f"Explain this nutrition label in simple plain English for someone with no nutrition knowledge. "
                            f"Product: {last_scan['product']}. Nutrients: {nutrients_txt}. Score: {last_scan['score']}/100.",
                        "Is this healthy for me?":
                            f"Based on my profile and goal ({goal}), is {last_scan['product']} a good choice? "
                            f"Nutrients: {nutrients_txt}. Score: {last_scan['score']}/100. Give a direct yes/no with brief reasoning.",
                        "What are the concerning ingredients?":
                            f"What should I be concerned about in {last_scan['product']}? "
                            f"Nutrients: {nutrients_txt}. Focus on anything above recommended daily limits.",
                        "How does this fit my daily goal?":
                            f"I have a {goal} goal. I just ate {last_scan['product']} ({last_scan.get('calories','?')} kcal). "
                            f"How does this fit into my day? What should I eat for my remaining meals? "
                            f"Nutrients: {nutrients_txt}.",
                    }
                    system = "You are a concise, friendly nutritionist. Keep answers under 180 words. Be specific and practical. " + ctx
                    reply  = _groq_chat([{"role": "user", "content": prompts[explain_mode]}], system, max_tokens=400)
                st.markdown(
                    f"<div class='ns-card'>"
                    f"<div class='sec-hdr'>💡 {explain_mode}</div>"
                    f"{reply.replace(chr(10), '<br>')}"
                    f"</div>",
                    unsafe_allow_html=True
                )


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

    tab_scan, tab_bmi, tab_dash, tab_ai = st.tabs(["🔬 Scan Label","⚖️ BMI & Energy","📊 My Dashboard","🤖 AI Advisor"])

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
            # ── Run OCR when Analyse button clicked ───────────────────────────
            if uploaded and analyse:
                with st.spinner("🔍 Scanning label…"):
                    try:
                        ocr_res = run_ocr(pil_img)
                    except Exception as e:
                        st.error(f"OCR engine failed: {e}")
                        ocr_res = []

                if not ocr_res:
                    st.error("OCR returned no text. Try a clearer, well-lit photo.")
                else:
                    data    = parse_label(ocr_res)
                    pc      = data.get("parse_confidence", "low")
                    missing = data.get("missing_nutrients", [])

                    # Groq vision fallback if confidence is low
                    if (pc in ("low", "medium") or missing) and bool(_get_groq_key()):
                        with st.spinner("Enhancing with AI…"):
                            groq_json = groq_extract(pil_img)
                            if groq_json:
                                data    = groq_to_parsed_data(groq_json, data)
                                pc      = data.get("parse_confidence", "low")
                                missing = data.get("missing_nutrients", [])

                    # Apply user-typed name/notes before storing
                    if manual_name.strip():
                        data["product_name"] = manual_name.strip().title()
                    if scan_notes.strip():
                        data["notes"] = scan_notes.strip()

                    # Store everything in session_state so reruns keep the result
                    st.session_state["parsed_data"]    = data
                    st.session_state["parsed_pc"]      = pc
                    st.session_state["parsed_missing"] = missing
                    st.session_state["parsed_cat"]     = sel_cat
                    st.session_state["parsed_cat_cfg"] = cat_cfg
                    st.session_state["scan_saved"]     = False  # reset so history saves once

            # ── Render from session_state (survives Apply button reruns) ─────
            data     = st.session_state.get("parsed_data")
            pc       = st.session_state.get("parsed_pc", "low")
            missing  = st.session_state.get("parsed_missing", [])
            r_cat    = st.session_state.get("parsed_cat", sel_cat)
            r_catcfg = st.session_state.get("parsed_cat_cfg", cat_cfg)

            if data:
                strat = st.session_state.get("ocr_strategy", "—")

                # Confidence banner
                prod_name = data.get("product_name") or "Scanned Product"
                if pc == "high":
                    st.success(f"✅ High confidence · {prod_name} · OCR: {strat}")
                elif pc == "medium":
                    st.warning(f"⚠️ Partial parse — fill in missing values below if needed.")
                else:
                    if not bool(_get_groq_key()):
                        st.error("⚠️ Low confidence. Fill in values below or add a Groq key for AI-powered extraction.")
                    else:
                        st.error("⚠️ Low confidence. Fill in values manually below.")

                # Manual entry form — stays visible across reruns
                if pc in ("low", "medium") or missing:
                    ov = fallback_form(missing, data)
                    if st.button("✅ Apply Manual Values & Score", key="apply_btn"):
                        data = apply_overrides(data, {k: v for k, v in ov.items() if v is not None})
                        st.session_state["parsed_data"]    = data
                        st.session_state["parsed_pc"]      = data.get("parse_confidence","low")
                        st.session_state["parsed_missing"] = data.get("missing_nutrients",[])
                        st.session_state["scan_saved"]     = False
                        st.rerun()

                # Render analysis + save history (only once per scan)
                try:
                    sr = score_label(data, goal_profile, r_catcfg)
                    if not st.session_state.get("scan_saved", False):
                        name = data.get("product_name") or "Scanned Product"
                        save_history(name, sr["total"], sr["grade"], r_cat, data)
                        st.session_state["scan_saved"] = True
                    render_analysis(data, sr, goal_profile, user_profile, r_catcfg, r_cat)
                except Exception as e:
                    st.error(f"Analysis rendering failed: {e}")
                    st.info("Tips: high-res photo, flat angle, good lighting.")

            elif not uploaded:
                last = st.session_state.history[0] if st.session_state.history else None
                if last:
                    lc       = P["good"] if last["score"]>=70 else (P["warn"] if last["score"]>=45 else P["bad"])
                    _txt     = P["text"]
                    _muted   = P["muted"]
                    _grade   = last["grade"]
                    _product = last["product"]
                    _cat     = last.get("category","")
                    _ts      = last["timestamp"]
                    _score   = last["score"]
                    _notes_h = f"<div style='font-size:.72rem;color:{_muted};font-style:italic;margin-top:.2rem;'>{last['notes'][:50]}</div>" if last.get("notes") else ""
                    st.markdown(
                        f"<div class='ns-card' style='margin-top:.5rem;'>"
                        f"<div class='sec-hdr'>🕐 Last Scanned</div>"
                        f"<div style='display:flex;align-items:center;gap:1.2rem;padding:.4rem 0;'>"
                        f"<div style='font-family:Playfair Display,serif;font-size:2.8rem;color:{lc};font-weight:700;min-width:52px;text-align:center;'>{_grade}</div>"
                        f"<div>"
                        f"<div style='font-size:1rem;font-weight:600;color:{_txt};'>{_product}</div>"
                        f"<div style='font-size:.78rem;color:{_muted};margin:.1rem 0;'>{_cat} · {_ts}</div>"
                        f"<div style='font-size:.85rem;color:{lc};font-weight:600;'>{_score}/100</div>"
                        f"{_notes_h}"
                        f"</div></div></div>",
                        unsafe_allow_html=True
                    )
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

    with tab_ai:
        render_ai_advisor(user_profile, goal_profile)

    st.markdown("---")
    st.markdown(f"<p style='text-align:center;color:{P['border']};font-size:.7rem;'>NutriScan v1.2 · Multi-strategy OCR · Not a substitute for medical/dietary advice</p>",unsafe_allow_html=True)


if __name__ == "__main__":
    main()