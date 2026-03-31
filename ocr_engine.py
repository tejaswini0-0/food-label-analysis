"""
ocr_engine.py — NutriScan OCR & Parsing Engine
================================================
Handles:
  - Image preprocessing (6 strategies, fast-path early exit)
  - EasyOCR execution
  - Groq vision fallback
  - 3-pass label parser (regex + positional row-pair + joined fallback)
  - Number extraction with g→9 / 0g→09 correction
"""

import re
import os
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MICRO_KEYS = {"vitamin_d", "calcium", "iron", "potassium"}

_UNIT_MAP = {
    "calories": "kcal", "total_fat": "g", "saturated_fat": "g", "trans_fat": "g",
    "cholesterol": "mg", "sodium": "mg", "carbohydrates": "g", "fiber": "g",
    "sugar": "g", "added_sugar": "g", "protein": "g",
    "vitamin_d": "mcg", "calcium": "mg", "iron": "mg", "potassium": "mg",
}

# ─────────────────────────────────────────────
# OCR READER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


# ─────────────────────────────────────────────
# PREPROCESSING STRATEGIES
# ─────────────────────────────────────────────
def _upscale(arr: np.ndarray, min_dim: int = 1200) -> np.ndarray:
    """Upscale image so smallest label text is large enough for EasyOCR.
    Reduced to 1200 from 1800 for ~30% speed improvement (~8-10s down to 5-6s per pass).
    """
    h, w = arr.shape[:2]
    if max(h, w) < min_dim:
        s = min_dim / max(h, w)
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
    return arr


def _hsv_value(arr: np.ndarray) -> np.ndarray:
    """★ Best for coloured backgrounds (FDA labels). V channel ignores hue."""
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v   = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v   = clahe.apply(v)
    v   = cv2.fastNlMeansDenoising(v, h=8)
    _, b = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def _lab_lum(arr: np.ndarray) -> np.ndarray:
    """CLAHE on LAB L-channel. Good for tinted/pastel backgrounds."""
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    _, b2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b2, cv2.COLOR_GRAY2RGB)


def _adaptive(arr: np.ndarray) -> np.ndarray:
    """CLAHE + adaptive threshold. Best for uneven lighting."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=12)
    b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 9)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def _otsu(arr: np.ndarray) -> np.ndarray:
    """Gaussian + Otsu. Best for clean high-contrast labels."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def _fixed_thresh(arr: np.ndarray) -> np.ndarray:
    """
    Fixed threshold at 180 — preserves bold/thick strokes that Otsu merges.
    On a standard black-on-white FDA label: Otsu picks ~127 which can fill
    in the thick counters of bold letters. 180 keeps them open.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Mild sharpening to enhance stroke edges before threshold
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray   = cv2.filter2D(gray, -1, kernel)
    gray   = np.clip(gray, 0, 255).astype(np.uint8)
    _, b   = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


def _dilate_thin(arr: np.ndarray) -> np.ndarray:
    """
    Slight dilation before threshold — recovers thin strokes that denoise eroded.
    Good for labels photographed at slight angles where thin text gets compressed.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=6)
    # Dilate slightly to thicken strokes before thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray   = cv2.dilate(gray, kernel, iterations=1)
    _, b   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────
# OCR SCORER & RUNNER
# ─────────────────────────────────────────────
_SCORE_KW = ["sodium", "carbohydrate", "protein", "calorie", "fat", "fiber", "sugar"]
_FAST_KW_THRESHOLD = 4  # accept immediately if this many keywords found


def _ocr_score(results: List[Tuple]) -> float:
    if not results:
        return 0.
    total_chars = sum(len(t) for _, t, _ in results)
    avg_conf    = sum(c for _, _, c in results) / len(results)
    base        = total_chars * avg_conf
    txt         = " ".join(t.lower() for _, t, _ in results)
    bonus       = sum(0.15 for kw in _SCORE_KW if kw in txt)
    return base * (1 + bonus)


def _kw_count(results: List[Tuple]) -> int:
    txt = " ".join(t.lower() for _, t, _ in results)
    return sum(1 for kw in _SCORE_KW if kw in txt)


def run_ocr(pil_img: Image.Image) -> List[Tuple]:
    """
    2-call maximum OCR for ~5s target time.

    Pass 1 — fixed_thresh (preserves bold strokes, works on B&W FDA labels).
              If ≥4 keywords found → done. This handles ~80% of labels.
    Pass 2 — hsv_value (coloured background labels like the FDA sample).
              Only runs if Pass 1 found <4 keywords.
    Fallback — raw image. Only if both passes returned nothing.

    Each EasyOCR call on a 1200px image takes ~5-6s on CPU (down from 8-12s at 1800px).
    2 calls max = ~10-12s worst case. 1 call = ~5s best case.
    """
    reader = load_reader()
    arr    = _upscale(np.array(pil_img.convert("RGB")))

    best_results: List[Tuple] = []
    best_score = 0.
    best_name  = "none"

    def _try(name: str, fn) -> Tuple[List, float, int]:
        nonlocal best_results, best_score, best_name
        try:
            processed = fn(arr)
            results   = reader.readtext(processed, detail=1, paragraph=False)
            sc  = _ocr_score(results)
            kwc = _kw_count(results)
            if sc > best_score:
                best_score, best_results, best_name = sc, results, name
            return results, sc, kwc
        except Exception:
            return [], 0., 0

    # Pass 1: fixed threshold — bold-preserving, fast
    _, _, kwc1 = _try("fixed_thresh", _fixed_thresh)
    if kwc1 >= 4:
        st.session_state["ocr_strategy"] = best_name
        return best_results

    # Pass 2: HSV value channel — coloured backgrounds
    _try("hsv_value", _hsv_value)
    if best_results:
        st.session_state["ocr_strategy"] = best_name
        return best_results

    # Fallback: raw image
    _try("raw", lambda a: a)
    st.session_state["ocr_strategy"] = best_name
    return best_results


# ─────────────────────────────────────────────
# NUMBER EXTRACTION  (g→9 aware)
# ─────────────────────────────────────────────
# EasyOCR commonly misreads 'g' as '9' in tokens like '46g' → '469',
# '0g' → '09', '3g' → '39'. We detect and correct this before parsing.

# Translation for letter→digit OCR errors (NOT 'g' — handled separately)
_FIX = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1", "|": "1",
    "B": "8",
    "Z": "2",
    # S→5 intentionally excluded: breaks Sodium/Sugar/Saturated
    # g→9 intentionally excluded: handled by _fix_g9 below
})

# Plausible per-nutrient value ranges (grams unless noted)
_SANE = {
    "calories":      (0, 1500),   # kcal
    "total_fat":     (0, 100),    # g
    "saturated_fat": (0, 60),
    "trans_fat":     (0, 10),
    "cholesterol":   (0, 600),    # mg
    "sodium":        (0, 3000),   # mg
    "carbohydrates": (0, 200),    # g
    "fiber":         (0, 60),
    "sugar":         (0, 150),
    "added_sugar":   (0, 150),
    "protein":       (0, 100),
    "vitamin_d":     (0, 200),    # mcg
    "calcium":       (0, 2000),   # mg
    "iron":          (0, 100),    # mg
    "potassium":     (0, 5000),   # mg
}


def _fix_g9(token: str) -> str:
    """
    Correct EasyOCR's g→9 misread.
    Handles both integer tokens ("469"→"46") and decimal tokens ("1.59"→"1.5").

    Integer rule: pure digits ending in 9, where dropping the 9 gives ≤300 → drop it.
    Decimal rule: decimal string ending in 9 after decimal point ("1.59"→"1.5") → drop trailing 9.
      Rationale: nutrient gram values rarely have 2 meaningful decimal places.

    Examples:
        "469"  → "46"    (46g)
        "1.59" → "1.5"   (1.5g saturated fat)
        "09"   → "0"     (0g trans fat)
        "119"  → "11"    (11g protein)
        "430"  → "430"   (430mg sodium — not dropped, > 300)
        "1.23" → "1.23"  (no trailing 9 — unchanged)
    """
    t = token.strip()

    # Decimal case: "X.Y9" → "X.Y" (drop trailing 9 after decimal)
    dm = re.match(r'^(\d+\.\d+)9$', t)
    if dm:
        candidate = dm.group(1)
        try:
            v = float(candidate)
            if v <= 300:
                return candidate
        except ValueError:
            pass

    # Integer case: pure digits ending in 9, dropping gives ≤300
    if re.match(r'^\d+9$', t) and len(t) >= 2:
        candidate = t[:-1]
        try:
            v = float(candidate)
            if v <= 300:
                return candidate
        except ValueError:
            pass

    return t


def ocr_num(token: str, nutrient_key: str = "") -> Optional[float]:
    """
    Convert an OCR token to a float, handling common misreads.
    Steps:
      1. Strip trailing unit letters (mg, g, kcal, etc.)
      2. Apply g→9 correction on what remains
      3. Apply letter→digit fixes
      4. Extract digits
      5. Sanity-check against per-nutrient plausible range
    """
    t = token.strip()

    # Step 1: strip known unit suffixes from the end
    t = re.sub(r'(?i)\s*(mg|mcg|µg|kcal|cal|ml|oz|g)\s*$', '', t).strip()

    # Step 2: g→9 correction
    t = _fix_g9(t)

    # Step 3: letter→digit fixes
    t = t.translate(_FIX)

    # Step 4: keep only digits and decimal point
    c = re.sub(r'[^0-9.]', '', t)
    parts = c.split('.')
    if len(parts) > 2:
        c = parts[0] + '.' + ''.join(parts[1:])

    if not c:
        return None

    try:
        v = float(c)
    except ValueError:
        return None

    # Step 5: sanity check
    if v < 0 or v >= 1e5:
        return None
    if nutrient_key and nutrient_key in _SANE:
        lo, hi = _SANE[nutrient_key]
        if not (lo <= v <= hi):
            return None

    return v


# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────
# Value pattern: number followed by required unit.
# (?!\\s*%) lookahead rejects % Daily Value numbers.
_V  = r'(?P<val>[\d.]+)\s*(?P<unit>g|mg)'
_VK = r'(?P<val>[\d.]+)\s*(?P<unit>kcal|cal)'
_VI = r'(?P<val>[\d.]+)\s*(?P<unit>mcg|µg|iu|mg|g)'

# ── Fuzzy value pattern: also match when unit is absent (g→9 fixes unit later)
_VF = r"(?P<val>[\d.]+)\s*(?P<unit>g|mg)?"   # unit optional — for garbled labels

NPATS: Dict[str, List[str]] = {
    "serving_size":           [r"serving\s*size[\s:]+(?P<val>[\d.]+)\s*(?P<unit>g|ml|oz)",
                               r"serving\s*size[\s:]+\d+\s+\w+\s*\((?P<val>[\d.]+)\s*(?P<unit>g|ml)\)"],
    "servings_per_container": [r"(?:about\s*)?(?P<val>[\d.]+)\s*servings?\s*per\s*container",
                               r"servings?\s*per\s*container[\s:]+(?P<val>[\d.]+)"],
    # Calories: number can appear BEFORE or AFTER the keyword (garbled labels)
    "calories":               [r"calories[\s:]*(?P<val>\d{2,4})(?!\d)",        # "Calories 240"
                               r"(?P<val>\d{2,4})\s*calories",                   # "240 calories"
                               r"energy[\s:]+(?P<val>\d{2,4})\s*(?:kcal|cal)",
                               r"^(?P<val>\d{2,4})\s*(?:cal(?:ories)?)?$"],
    "total_fat":              [r"total\s*fat[\s:]*" + _V,
                               r"tet\S{0,3}\s*fat[\s:]*" + _V,                 # "Tet_! Fat"
                               r"tet\S{0,3}\s*fat[\s:]*(?P<val>[\d.]+)(?P<unit>g?)",
                               r"fat,?\s*total[\s:]+" + _V],
    # Saturated fat: value may end in 9 (g→9), e.g. "1.59" means "1.5g"
    "saturated_fat":          [r"sat\w{0,8}\s*fat[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?"],
    "trans_fat":              [r"tr[ae]ns\s*fat[\s:]*" + _V,
                               r"tr\w{0,4}\s*fat[\s:]*(?P<val>[\d.]+)(?P<unit>g?)"],
    "cholesterol":            [r"chol\w{0,8}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mg|g)"],
    # Sodium: "sedtum 430m9" — match digits followed by m (mg garbled)
    "sodium":                 [r"sodium[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mg|g)",
                               r"s[eo]d\w{0,5}[\s:]*(?P<val>[\d.]+)\s*m",    # "sedtum 430m"
                               r"salt[\s:]+" + _V],
    # Carbohydrates: "carbehydrato 469" — 469 is "46g" (g→9 fix applied in extr)
    "carbohydrates":          [r"total\s*carb\w{0,12}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",
                               r"carb\w{0,12}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",
                               r"hydrate[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",  # catches "hydrate" if "carbo" missed
                               r"(?:^|\n|\s)carbs?[\s:]+" + _V,
                               r"(?:^|\n|\s)carbs?[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?$"],  # carbs at end of line
    # Fiber: "Fiber 79" → "Fiber 7g" after g→9 fix
    "fiber":                  [r"dietary\s*fi\w{0,5}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",
                               r"fi\w{0,5}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?"],
    "sugar":                  [r"total\s*sug\w{0,4}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",
                               r"(?<!added\s)sug\w{0,4}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?"],
    "added_sugar":            [r"added\s*sug\w{0,4}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?",
                               r"includes?\s+(?P<val>[\d.]+)\s*g\s*added"],
    # Protein: "Protoln 119" → "11g" after g→9 fix
    "protein":                [r"prot\w{0,6}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>g|mg)?"],
    # Vitamin D: "Vllamln D Zmcg" — Z is garbled 2
    "vitamin_d":              [r"vit\w{0,6}\s*d[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mcg|µg|iu|mg|g)",
                               r"vl\w{0,6}\s*d[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mcg|µg|iu|mg|g)"],
    # Calcium: "Cakcium 260mng" — mng is garbled mg
    "calcium":                [r"ca[kl]\w{0,6}[\s:]*(?P<val>[\d.]+)\s*m",    # "cakcium 260m"
                               r"calcium[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mg|g)"],
    "iron":                   [r"iron[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mg|g)",
                               r"iron[\s:]*(?P<val>[\d.]+)\s*m"],
    # Potassium: "potasslum 240mg" — double-l
    "potassium":              [r"pot\w{0,7}[\s:]*(?P<val>[\d.]+)\s*(?P<unit>mg|g)",
                               r"pot\w{0,7}[\s:]*(?P<val>[\d.]+)\s*m"],
}

DV_PATS = {
    "total_fat":     r"total\s*fat[^%\n]{0,25}?(\d+)\s*%",
    "saturated_fat": r"saturated\s*fat[^%\n]{0,25}?(\d+)\s*%",
    "cholesterol":   r"cholesterol[^%\n]{0,25}?(\d+)\s*%",
    "sodium":        r"sodium[^%\n]{0,25}?(\d+)\s*%",
    "carbohydrates": r"total\s*carbohydrate[^%\n]{0,25}?(\d+)\s*%",
    "fiber":         r"(?:dietary\s*)?fi(?:ber|bre)[^%\n]{0,25}?(\d+)\s*%",
    "protein":       r"protein[^%\n]{0,25}?(\d+)\s*%",
    "vitamin_d":     r"vitamin\s*d[^%\n]{0,25}?(\d+)\s*%",
    "calcium":       r"calcium[^%\n]{0,25}?(\d+)\s*%",
    "iron":          r"iron[^%\n]{0,25}?(\d+)\s*%",
    "potassium":     r"potassium[^%\n]{0,25}?(\d+)\s*%",
}


def extr(text: str, pats: List[str], key: str = "") -> Tuple[Optional[float], str]:
    for pat in pats:
        m = re.search(pat, text, re.I | re.M)
        if m:
            gd      = m.groupdict()
            raw_val = gd.get("val", "")
            # Apply g→9 correction before ocr_num (handles "469"→"46", "79"→"7")
            raw_val = _fix_g9(raw_val)
            val     = ocr_num(raw_val, key)
            unit    = (gd.get("unit") or "").strip().lower()
            if val is not None:
                if not unit:
                    unit = "mg" if "mg" in pat.lower() else "g"
                return val, unit
    return None, ""


# ─────────────────────────────────────────────
# POSITIONAL ROW-PAIR PARSER
# ─────────────────────────────────────────────
_KW_MAP = {
    "energy":              "calories",
    "calories":            "calories",
    "calorie":             "calories",
    "kcal":                "calories",
    "total fat":           "total_fat",
    "fat":                 "total_fat",
    "saturated fat":       "saturated_fat",
    "sat. fat":            "saturated_fat",
    "sat fat":             "saturated_fat",
    "trans fat":           "trans_fat",
    "trans":               "trans_fat",
    "cholesterol":         "cholesterol",
    "sodium":              "sodium",
    "salt":                "sodium",
    "carbohydrate":        "carbohydrates",
    "carbohydrates":       "carbohydrates",
    "total carbohydrate":  "carbohydrates",
    "total carbohydrates": "carbohydrates",
    "carbs":               "carbohydrates",
    "carb":                "carbohydrates",
    "hydrate":             "carbohydrates",  # fallback if "carbo" missed
    "carbo":               "carbohydrates",  # partial match
    "dietary fiber":       "fiber",
    "dietary fibre":       "fiber",
    "fiber":               "fiber",
    "fibre":               "fiber",
    "total sugars":        "sugar",
    "sugars":              "sugar",
    "sugar":               "sugar",
    "added sugars":        "added_sugar",
    "added sugar":         "added_sugar",
    "protein":             "protein",
    "vitamin d":           "vitamin_d",
    "calcium":             "calcium",
    "iron":                "iron",
    "potassium":           "potassium",
}


def _cy(bbox) -> float:
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2.0


def _cx(bbox) -> float:
    xs = [pt[0] for pt in bbox]
    return (min(xs) + max(xs)) / 2.0


def _group_rows(sorted_res: List[Tuple], tol_ratio: float = 0.012) -> List[List[Tuple]]:
    if not sorted_res:
        return []
    img_h = max(_cy(b) for b, t, c in sorted_res) or 1
    tol   = max(8, img_h * tol_ratio)
    rows, cur, cur_y = [], [sorted_res[0]], _cy(sorted_res[0][0])
    for item in sorted_res[1:]:
        y = _cy(item[0])
        if abs(y - cur_y) <= tol:
            cur.append(item)
        else:
            rows.append(sorted(cur, key=lambda x: _cx(x[0])))
            cur, cur_y = [item], y
    if cur:
        rows.append(sorted(cur, key=lambda x: _cx(x[0])))
    return rows


def _nutrient_in_row(row: List[Tuple]) -> Optional[str]:
    text = " ".join(t.lower() for _, t, _ in row)
    for kw in sorted(_KW_MAP, key=len, reverse=True):
        if kw in text:
            return _KW_MAP[kw]
    return None


def _number_in_row(row: List[Tuple], nut_key: str = "") -> Tuple[Optional[float], str]:
    """
    Extract first value+unit from row, requiring explicit unit (g/mg/kcal).
    Also applies g→9 correction on each token individually before joining.
    """
    # Try each token individually first (catches '469' → '46' per token)
    for _, token, _ in row:
        t = token.strip()
        # Direct token like "46g", "160mg", "3g"
        m = re.match(r'^([\d.]+)\s*(g|mg|mcg|kcal|cal)$', t, re.I)
        if m:
            v = ocr_num(m.group(1) + m.group(2), nut_key)
            if v is not None:
                return v, m.group(2).lower()
        # Token is pure digits possibly with g→9 error
        if re.match(r'^\d+9$', t):
            fixed = _fix_g9(t)
            if fixed != t:  # correction was applied
                v = ocr_num(fixed, nut_key)
                if v is not None:
                    return v, _UNIT_MAP.get(nut_key, "g")

    # Fallback: join row and find value+unit pattern
    row_text = " ".join(t for _, t, _ in row)
    m = re.search(r'(?<!\d)([\d]+(?:[.,]\d+)?)\s*(g|mg|mcg|µg|kcal|cal)(?!\d)',
                  row_text, re.I)
    if m:
        raw = m.group(1).replace(",", ".")
        v   = ocr_num(raw, nut_key)
        u   = m.group(2).lower()
        if v is not None:
            # Sanity cap
            if u == "g"    and v > 300: return None, ""
            if u == "mg"   and v > 5000: return None, ""
            if u == "kcal" and v > 2000: return None, ""
            return v, u
    return None, ""


def _positional_parse(rows: List[List[Tuple]], out: Dict) -> Dict:
    n = len(rows)
    for i, row in enumerate(rows):
        nut_key = _nutrient_in_row(row)
        if nut_key is None:
            continue
        bucket = "micronutrients" if nut_key in MICRO_KEYS else "nutrients"
        if nut_key in out[bucket]:
            continue

        val, unit = _number_in_row(row, nut_key)
        if val is None and i + 1 < n:
            val, unit = _number_in_row(rows[i + 1], nut_key)

        if val is not None:
            if not unit:
                unit = _UNIT_MAP.get(nut_key, "g")
            out[bucket][nut_key] = {"value": val, "unit": unit, "source": "positional"}
    return out


# ─────────────────────────────────────────────
# MAIN PARSER  (3-pass)
# ─────────────────────────────────────────────
def parse_label(ocr_res: List[Tuple]) -> Dict[str, Any]:
    """
    Pass 1 — regex on newline-joined text
    Pass 2 — positional row-pair matching  (fixes coloured-label token bleed)
    Pass 3 — regex on space-joined text    (line-wrap fallback)
    """
    sorted_res = sorted(ocr_res, key=lambda r: _cy(r[0]))
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

    # Pass 1 — regex on newline-structured text (primary)
    for key, pats in NPATS.items():
        val, unit = extr(full_l, pats, key)
        if val is not None:
            bucket = "micronutrients" if key in MICRO_KEYS else "nutrients"
            out[bucket][key] = {"value": val, "unit": unit, "source": "regex"}

    # Pass 2 — regex on space-joined text (catches line-wrap OCR tokens)
    for key, pats in NPATS.items():
        bucket = "micronutrients" if key in MICRO_KEYS else "nutrients"
        if key not in out[bucket]:
            val, unit = extr(join_l, pats, key)
            if val is not None:
                out[bucket][key] = {"value": val, "unit": unit, "source": "joined"}

    # Positional pass intentionally disabled — regex is more reliable on real labels.

    # Daily values
    for key, pat in DV_PATS.items():
        m = re.search(pat, full_l, re.I)
        if m:
            dv = ocr_num(m.group(1))
            if dv is not None:
                out["daily_values"][key] = dv

    # Product name
    nf = full_l.find("nutrition facts")
    if nf > 0:
        lines = [l.strip() for l in full[:nf].split("\n") if l.strip() and len(l.strip()) > 2]
        if lines:
            out["product_name"] = lines[-1].title()
    elif hi_conf:
        for t in hi_conf[:5]:
            if not re.match(r"^\d", t) and len(t) > 3:
                out["product_name"] = t.strip().title()
                break

    # Confidence
    core  = {"calories", "protein", "carbohydrates", "total_fat", "sodium"}
    found = core & out["nutrients"].keys()
    out["missing_nutrients"] = list(core - found)
    out["parse_confidence"]  = (
        "high"   if len(found) >= 4 else
        "medium" if len(found) >= 2 else "low"
    )
    return out


def apply_overrides(data: Dict, ov: Dict[str, Optional[float]]) -> Dict:
    umap = {"calories": "kcal", "total_fat": "g", "saturated_fat": "g",
            "trans_fat": "g", "carbohydrates": "g", "fiber": "g",
            "sugar": "g", "protein": "g", "sodium": "mg"}
    for k, v in ov.items():
        if v is not None and v >= 0:
            bucket = "micronutrients" if k in MICRO_KEYS else "nutrients"
            data[bucket][k] = {"value": float(v), "unit": umap.get(k, "g"), "manual": True}
    core  = {"calories", "protein", "carbohydrates", "total_fat", "sodium"}
    found = core & data["nutrients"].keys()
    data["missing_nutrients"] = list(core - found)
    data["parse_confidence"]  = (
        "high"   if len(found) >= 4 else
        "medium" if len(found) >= 2 else "low"
    )
    return data


# ─────────────────────────────────────────────
# GROQ VISION FALLBACK
# ─────────────────────────────────────────────
_GROQ_SYSTEM = """You are a nutrition label parser.
Extract EVERY nutrient and return ONLY a valid JSON object — no markdown, no explanation.
Format:
{"calories":null,"total_fat":null,"saturated_fat":null,"trans_fat":null,
 "cholesterol":null,"sodium":null,"carbohydrates":null,"fiber":null,
 "sugar":null,"added_sugar":null,"protein":null,"vitamin_d":null,
 "calcium":null,"iron":null,"potassium":null,
 "serving_size_g":null,"servings_per_container":null,"product_name":null}
Units: fat/carb/fiber/sugar/protein in g. cholesterol/sodium/calcium/iron/potassium in mg.
vitamin_d in mcg. calories in kcal. null for anything not visible."""


def _get_groq_key() -> Optional[str]:
    """
    Read Groq API key — tries every possible access pattern.
    Handles Streamlit's AttrDict secrets object robustly.
    """
    import os
    # Try all Streamlit secrets access patterns
    for attr in ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY"]:
        try:
            # Direct attribute access (most reliable on Streamlit Cloud)
            val = getattr(st.secrets, attr, None)
            if val and str(val).strip().startswith("gsk_"):
                return str(val).strip()
        except Exception:
            pass
        try:
            # Dict-style access
            val = st.secrets[attr]
            if val and str(val).strip().startswith("gsk_"):
                return str(val).strip()
        except Exception:
            pass
        try:
            # .get() style
            val = st.secrets.get(attr)
            if val and str(val).strip().startswith("gsk_"):
                return str(val).strip()
        except Exception:
            pass
    # Environment variable fallback
    for env in ["GROQ_API_KEY", "groq_api_key"]:
        val = os.environ.get(env)
        if val and val.strip().startswith("gsk_"):
            return val.strip()
    return None


def groq_extract(pil_img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Groq vision API for extracting nutrition facts from images.
    
    Note: As of March 2026, all Groq vision models are decommissioned.
    This function returns None - use OCR (EasyOCR) for text extraction instead.
    See: https://console.groq.com/docs/deprecations
    """
    # Vision models no longer available on Groq
    # Returning None tells the app to rely on OCR results only
    return None


def groq_to_parsed_data(groq_json: Dict, original_data: Dict) -> Dict:
    micro = {"vitamin_d", "calcium", "iron", "potassium"}
    for key, unit in _UNIT_MAP.items():
        val = groq_json.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if val < 0 or val >= 1e5:
            continue
        bucket = "micronutrients" if key in micro else "nutrients"
        if key not in original_data[bucket]:
            original_data[bucket][key] = {"value": val, "unit": unit, "source": "groq"}

    ssv = groq_json.get("serving_size_g")
    if ssv and "serving_size" not in original_data["nutrients"]:
        try:
            original_data["nutrients"]["serving_size"] = {
                "value": float(ssv), "unit": "g", "source": "groq"}
        except Exception:
            pass

    if groq_json.get("product_name") and not original_data.get("product_name"):
        original_data["product_name"] = str(groq_json["product_name"]).title()

    core  = {"calories", "protein", "carbohydrates", "total_fat", "sodium"}
    found = core & original_data["nutrients"].keys()
    original_data["missing_nutrients"] = list(core - found)
    original_data["parse_confidence"]  = (
        "high"   if len(found) >= 4 else
        "medium" if len(found) >= 2 else "low"
    )
    return original_data