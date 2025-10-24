from __future__ import annotations
import os
import time
import json
import math
import base64
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import feedparser
import pandas as pd
import requests
import tldextract

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from dotenv import load_dotenv
load_dotenv()

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

@dataclass
class Article:
    title: str
    summary: str
    link: str
    published: Optional[str]
    source: str
    country: Optional[str] = None
    sentiment: Optional[float] = None  # compound score (-1..1)

@dataclass
class CountryBrief:
    country: str
    count: int
    mean_sentiment: float
    summary_points: str

COUNTRY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    'India': (20.5937, 78.9629),
    'United States': (39.8283, -98.5795),
    'USA': (39.8283, -98.5795),
    'United Kingdom': (55.3781, -3.4360),
    'UK': (55.3781, -3.4360),
    'Germany': (51.1657, 10.4515),
    'France': (46.2276, 2.2137),
    'Italy': (41.8719, 12.5674),
    'Spain': (40.4637, -3.7492),
    'Portugal': (39.3999, -8.2245),
    'Netherlands': (52.1326, 5.2913),
    'Belgium': (50.5039, 4.4699),
    'Switzerland': (46.8182, 8.2275),
    'Austria': (47.5162, 14.5501),
    'Poland': (51.9194, 19.1451),
    'Czechia': (49.8175, 15.4730),
    'Hungary': (47.1625, 19.5033),
    'Romania': (45.9432, 24.9668),
    'Greece': (39.0742, 21.8243),
    'Ireland': (53.1424, -7.6921),
    'Norway': (60.4720, 8.4689),
    'Sweden': (60.1282, 18.6435),
    'Denmark': (56.2639, 9.5018),
    'Finland': (61.9241, 25.7482),
    'Russia': (61.5240, 105.3188),
    'Ukraine': (48.3794, 31.1656),
    'Turkey': (38.9637, 35.2433),
    'Israel': (31.0461, 34.8516),
    'Saudi Arabia': (23.8859, 45.0792),
    'UAE': (23.4241, 53.8478),
    'Qatar': (25.3548, 51.1839),
    'Iran': (32.4279, 53.6880),
    'Iraq': (33.2232, 43.6793),
    'Egypt': (26.8206, 30.8025),
    'South Africa': (30.5595, 22.9375),
    'Nigeria': (9.0820, 8.6753),
    'Kenya': (0.0236, 37.9062),
    'Ethiopia': (9.1450, 40.4897),
    'Morocco': (31.7917, -7.0926),
    'Algeria': (28.0339, 1.6596),
    'Tunisia': (33.8869, 9.5375),
    'China': (35.8617, 104.1954),
    'Japan': (36.2048, 138.2529),
    'South Korea': (35.9078, 127.7669),
    'North Korea': (40.3399, 127.5101),
    'Singapore': (1.3521, 103.8198),
    'Malaysia': (4.2105, 101.9758),
    'Thailand': (15.8700, 100.9925),
    'Vietnam': (14.0583, 108.2772),
    'Philippines': (12.8797, 121.7740),
    'Indonesia': (-0.7893, 113.9213),
    'Bangladesh': (23.6850, 90.3563),
    'Pakistan': (30.3753, 69.3451),
    'Sri Lanka': (7.8731, 80.7718),
    'Nepal': (28.3949, 84.1240),
    'Bhutan': (27.5142, 90.4336),
    'Australia': (-25.2744, 133.7751),
    'New Zealand': (-40.9006, 174.8860),
    'Canada': (56.1304, -106.3468),
    'Mexico': (23.6345, -102.5528),
    'Brazil': (-14.2350, -51.9253),
    'Argentina': (-38.4161, -63.6167),
    'Chile': (-35.6751, -71.5430),
    'Colombia': (4.5709, -74.2973),
    'Peru': (-9.1900, -75.0152),
}

COUNTRY_ALIASES: Dict[str, str] = {
    'US': 'United States', 'U.S.': 'United States', 'USA': 'United States', 'U.S.A.': 'United States',
    'UK': 'United Kingdom', 'U.K.': 'United Kingdom', 'UAE': 'UAE',
}

DOMAIN_DEFAULT_COUNTRY = {
    'bbc': 'United Kingdom',
    'reuters': 'United Kingdom',
    'aljazeera': 'Qatar',
    'thehindu': 'India',
    'indianexpress': 'India',
    'hindustantimes': 'India',
    'ndtv': 'India',
    'toi': 'India',
    'timesofindia': 'India',
    'washingtonpost': 'United States',
    'nytimes': 'United States',
    'apnews': 'United States',
    'cnn': 'United States',
    'foxnews': 'United States',
    'cbc': 'Canada',
    'abc': 'Australia',
    'abcnews': 'United States',
    'al-arabiya': 'Saudi Arabia',
}


RSS_FEEDS = [
    # Global
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en",
    # India
    "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    # Tech & Business
    "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
    # Reputable outlets
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.thehindu.com/news/international/feeder/default.rss",
    "https://indianexpress.com/section/world/feed/",
]

def gemini_generate_text(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment (.env)")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    r = requests.post(
        GEMINI_ENDPOINT,
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    for cand in data.get("candidates", []):
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p:
                return p["text"].strip()
    return ""

COUNTRY_KEYWORDS = {k: k for k in COUNTRY_CENTROIDS.keys()}
COUNTRY_KEYWORDS.update(COUNTRY_ALIASES)

def infer_country_heuristic(text: str, source_domain: str | None) -> Optional[str]:
    t = (text or "").lower()
    for k, canonical in COUNTRY_KEYWORDS.items():
        if k.lower() in t:
            return COUNTRY_ALIASES.get(k, k if k in COUNTRY_CENTROIDS else canonical)
    if source_domain:
        dom = tldextract.extract(source_domain).domain
        if dom in DOMAIN_DEFAULT_COUNTRY:
            return DOMAIN_DEFAULT_COUNTRY[dom]
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def infer_country_with_gemini(text: str) -> Optional[str]:
    prompt = f"""
You are given a news headline and snippet. Identify the SINGLE most relevant country involved.
Return ONLY the country name (e.g., India, United States, United Kingdom). If uncertain, reply: Unknown.
Text: {text}
"""
    try:
        out = gemini_generate_text(prompt)
        out = out.strip().replace("\n", " ")
        if out.lower() == "unknown":
            return None
        if out in COUNTRY_ALIASES:
            out = COUNTRY_ALIASES[out]
        return out
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=600)
def fetch_rss_articles(max_per_feed: int = 30) -> List[Article]:
    seen_links = set()
    items: List[Article] = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                link = entry.get('link')
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                published = entry.get('published', None)
                source = feed.feed.get('title', tldextract.extract(link).domain)
                items.append(Article(title=title, summary=summary, link=link, published=published, source=source))
        except Exception:
            continue
    return items

sia = SentimentIntensityAnalyzer()

def score_sentiment(text: str) -> float:
    try:
        return float(sia.polarity_scores(text or "")["compound"])
    except Exception:
        return 0.0


def assign_countries(articles: List[Article], allow_gemini_backfill: bool = True) -> List[Article]:
    for a in articles:
        domain = tldextract.extract(a.link).registered_domain
        c = infer_country_heuristic(f"{a.title} {a.summary}", domain)
        if not c and allow_gemini_backfill:
            c = infer_country_with_gemini(f"Title: {a.title}\nSummary: {a.summary}")
        a.country = c
        a.sentiment = score_sentiment(f"{a.title}. {a.summary}")
    return [a for a in articles if a.country in COUNTRY_CENTROIDS]

@st.cache_data(show_spinner=False, ttl=900)
def build_country_briefs(articles: List[Article], max_countries: int = 25) -> List[CountryBrief]:
    df = pd.DataFrame([{
        'country': a.country,
        'sentiment': a.sentiment,
        'title': a.title,
        'summary': a.summary,
        'link': a.link,
    } for a in articles if a.country])
    if df.empty:
        return []
    grouped = df.groupby('country')
    rows: List[CountryBrief] = []

    counts = grouped.size().sort_values(ascending=False)
    top_countries = list(counts.index)[:max_countries]

    for country in top_countries:
        sub = grouped.get_group(country)
        count = len(sub)
        mean_sent = float(sub['sentiment'].mean()) if count else 0.0
        titles = "\n".join([f"- {t}" for t in sub['title'].tolist()[:15]])
        prompt = f"""
Summarize the top themes in recent news for {country} in 3 short bullet points.
Focus on clarity and avoid hype. Base your summary strictly on these headlines:
{titles}
"""
        try:
            summary_points = gemini_generate_text(prompt)
        except Exception:
            summary_points = "(AI summary unavailable)"
        rows.append(CountryBrief(country=country, count=count, mean_sentiment=mean_sent, summary_points=summary_points))
    return rows


st.set_page_config(page_title="AI Global News Mapper", page_icon="🗺️", layout="wide")
st.title("AI Global News Mapper")
st.caption("Live RSS → Country clustering → Sentiment → Gemini summaries. All free tools; Gemini API key required.")

with st.sidebar:
    st.header("Controls")
    allow_gemini_backfill = st.toggle("Use Gemini when country is unclear", value=True, help="Saves tokens if off.")
    sentiment_filter = st.select_slider("Sentiment filter", options=["All", "Positive", "Neutral", "Negative"], value="All")
    st.write(":arrow_right: Click a country in the table to view articles below.")
    if st.button("Refresh feeds"):
        fetch_rss_articles.clear()
        infer_country_with_gemini.clear()
        build_country_briefs.clear()
        st.success("Caches cleared. Data will refresh on next load.")

with st.spinner("Fetching news feeds…"):
    articles = fetch_rss_articles(max_per_feed=30)

with st.spinner("Classifying countries & sentiment…"):
    articles = assign_countries(articles, allow_gemini_backfill=allow_gemini_backfill)

if sentiment_filter != "All":
    def bucket(s):
        if s is None:
            return "Neutral"
        return "Positive" if s > 0.2 else ("Negative" if s < -0.2 else "Neutral")
    filt = {
        'Positive': lambda a: bucket(a.sentiment) == 'Positive',
        'Neutral': lambda a: bucket(a.sentiment) == 'Neutral',
        'Negative': lambda a: bucket(a.sentiment) == 'Negative',
    }[sentiment_filter]
    articles = [a for a in articles if filt(a)]

with st.spinner("Building country briefings (Gemini)…"):
    briefs = build_country_briefs(articles)

map_rows = []
for b in briefs:
    if b.country in COUNTRY_CENTROIDS:
        lat, lon = COUNTRY_CENTROIDS[b.country]
        map_rows.append({
            'country': b.country,
            'lat': lat,
            'lon': lon,
            'count': b.count,
            'sentiment': b.mean_sentiment,
            'summary': b.summary_points,
        })

map_df = pd.DataFrame(map_rows)

import pydeck as pdk

if not map_df.empty:
    max_count = max(map_df['count']) if len(map_df) else 1
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[lon, lat]',
        get_radius=f"count/{max_count}*800000+80000",
        get_fill_color='[ (1-sentiment)*128 + 128, 50, (sentiment+1)*128, 180 ]',
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2)
    tooltip = {
        'html': '<b>{country}</b><br/>Articles: {count}<br/>Sentiment: {sentiment}',
        'style': {'backgroundColor': 'steelblue', 'color': 'white'}
    }
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style='mapbox://styles/mapbox/light-v10')
    st.pydeck_chart(r, use_container_width=True)
else:
    st.info("No data to plot yet. Try refreshing or enabling Gemini backfill.")

if briefs:
    tbl = pd.DataFrame([{
        'Country': b.country,
        'Articles': b.count,
        'Mean Sentiment': round(b.mean_sentiment, 3)
    } for b in briefs]).sort_values(by=['Articles'], ascending=False)
    st.subheader("Country Overview")
    st.dataframe(tbl, use_container_width=True)

    sel_country = st.selectbox("View details for country", [b.country for b in briefs])
    if sel_country:
        b = next(x for x in briefs if x.country == sel_country)
        st.markdown(f"### {b.country} — AI Summary")
        st.markdown(b.summary_points)

        subset = [a for a in articles if a.country == sel_country]
        st.markdown(f"#### Articles ({len(subset)})")
        for a in subset:
            sent_bucket = '🟢' if (a.sentiment or 0) > 0.2 else ('🔴' if (a.sentiment or 0) < -0.2 else '🟡')
            st.markdown(f"{sent_bucket} **{a.title}**  ")
            if a.published:
                st.markdown(f"<span style='color:#666'>Published: {a.published}</span>", unsafe_allow_html=True)
            st.markdown(f"[Read]({a.link})")
            st.markdown("---")
else:
    st.warning("No country summaries available. Try enabling Gemini backfill and refresh.")

st.caption("Built with Streamlit, RSS, VADER, PyDeck, and Gemini. Colors reflect sentiment; size reflects volume.")
