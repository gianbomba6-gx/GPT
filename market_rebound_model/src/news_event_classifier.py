"""Deterministic event features for causal rebound-model news V3."""
from __future__ import annotations
import re
import pandas as pd

EVENT_PATTERNS = {
    "earnings": r"\b(earnings|revenue|profit|eps|quarterly results|results)\b",
    "guidance": r"\b(guidance|outlook|forecast|raises? guidance|cuts? guidance)\b",
    "analyst": r"\b(upgrade|downgrade|price target|analyst|rating)\b",
    "regulatory": r"\b(regulator|regulatory|antitrust|lawsuit|investigation|fine|sec)\b",
    "ma": r"\b(acquire|acquisition|merger|takeover|deal)\b",
    "product": r"\b(product|launch|recall|delivery|deliveries|production)\b",
    "macro": r"\b(fed|inflation|rates?|interest rate|tariff|recession|gdp)\b",
}
EVENT_TYPES = tuple(EVENT_PATTERNS)

NEGATIVE_WORDS = re.compile(r"\b(miss|misses|weak|cut|cuts|lower|warning|decline|falls?|drop|plunge|loss|lawsuit|investigation|fine|downgrade)\b", re.I)
POSITIVE_WORDS = re.compile(r"\b(beat|beats|strong|raise|raises|higher|growth|profit|upgrade|buy|approval|record)\b", re.I)


def classify_headline(text: str) -> tuple[str, float, float]:
    text = str(text or "")
    low = text.lower()
    matches = [name for name, pattern in EVENT_PATTERNS.items() if re.search(pattern, low)]
    event = matches[0] if matches else "other"
    neg = len(NEGATIVE_WORDS.findall(text))
    pos = len(POSITIVE_WORDS.findall(text))
    polarity = (pos - neg) / max(pos + neg, 1)
    event_intensity = min(1.0, (len(matches) + neg + pos) / 4.0)
    return event, polarity, event_intensity


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "headline" not in out.columns:
        out["headline"] = ""
    if "summary" not in out.columns:
        out["summary"] = ""
    text = out["headline"].fillna("").astype(str).str.strip()
    fallback = out["summary"].fillna("").astype(str).str.strip()
    text = text.where(text.ne(""), fallback)
    classified = text.map(classify_headline)
    out["classification_text"] = text
    out["event_type"] = classified.map(lambda x: x[0])
    out["event_polarity"] = classified.map(lambda x: x[1])
    out["event_intensity"] = classified.map(lambda x: x[2])
    out["is_negative_event"] = (out["event_polarity"] < 0).astype(int)
    out["is_material_event"] = (out["event_intensity"] >= 0.5).astype(int)
    return out


def build_daily_event_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_event_features(df)
    x["Date"] = pd.to_datetime(x["published_at"], utc=True).dt.normalize()
    keys = ["Date", "symbol"]
    daily = (x.groupby(keys, as_index=False)
        .agg(news_count=("classification_text", "size"),
             negative_news_share=("is_negative_event", "mean"),
             material_event_share=("is_material_event", "mean"),
             event_polarity=("event_polarity", "mean"),
             event_intensity=("event_intensity", "mean"),
             unique_event_types=("event_type", "nunique")))
    # Reason-specific shares: fraction of the day's classified articles in each event family.
    event_counts = pd.crosstab([x["Date"], x["symbol"]], x["event_type"]).reset_index()
    event_counts = event_counts.rename(columns={c: f"event_{c}_share" for c in event_counts.columns if c not in keys})
    daily = daily.merge(event_counts, on=keys, how="left")
    for event in EVENT_TYPES + ("other",):
        col = f"event_{event}_share"
        if col not in daily.columns:
            daily[col] = 0.0
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
        # Crosstab gives counts; convert to fractions after merge.
        daily[col] = daily[col] / daily["news_count"].clip(lower=1)
    return daily
