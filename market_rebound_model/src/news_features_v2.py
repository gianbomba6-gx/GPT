"""Combine normalized daily news and event features for the V2 model."""
from __future__ import annotations
import pandas as pd
from news_event_classifier import build_daily_event_features


def build_news_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["Date", "symbol", "news_count"])
    event = build_daily_event_features(raw)
    return event.sort_values(["symbol", "Date"]).reset_index(drop=True)


def merge_with_market(market: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    m = market.copy()
    m["Date"] = pd.to_datetime(m["Date"]).dt.normalize()
    n = news_daily.copy()
    n["Date"] = pd.to_datetime(n["Date"]).dt.normalize()
    return m.merge(n, on=["Date", "symbol"], how="left").fillna({
        "news_count": 0,
        "negative_news_share": 0,
        "material_event_share": 0,
        "event_polarity": 0,
        "event_intensity": 0,
        "unique_event_types": 0,
    })
