"""Leakage-safe news features aligned to the next tradable session."""
from __future__ import annotations
import pandas as pd
from news_event_classifier import add_event_features

# UTC close times for the exchanges used by the current universe in summer.
# The live system will later derive these from exchange calendars/time zones.
CLOSE_UTC_HOUR = {"STLAM.MI": 15.5, "STLA": 15.5, "SPCX": 20.0, "NVDA": 20.0, "TSLA": 20.0}


def build_news_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["available_date", "symbol", "news_count"])
    x = add_event_features(raw.copy())
    ts = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x = x.loc[ts.notna()].copy()
    x["_ts"] = ts.loc[x.index]
    x["_date"] = x["_ts"].dt.normalize()
    hours = x["symbol"].map(CLOSE_UTC_HOUR).fillna(20.0)
    minutes_from_midnight = x["_ts"].dt.hour * 60 + x["_ts"].dt.minute + x["_ts"].dt.second / 60
    cutoff = hours * 60
    # A signal made after the close cannot use an article published later that day.
    # Such an article becomes available for the following market session.
    x["available_date"] = x["_date"] + pd.to_timedelta((minutes_from_midnight > cutoff).astype(int), unit="D")
    return (x.groupby(["available_date", "symbol"], as_index=False)
        .agg(news_count=("headline", "size"),
             negative_news_share=("is_negative_event", "mean"),
             material_event_share=("is_material_event", "mean"),
             event_polarity=("event_polarity", "mean"),
             event_intensity=("event_intensity", "mean"),
             unique_event_types=("event_type", "nunique")))


def merge_with_market(market: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach each news bucket to the first market session on/after availability."""
    m = market.copy()
    m["Date"] = pd.to_datetime(m["Date"], utc=True, errors="coerce").dt.normalize().dt.tz_localize(None)
    n = news_daily.copy()
    n["available_date"] = pd.to_datetime(n["available_date"], utc=True, errors="coerce").dt.normalize().dt.tz_localize(None)
    n = n.dropna(subset=["available_date", "symbol"])
    parts = []
    for symbol, ms in m.groupby("symbol", sort=False):
        ns = n[n["symbol"] == symbol].sort_values("available_date")
        ms = ms.sort_values("Date").copy()
        if ns.empty:
            parts.append(ms)
            continue
        merged = pd.merge_asof(ms, ns, left_on="Date", right_on="available_date", direction="backward")
        parts.append(merged)
    out = pd.concat(parts, ignore_index=True) if parts else m
    for c in ["news_count", "negative_news_share", "material_event_share", "event_polarity", "event_intensity", "unique_event_types"]:
        if c not in out:
            out[c] = 0.0
    return out.drop(columns=["available_date", "symbol_y"], errors="ignore").rename(columns={"symbol_x": "symbol"}).fillna({
        "news_count": 0,
        "negative_news_share": 0,
        "material_event_share": 0,
        "event_polarity": 0,
        "event_intensity": 0,
        "unique_event_types": 0,
    })
