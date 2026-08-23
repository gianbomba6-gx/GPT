"""Leakage-safe news features aligned to the next tradable session."""
from __future__ import annotations
import pandas as pd
from news_event_classifier import add_event_features

CLOSE_UTC_MINUTE = {"STLAM.MI": 15 * 60 + 30, "STLA": 15 * 60 + 30, "SPCX": 20 * 60, "NVDA": 20 * 60, "TSLA": 20 * 60}


def build_news_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["available_date", "symbol", "news_count"])
    x = add_event_features(raw.copy())
    ts = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x = x.loc[ts.notna()].copy()
    x["_ts"] = ts.loc[x.index]
    x["_date"] = x["_ts"].dt.normalize()
    cutoff = x["symbol"].map(CLOSE_UTC_MINUTE).fillna(20 * 60)
    minute = x["_ts"].dt.hour * 60 + x["_ts"].dt.minute + x["_ts"].dt.second / 60
    x["available_date"] = x["_date"] + pd.to_timedelta((minute > cutoff).astype(int), unit="D")
    return (x.groupby(["available_date", "symbol"], as_index=False)
        .agg(news_count=("headline", "size"),
             negative_news_share=("is_negative_event", "mean"),
             material_event_share=("is_material_event", "mean"),
             event_polarity=("event_polarity", "mean"),
             event_intensity=("event_intensity", "mean"),
             unique_event_types=("event_type", "nunique")))


def merge_with_market(market: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """Map each news bucket to the first available market session, then exact-merge.

    Unlike a backward asof merge, this never carries old news forward into a later
    session with no new articles.
    """
    m = market.copy()
    m["Date"] = pd.to_datetime(m["Date"], errors="coerce", utc=True).dt.normalize().dt.tz_localize(None)
    n = news_daily.copy()
    n["available_date"] = pd.to_datetime(n["available_date"], errors="coerce", utc=True).dt.normalize().dt.tz_localize(None)
    n = n.dropna(subset=["available_date", "symbol"])
    mapped = []
    for symbol, ns in n.groupby("symbol", sort=False):
        ms = m[m["symbol"] == symbol][["Date"]].drop_duplicates().sort_values("Date")
        if ms.empty:
            continue
        z = pd.merge_asof(ns.sort_values("available_date"), ms,
                          left_on="available_date", right_on="Date", direction="forward")
        mapped.append(z.dropna(subset=["Date"]))
    if mapped:
        n = pd.concat(mapped, ignore_index=True)
        n = (n.groupby(["Date", "symbol"], as_index=False)
             .agg(news_count=("news_count", "sum"),
                  negative_news_share=("negative_news_share", "mean"),
                  material_event_share=("material_event_share", "mean"),
                  event_polarity=("event_polarity", "mean"),
                  event_intensity=("event_intensity", "mean"),
                  unique_event_types=("unique_event_types", "max")))
        out = m.merge(n, on=["Date", "symbol"], how="left")
    else:
        out = m.copy()
    for c in ["news_count", "negative_news_share", "material_event_share", "event_polarity", "event_intensity", "unique_event_types"]:
        if c not in out:
            out[c] = 0.0
    return out.fillna({
        "news_count": 0,
        "negative_news_share": 0,
        "material_event_share": 0,
        "event_polarity": 0,
        "event_intensity": 0,
        "unique_event_types": 0,
    })
