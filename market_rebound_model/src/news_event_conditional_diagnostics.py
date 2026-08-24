"""Causal, out-of-sample diagnostics for event-conditioned rebound outcomes."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from .news_event_classifier import EVENT_TYPES, add_event_features
except ImportError:
    from news_event_classifier import EVENT_TYPES, add_event_features

MARKET_META = {
    "STLAM.MI": ("Europe/Rome", time(17, 30)),
    "SPCX": ("America/New_York", time(16, 0)),
    "NVDA": ("America/New_York", time(16, 0)),
    "TSLA": ("America/New_York", time(16, 0)),
}

REQUIRED_OOS = {"symbol", "Date", "next_ret", "next_high"}
REQUIRED_RAW = {"published_at", "symbol", "candidate_day"}


def close_utc(day: date, symbol: str) -> pd.Timestamp:
    tz_name, close_time = MARKET_META[symbol]
    local = datetime.combine(day, close_time, tzinfo=ZoneInfo(tz_name))
    return pd.Timestamp(local).tz_convert("UTC")


def _classify_raw(raw: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_RAW - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    x = raw.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x["candidate_day"] = pd.to_datetime(x["candidate_day"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["published_at", "candidate_day", "symbol"]).copy()

    cutoffs = []
    for symbol, day in zip(x["symbol"], x["candidate_day"]):
        if symbol not in MARKET_META:
            cutoffs.append(pd.NaT)
        else:
            cutoffs.append(close_utc(day.date(), symbol))
    x["market_close_utc"] = cutoffs
    x = x[x["market_close_utc"].notna() & (x["published_at"] <= x["market_close_utc"])].copy()
    if x.empty:
        return x
    return add_event_features(x)


def _event_day_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = _classify_raw(raw)
    keys = ["candidate_day", "symbol"]
    if x.empty:
        return pd.DataFrame(columns=keys)

    base = x.groupby(keys, as_index=False).agg(news_count=("event_type", "size"))
    counts = pd.crosstab([x["candidate_day"], x["symbol"]], x["event_type"]).reset_index()
    counts = counts.rename(columns={c: f"event_{c}_count" for c in counts.columns if c not in keys})
    negative = x[x["is_negative_event"] == 1]
    neg_counts = pd.crosstab([negative["candidate_day"], negative["symbol"]], negative["event_type"]).reset_index()
    neg_counts = neg_counts.rename(columns={c: f"negative_event_{c}_count" for c in neg_counts.columns if c not in keys})

    out = base.merge(counts, on=keys, how="left").merge(neg_counts, on=keys, how="left")
    for event in EVENT_TYPES:
        ec = f"event_{event}_count"
        nc = f"negative_event_{event}_count"
        if ec not in out.columns:
            out[ec] = 0
        if nc not in out.columns:
            out[nc] = 0
        out[ec] = pd.to_numeric(out[ec], errors="coerce").fillna(0)
        out[nc] = pd.to_numeric(out[nc], errors="coerce").fillna(0)
        out[f"event_{event}_share"] = out[ec] / out["news_count"].clip(lower=1)
        out[f"negative_event_{event}_share"] = out[nc] / out["news_count"].clip(lower=1)

    return out


def _stats(x: pd.DataFrame, symbol: str, event: str, condition: str) -> dict:
    x = x[x["next_ret"].notna()].copy()
    n = len(x)
    return {
        "symbol": symbol,
        "event": event,
        "condition": condition,
        "n": n,
        "mean_next_ret": float(x["next_ret"].mean()) if n else None,
        "median_next_ret": float(x["next_ret"].median()) if n else None,
        "hit_2pct": float((x["next_ret"] >= 0.02).mean()) if n else None,
        "hit_3pct": float((x["next_ret"] >= 0.03).mean()) if n else None,
        "hit_5pct": float((x["next_ret"] >= 0.05).mean()) if n else None,
        "mean_next_high": float(x["next_high"].mean()) if n else None,
    }


def build_conditional_diagnostics(oos: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_OOS - set(oos.columns)
    if missing:
        raise ValueError(f"Missing OOS columns: {sorted(missing)}")
    x = oos.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["next_high"] = pd.to_numeric(x["next_high"], errors="coerce")
    x = x.dropna(subset=["Date", "symbol"]).copy()

    event_days = _event_day_features(raw).rename(columns={"candidate_day": "Date"})
    if event_days.empty:
        raise ValueError("No causal event-day features available after market-close filtering")
    event_days["Date"] = pd.to_datetime(event_days["Date"], errors="coerce").dt.normalize()
    x = x.merge(event_days, on=["Date", "symbol"], how="left")

    rows: list[dict] = []
    for symbol, s in x.groupby("symbol", sort=True):
        rows.append(_stats(s, symbol, "all", "all_candidates"))
        dominant_share = pd.concat([
            pd.to_numeric(s[f"event_{e}_share"], errors="coerce").fillna(0).rename(e)
            for e in EVENT_TYPES
        ], axis=1)
        primary = dominant_share.idxmax(axis=1)
        max_share = dominant_share.max(axis=1)
        for event in EVENT_TYPES:
            share = pd.to_numeric(s[f"event_{event}_share"], errors="coerce").fillna(0)
            neg_share = pd.to_numeric(s[f"negative_event_{event}_share"], errors="coerce").fillna(0)
            rows.append(_stats(s[share > 0], symbol, event, "event_present"))
            rows.append(_stats(s[neg_share > 0], symbol, event, "event_negative"))
            dominant = (primary == event) & (max_share > 0)
            rows.append(_stats(s[dominant], symbol, event, "event_dominant"))
            rows.append(_stats(s[dominant & (neg_share > 0)], symbol, event, "event_dominant_negative"))

    return pd.DataFrame(rows).sort_values(["symbol", "condition", "event"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oos_predictions")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_event_conditional_diagnostics.csv")
    args = ap.parse_args()

    oos = pd.read_csv(args.oos_predictions)
    raw = pd.read_csv(args.raw_gkg)
    out = build_conditional_diagnostics(oos, raw)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
