"""Candidate-day GDELT GKG backfill for causal rebound research.

Downloads GKG only for days where a tracked stock falls at least the configured
threshold. The model decision is made after that day's close, so every article
published on the same calendar day—including post-close articles—is available
for the next-session prediction and remains aligned to the candidate day.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

try:
    from .gkg_historical_news import GkgHistoricalProvider
    from .news_event_classifier import add_event_features
except ImportError:
    from gkg_historical_news import GkgHistoricalProvider
    from news_event_classifier import add_event_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
DEFAULT_SYMBOLS = [x["symbol"] for x in CONFIG["tickers"] if x["type"] == "equity"]
DEFAULT_THRESHOLD = float(CONFIG["signal"]["down_day_threshold"])
MARKET_META = {
    "STLAM.MI": ("Europe/Rome", time(17, 30)),
    "SPCX": ("America/New_York", time(16, 0)),
    "NVDA": ("America/New_York", time(16, 0)),
    "TSLA": ("America/New_York", time(16, 0)),
}


def fetch_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Close": "Ultimo"})
    raw["Date"] = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw["ret"] = raw["Ultimo"].pct_change()
    return raw[["Date", "ret"]].dropna()


def candidate_dates(symbols: list[str], start: str, end: str, threshold: float) -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    for symbol in symbols:
        d = fetch_prices(symbol, start, end)
        out[symbol] = [x.Date.date() for _, x in d.iterrows() if float(x.ret) <= threshold]
    return out


def close_utc(day: date, symbol: str) -> pd.Timestamp:
    tz_name, close_time = MARKET_META[symbol]
    local = datetime.combine(day, close_time, tzinfo=ZoneInfo(tz_name))
    return pd.Timestamp(local).tz_convert("UTC")


def filter_at_close(df: pd.DataFrame, symbol: str, day: date) -> pd.DataFrame:
    """Return only news known by the local market close; kept for diagnostics/tests."""
    if df.empty:
        return df
    cutoff = close_utc(day, symbol)
    x = df.copy()
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    return x[x["published_at"] <= cutoff].copy()


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date", "symbol", "news_sentiment", "news_intensity", "news_relevance",
        "news_novelty", "news_count", "negative_news_share", "material_event_share",
        "event_polarity", "event_intensity", "unique_event_types", "news_available",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    x = add_event_features(raw.copy())
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x["candidate_day"] = pd.to_datetime(x["candidate_day"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["published_at", "candidate_day", "symbol"]).copy()

    x["Date"] = x["candidate_day"]
    daily = (x.groupby(["Date", "symbol"], as_index=False)
        .agg(news_sentiment=("sentiment", "mean"),
             news_intensity=("intensity", "mean"),
             news_relevance=("relevance", "mean"),
             news_novelty=("novelty", "mean"),
             news_count=("classification_text", "size"),
             negative_news_share=("is_negative_event", "mean"),
             material_event_share=("is_material_event", "mean"),
             event_polarity=("event_polarity", "mean"),
             event_intensity=("event_intensity", "mean"),
             unique_event_types=("event_type", "nunique")))
    daily["news_available"] = 1.0
    return daily[columns]


def run(symbols: list[str], start: str, end: str, threshold: float, out_raw: str, out_daily: str) -> None:
    candidates = candidate_dates(symbols, start, end, threshold)
    union_days = sorted({d for dates in candidates.values() for d in dates})
    counts = {s: len(v) for s, v in candidates.items()}
    print(f"CANDIDATE DAYS: {counts}")
    print(f"UNION DAYS: {len(union_days)}")

    provider = GkgHistoricalProvider(timeout=90)
    raw_chunks: list[pd.DataFrame] = []
    for day in union_days:
        day_symbols = [symbol for symbol in symbols if day in candidates[symbol]]
        if not day_symbols:
            continue
        df = provider.fetch_day_multi(day_symbols, day)
        if df.empty:
            print(f"NEWS {day}: 0 articles for {','.join(day_symbols)}")
            continue
        df["candidate_day"] = day.isoformat()
        raw_chunks.append(df)
        for symbol in day_symbols:
            symbol_df = df[df["symbol"] == symbol]
            cutoff = close_utc(day, symbol)
            post_close = int((pd.to_datetime(symbol_df["published_at"], utc=True, errors="coerce") > cutoff).sum())
            print(f"NEWS {symbol} {day}: {len(symbol_df)} articles; post-close={post_close}")

    raw = pd.concat(raw_chunks, ignore_index=True) if raw_chunks else pd.DataFrame()
    raw_path = Path(out_raw)
    daily_path = Path(out_daily)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    if raw.empty:
        raw = pd.DataFrame(columns=[
            "published_at", "symbol", "headline", "source", "url", "summary",
            "category", "sentiment", "intensity", "relevance", "novelty", "candidate_day",
        ])
        raw.to_csv(raw_path, index=False)
        aggregate_daily(raw).to_csv(daily_path, index=False)
        print(f"WARNING: no matching GKG articles for {start}..{end}")
        return

    daily = aggregate_daily(raw)
    raw.to_csv(raw_path, index=False)
    daily.to_csv(daily_path, index=False)
    print(f"Saved raw articles={len(raw)} to {raw_path}")
    print(f"Saved daily news rows={len(daily)} to {daily_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out-raw", default="results/news_candidate_raw.csv")
    ap.add_argument("--out-daily", default="results/news_candidate_daily.csv")
    args = ap.parse_args()
    run(args.symbols, args.start, args.end, args.threshold, args.out_raw, args.out_daily)


if __name__ == "__main__":
    main()
