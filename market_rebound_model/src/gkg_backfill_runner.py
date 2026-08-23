"""Checkpointed historical GDELT GKG backfill, one daily archive at a time."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    from .gkg_historical_news import GkgHistoricalProvider
except ImportError:
    from gkg_historical_news import GkgHistoricalProvider

DEFAULT_SYMBOLS = ["STLAM.MI", "SPCX", "NVDA", "TSLA"]
KEYS = ["published_at", "symbol", "headline", "source", "url"]


def run(symbols: list[str], start: date, end: date, out: str) -> None:
    if end < start:
        raise ValueError("end must be >= start")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out_path.with_suffix(out_path.suffix + ".checkpoint.csv")
    if checkpoint.exists():
        existing = pd.read_csv(checkpoint)
    elif out_path.exists():
        existing = pd.read_csv(out_path)
    else:
        existing = pd.DataFrame()

    completed_days = set()
    if not existing.empty and "_day" in existing.columns:
        completed_days = set(existing["_day"].astype(str))
    chunks = [existing] if not existing.empty else []

    provider = GkgHistoricalProvider()
    cursor = start
    while cursor <= end:
        key = cursor.isoformat()
        if key in completed_days:
            cursor += timedelta(days=1)
            continue
        print(f"GKG DAY: {key}")
        day_frames = []
        for symbol in symbols:
            df = provider.fetch_day(symbol, cursor).copy()
            df["_day"] = key
            day_frames.append(df)
            print(f"  {symbol}: {len(df)} articles")
        day_df = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
        chunks.append(day_df)
        completed_days.add(key)
        combined = pd.concat(chunks, ignore_index=True)
        combined.to_csv(checkpoint, index=False)
        combined.drop(columns=["_day"], errors="ignore").drop_duplicates(subset=KEYS).to_csv(out_path, index=False)
        cursor += timedelta(days=1)

    final = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    final.drop(columns=["_day"], errors="ignore").drop_duplicates(subset=KEYS).to_csv(out_path, index=False)
    print(f"Saved {len(final)} rows to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out", default="results/news_gkg_raw.csv")
    args = ap.parse_args()
    run(args.symbols, date.fromisoformat(args.start), date.fromisoformat(args.end), args.out)


if __name__ == "__main__":
    main()
