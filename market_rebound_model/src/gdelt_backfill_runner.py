"""Checkpointed historical news backfill using GDELT DOC 2.0."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
try:
    from .gdelt_news import GdeltNewsProvider
    from .news_provider import NewsQuery, NORMALIZED_COLUMNS
except ImportError:
    from gdelt_news import GdeltNewsProvider
    from news_provider import NewsQuery, NORMALIZED_COLUMNS

DEFAULT_SYMBOLS = ["STLAM.MI", "SPCX", "NVDA", "TSLA"]


def run(symbols: list[str], start: datetime, end: datetime, out: str, pause: float, window_days: int) -> None:
    provider = GdeltNewsProvider(pause=pause)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = out_path.with_suffix(out_path.suffix + ".checkpoint.csv")
    existing = pd.read_csv(checkpoint) if checkpoint.exists() else (pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=NORMALIZED_COLUMNS))
    completed = set(existing.get("_window_key", pd.Series(dtype=str)).astype(str))
    chunks = [existing.drop(columns=["_window_key"], errors="ignore")] if not existing.empty else []
    cursor = start
    while cursor < end:
        w_end = min(cursor + timedelta(days=window_days), end)
        for symbol in symbols:
            key = f"{symbol}|{cursor.isoformat()}|{w_end.isoformat()}"
            if key in completed:
                continue
            print(f"GDELT {symbol}: {cursor.date()} -> {w_end.date()}")
            df = provider.fetch(NewsQuery(symbol=symbol, start=cursor, end=w_end)).copy()
            df["_window_key"] = key
            chunks.append(df)
            completed.add(key)
            checkpoint_df = pd.concat(chunks, ignore_index=True)
            checkpoint_df.to_csv(checkpoint, index=False)
            checkpoint_df.drop(columns=["_window_key"], errors="ignore").drop_duplicates(
                subset=["published_at", "symbol", "headline", "url"]
            ).to_csv(out_path, index=False)
            print(f"  articles={len(df)}")
        cursor = w_end
    final = pd.concat(chunks, ignore_index=True).drop(columns=["_window_key"], errors="ignore").drop_duplicates(
        subset=["published_at", "symbol", "headline", "url"]
    )
    final.to_csv(out_path, index=False)
    print(f"Saved {len(final)} articles to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out", default="results/news_gdelt_raw.csv")
    ap.add_argument("--pause", type=float, default=0.25)
    ap.add_argument("--window-days", type=int, default=30)
    args = ap.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if end <= start:
        raise SystemExit("--end must be after --start")
    run(args.symbols, start, end, args.out, args.pause, args.window_days)

if __name__ == "__main__":
    main()
