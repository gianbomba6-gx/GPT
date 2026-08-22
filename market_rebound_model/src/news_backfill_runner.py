"""Historical news backfill runner for the V2 validation universe."""
from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from alphavantage_news import AlphaVantageNewsProvider
from news_provider import NewsQuery

DEFAULT_SYMBOLS = ["STLAM.MI", "SPCX", "NVDA", "TSLA"]

def run(symbols, start, end, out, pause=1.0):
    provider = AlphaVantageNewsProvider()
    all_rows = []
    for symbol in symbols:
        q = NewsQuery(symbol=symbol, start=start, end=end)
        df = provider.fetch(q)
        print(f"NEWS {symbol}: {len(df)} articles")
        if not df.empty:
            all_rows.append(df)
        time.sleep(pause)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved {out}: {len(result)} articles")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01T00:00:00+00:00")
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out", default="results/news_raw.csv")
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args()
    run(args.symbols, datetime.fromisoformat(args.start), datetime.fromisoformat(args.end), args.out, args.pause)
