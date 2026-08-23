"""Historical news backfill runner with explicit provider selection."""
from __future__ import annotations
import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from alphavantage_news import AlphaVantageNewsProvider
from finnhub_news import FinnhubNewsProvider
from news_provider import NewsQuery

DEFAULT_SYMBOLS = ["STLAM.MI", "SPCX", "NVDA", "TSLA"]


def build_provider(name: str):
    if name == "finnhub":
        return FinnhubNewsProvider()
    if name == "alphavantage":
        return AlphaVantageNewsProvider()
    raise ValueError(f"Unknown news provider: {name}")


def run(symbols, start, end, out, pause=1.0, provider_name="auto"):
    if end <= start:
        raise ValueError("--end must be later than --start")

    if provider_name == "auto":
        if os.environ.get("FINNHUB_API_KEY"):
            provider_name = "finnhub"
        elif os.environ.get("ALPHAVANTAGE_API_KEY"):
            provider_name = "alphavantage"
        else:
            raise RuntimeError("Set FINNHUB_API_KEY or ALPHAVANTAGE_API_KEY")

    provider = build_provider(provider_name)
    if hasattr(provider, "validate_mapping"):
        provider.validate_mapping(symbols)

    print(f"NEWS PROVIDER: {provider.name}")
    all_rows = []
    for symbol in symbols:
        query_symbol = symbol
        if provider.name == "finnhub" and symbol == "STLAM.MI":
            query_symbol = "STLA"
        q = NewsQuery(symbol=query_symbol, start=start, end=end)
        df = provider.fetch(q)
        if not df.empty:
            df["symbol"] = symbol
            all_rows.append(df)
        print(f"NEWS {symbol}: {len(df)} articles")
        time.sleep(pause)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not result.empty:
        result = result.drop_duplicates(subset=["symbol", "published_at", "headline", "url"])
        result = result.sort_values(["symbol", "published_at"], kind="stable")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved {out}: {len(result)} articles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-23T00:00:00+00:00")
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out", default="results/news_raw.csv")
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--provider", choices=["auto", "finnhub", "alphavantage"], default="auto")
    args = ap.parse_args()
    run(
        args.symbols,
        datetime.fromisoformat(args.start),
        datetime.fromisoformat(args.end),
        args.out,
        args.pause,
        args.provider,
    )
