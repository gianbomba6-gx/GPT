"""Leakage-safe historical news backfill interface for rebound research."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "news_raw.csv"
COLUMNS = ["published_at","symbol","title","source","url","summary","category","sentiment","intensity","relevance","novelty"]

def normalize_articles(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    for c in COLUMNS:
        if c not in d.columns: d[c] = ""
    d = d[COLUMNS].copy()
    d["published_at"] = pd.to_datetime(d["published_at"], utc=True, errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    for c in ["sentiment","intensity","relevance","novelty"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["published_at","symbol"]).drop_duplicates(subset=["published_at","symbol","title"])

def apply_cutoff(articles: pd.DataFrame, signal_date: str, cutoff_utc: str | None = None) -> pd.DataFrame:
    cutoff = (pd.Timestamp(cutoff_utc, tz="UTC") if cutoff_utc else pd.Timestamp(signal_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return articles[articles["published_at"] <= cutoff].copy()

def build_daily_features(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return pd.DataFrame(columns=["Date","symbol","news_sentiment","news_intensity","news_relevance","news_novelty","news_count"])
    x = articles.copy(); x["Date"] = x["published_at"].dt.normalize()
    return (x.groupby(["Date","symbol"], as_index=False)
        .agg(news_sentiment=("sentiment","mean"), news_intensity=("intensity","mean"),
             news_relevance=("relevance","mean"), news_novelty=("novelty","mean"), news_count=("title","size")))

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input"); ap.add_argument("--output", default=str(DEFAULT_OUT)); args = ap.parse_args()
    if not args.input:
        print("No provider configured: schema/cutoff validation only."); return
    rows = json.loads(Path(args.input).read_text()); d = normalize_articles(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); d.to_csv(args.output, index=False)
    print(f"Saved {len(d)} normalized historical articles to {args.output}")

if __name__ == "__main__": main()
