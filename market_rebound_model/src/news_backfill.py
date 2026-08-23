"""Build daily news features from the raw historical news dataset.

Accepts either CSV or JSON input and supports both the legacy positional
invocation and explicit --input/--output arguments used by GitHub Actions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "news_daily.csv"
RAW_COLUMNS = ["published_at", "symbol", "headline", "source", "url", "summary", "category", "sentiment", "intensity", "relevance", "novelty"]

def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"News input not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else data.get("articles", data.get("feed", []))
        d = pd.DataFrame(rows)
    else:
        d = pd.read_csv(p)
    if d.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    # Accept both provider/raw naming and normalized naming.
    if "title" in d.columns and "headline" not in d.columns:
        d = d.rename(columns={"title": "headline"})
    for c in RAW_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[RAW_COLUMNS].copy()
    d["published_at"] = pd.to_datetime(d["published_at"], utc=True, errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    for c in ["sentiment", "intensity", "relevance", "novelty"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=["published_at", "symbol"])

def build_daily_features(articles: pd.DataFrame) -> pd.DataFrame:
    columns = ["Date", "symbol", "news_sentiment", "news_intensity", "news_relevance", "news_novelty", "news_count"]
    if articles.empty:
        return pd.DataFrame(columns=columns)
    x = articles.copy()
    x["Date"] = x["published_at"].dt.normalize()
    daily = (x.groupby(["Date", "symbol"], as_index=False)
        .agg(news_sentiment=("sentiment", "mean"),
             news_intensity=("intensity", "mean"),
             news_relevance=("relevance", "mean"),
             news_novelty=("novelty", "mean"),
             news_count=("headline", "size")))
    return daily

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_positional", nargs="?", help="Raw news CSV/JSON")
    ap.add_argument("--input", dest="input_arg")
    ap.add_argument("--output", "--out", dest="output", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    input_path = args.input_arg or args.input_positional
    if not input_path:
        raise SystemExit("An input news dataset is required")
    articles = load_input(input_path)
    daily = build_daily_features(articles)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False)
    print(f"Loaded {len(articles)} raw articles")
    print(f"Saved {len(daily)} daily symbol rows to {out}")

if __name__ == "__main__":
    main()
