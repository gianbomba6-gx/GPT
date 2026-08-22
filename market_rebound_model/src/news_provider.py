"""Provider-neutral news ingestion interface for V2.

The live/historical model must receive timestamped articles. A provider adapter
can implement fetch() and return the normalized columns below. Keeping the
adapter boundary explicit prevents accidental use of post-cutoff information.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

NORMALIZED_COLUMNS = [
    "published_at", "symbol", "headline", "source", "url", "summary",
    "category", "sentiment", "intensity", "relevance", "novelty"
]

@dataclass
class NewsQuery:
    symbol: str
    start: datetime
    end: datetime

class NewsProvider:
    name = "abstract"
    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        raise NotImplementedError

def validate_articles(df: pd.DataFrame, query: NewsQuery) -> pd.DataFrame:
    missing = [c for c in NORMALIZED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Provider missing normalized columns: {missing}")
    out = df.copy()
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    out = out.dropna(subset=["published_at", "headline"]).copy()
    # Hard temporal fence: articles after the requested end are never accepted.
    start = pd.Timestamp(query.start, tz="UTC")
    end = pd.Timestamp(query.end, tz="UTC")
    out = out[(out["published_at"] >= start) & (out["published_at"] <= end)]
    out["symbol"] = query.symbol.upper()
    return out[NORMALIZED_COLUMNS]

class CsvNewsProvider(NewsProvider):
    """Offline provider used for reproducible tests/backfills."""
    name = "csv"
    def __init__(self, path: str):
        self.path = path
    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        return validate_articles(df, query)
