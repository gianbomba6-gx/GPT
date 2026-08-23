"""GDELT DOC 2.0 historical news adapter.

GDELT is used only as a historical article source. The adapter deliberately
keeps the provider boundary separate from feature engineering and enforces the
requested timestamp fence before returning rows.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import time
import requests
import pandas as pd
from .news_provider import NewsProvider, NewsQuery, validate_articles, NORMALIZED_COLUMNS

SEARCH_TERMS = {
    "STLAM.MI": '("Stellantis" OR "STLA")',
    "STLA": '("Stellantis" OR "STLA")',
    "SPCX": '("SpaceX" OR "SPCX")',
    "NVDA": '("NVIDIA" OR "NVDA")',
    "TSLA": '("Tesla" OR "TSLA")',
}

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_seen_date(value: str) -> pd.Timestamp:
    # GDELT normally returns UTC timestamps such as 20260820153000.
    return pd.to_datetime(str(value), format="%Y%m%d%H%M%S", utc=True, errors="coerce")


class GdeltNewsProvider(NewsProvider):
    name = "gdelt"

    def __init__(self, pause: float = 0.25, max_records: int = 250, timeout: int = 30):
        self.pause = pause
        self.max_records = min(max_records, 250)
        self.timeout = timeout

    def _request(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        q = SEARCH_TERMS.get(symbol.upper())
        if not q:
            raise ValueError(f"No GDELT search mapping for {symbol}")
        params = {
            "query": q,
            "mode": "artlist",
            "format": "json",
            "maxrecords": self.max_records,
            "sort": "datedesc",
            "startdatetime": start.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S"),
        }
        r = requests.get(API, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        return payload.get("articles", [])

    def _fetch_window(self, symbol: str, start: datetime, end: datetime, depth: int = 0) -> list[dict]:
        if end <= start:
            return []
        articles = self._request(symbol, start, end)
        time.sleep(self.pause)
        # Article-list responses are capped at 250. Split only when necessary.
        if len(articles) >= self.max_records and depth < 16 and (end - start) > timedelta(hours=1):
            mid = start + (end - start) / 2
            left = self._fetch_window(symbol, start, mid, depth + 1)
            right = self._fetch_window(symbol, mid + timedelta(seconds=1), end, depth + 1)
            return left + right
        return articles

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        start = pd.Timestamp(query.start).to_pydatetime()
        end = pd.Timestamp(query.end).to_pydatetime()
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        articles = self._fetch_window(query.symbol, start, end)
        rows = []
        for a in articles:
            published = _parse_seen_date(a.get("seendate", ""))
            if pd.isna(published):
                continue
            rows.append({
                "published_at": published,
                "symbol": query.symbol,
                "headline": a.get("title", ""),
                "source": a.get("domain", ""),
                "url": a.get("url", ""),
                "summary": "",
                "category": "",
                "sentiment": pd.NA,
                "intensity": pd.NA,
                "relevance": pd.NA,
                "novelty": pd.NA,
            })
        if not rows:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        out = pd.DataFrame(rows).drop_duplicates(subset=["published_at", "headline", "url"])
        return validate_articles(out, query)
