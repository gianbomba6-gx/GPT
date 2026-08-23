"""Finnhub company-news adapter.

Used as the preferred historical source for North-American listings. Finnhub's
free company-news tier is limited to one year of historical news, so this
provider is intentionally bounded by the requested interval and never claims
coverage outside the provider's documented window.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
import requests
import pandas as pd
from news_provider import NewsQuery, NewsProvider, validate_articles

BASE_URL = "https://finnhub.io/api/v1/company-news"
EMPTY_COLUMNS = ["published_at","symbol","headline","source","url","summary","category","sentiment","intensity","relevance","novelty"]

class FinnhubNewsProvider(NewsProvider):
    name = "finnhub"
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY is required")
        self.timeout = timeout

    @staticmethod
    def _date(value: datetime) -> str:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d")

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        params = {
            "symbol": query.symbol,
            "from": self._date(query.start),
            "to": self._date(query.end),
            "token": self.api_key,
        }
        r = requests.get(BASE_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Finnhub company-news response is not a list: {payload}")
        rows = []
        for item in payload:
            rows.append({
                "published_at": pd.to_datetime(item.get("datetime"), unit="s", utc=True, errors="coerce"),
                "symbol": query.symbol,
                "headline": item.get("headline", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
                "category": item.get("category", "other"),
                "sentiment": None,
                "intensity": None,
                "relevance": None,
                "novelty": None,
            })
        if not rows:
            return pd.DataFrame(columns=EMPTY_COLUMNS)
        df = pd.DataFrame(rows)
        return validate_articles(df, query).drop_duplicates(subset=["published_at","headline","url"])
