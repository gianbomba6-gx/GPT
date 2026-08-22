"""Alpha Vantage NEWS_SENTIMENT adapter.

Alpha Vantage documents time_from/time_to historical filtering and ticker
filtering. The adapter preserves publication timestamps and provider metadata.
API credentials are read only from ALPHAVANTAGE_API_KEY.
"""
from __future__ import annotations
import os
from datetime import datetime
import requests
import pandas as pd
from news_provider import NewsQuery, NewsProvider, validate_articles

BASE_URL = "https://www.alphavantage.co/query"

class AlphaVantageNewsProvider(NewsProvider):
    name = "alphavantage"
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is required")
        self.timeout = timeout

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        start = self._utc_timestamp(query.start)
        end = self._utc_timestamp(query.end)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": query.symbol,
            "time_from": start.strftime("%Y%m%dT%H%M"),
            "time_to": end.strftime("%Y%m%dT%H%M"),
            "sort": "EARLIEST",
            "limit": 1000,
            "apikey": self.api_key,
        }
        r = requests.get(BASE_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        if "feed" not in payload:
            raise RuntimeError(f"Alpha Vantage response has no feed: {payload}")
        rows = []
        for item in payload["feed"]:
            rows.append({
                "published_at": item.get("time_published"),
                "symbol": query.symbol,
                "headline": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
                "category": item.get("category", "other"),
                "sentiment": item.get("overall_sentiment_score"),
                "intensity": None,
                "relevance": None,
                "novelty": None,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["published_at","symbol","headline","source","url","summary","category","sentiment","intensity","relevance","novelty"])
        df["published_at"] = pd.to_datetime(df["published_at"], format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
        return validate_articles(df, query)
