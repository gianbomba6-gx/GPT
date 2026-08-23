"""Alpha Vantage NEWS_SENTIMENT adapter with explicit issuer symbol mapping."""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
from news_provider import NewsQuery, NewsProvider, validate_articles

BASE_URL = "https://www.alphavantage.co/query"
MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "news_symbol_map.json"

EMPTY_COLUMNS = ["published_at","symbol","headline","source","url","summary","category","sentiment","intensity","relevance","novelty"]

class AlphaVantageNewsProvider(NewsProvider):
    name = "alphavantage"
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is required")
        self.timeout = timeout
        self.symbol_map = json.loads(MAP_PATH.read_text())

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def provider_symbol(self, symbol: str) -> str:
        try:
            return self.symbol_map[symbol]["provider_symbol"]
        except KeyError as exc:
            raise ValueError(f"No Alpha Vantage news mapping configured for {symbol}") from exc

    def validate_mapping(self, symbols: list[str]) -> None:
        missing = [s for s in symbols if s not in self.symbol_map]
        if missing:
            raise ValueError(f"Missing Alpha Vantage news mappings: {missing}")
        for symbol in symbols:
            provider_symbol = self.provider_symbol(symbol)
            if not provider_symbol.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid Alpha Vantage NEWS_SENTIMENT ticker mapping: {symbol} -> {provider_symbol}")

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        original_symbol = query.symbol
        provider_symbol = self.provider_symbol(original_symbol)
        start = self._utc_timestamp(query.start)
        end = self._utc_timestamp(query.end)
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": provider_symbol,
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
            safe_payload = {k: v for k, v in payload.items() if k.lower() not in {"apikey", "key"}}
            raise RuntimeError(f"Alpha Vantage response has no feed for {original_symbol} ({provider_symbol}): {safe_payload}")

        rows = []
        for item in payload["feed"]:
            ticker_sentiment = None
            for ts in item.get("ticker_sentiment", []) or []:
                if str(ts.get("ticker", "")).upper() == provider_symbol.upper():
                    ticker_sentiment = ts
                    break
            score = ticker_sentiment.get("ticker_sentiment_score") if ticker_sentiment else item.get("overall_sentiment_score")
            relevance = ticker_sentiment.get("relevance_score") if ticker_sentiment else None
            score_num = pd.to_numeric(score, errors="coerce")
            rows.append({
                "published_at": item.get("time_published"),
                "symbol": original_symbol,
                "headline": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
                "category": item.get("category", "other"),
                "sentiment": score_num,
                "intensity": abs(score_num) if pd.notna(score_num) else None,
                "relevance": pd.to_numeric(relevance, errors="coerce"),
                "novelty": None,
            })
        if not rows:
            return pd.DataFrame(columns=EMPTY_COLUMNS)
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
        return validate_articles(df, query)
