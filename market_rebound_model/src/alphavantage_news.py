"""Alpha Vantage NEWS_SENTIMENT adapter with issuer mapping and quota-safe fetching."""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
import time
import requests
import pandas as pd
from news_provider import NewsQuery, NewsProvider, validate_articles

BASE_URL = "https://www.alphavantage.co/query"
MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "news_symbol_map.json"
EMPTY_COLUMNS = ["published_at","symbol","headline","source","url","summary","category","sentiment","intensity","relevance","novelty"]
API_LIMIT = 1000
MAX_DAILY_REQUESTS = 20
CHUNK_DAYS = 7
MIN_REQUEST_INTERVAL = 1.1

class AlphaVantageQuotaError(RuntimeError):
    """Provider quota/rate limit; never retry recursively on this response."""

class AlphaVantageNewsProvider(NewsProvider):
    name = "alphavantage"
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is required")
        self.timeout = timeout
        self.symbol_map = json.loads(MAP_PATH.read_text())
        self.request_count = 0
        self._last_request = 0.0

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

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

    def _request(self, provider_symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        if self.request_count >= MAX_DAILY_REQUESTS:
            raise AlphaVantageQuotaError(f"Safety stop: {MAX_DAILY_REQUESTS} Alpha Vantage requests reached in this run.")
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        params = {"function":"NEWS_SENTIMENT","tickers":provider_symbol,"time_from":start.strftime("%Y%m%dT%H%M"),"time_to":end.strftime("%Y%m%dT%H%M"),"sort":"EARLIEST","limit":API_LIMIT,"apikey":self.api_key}
        self.request_count += 1
        r = requests.get(BASE_URL, params=params, timeout=self.timeout)
        self._last_request = time.monotonic()
        r.raise_for_status()
        payload = r.json()
        if "feed" not in payload:
            safe_payload = {k:v for k,v in payload.items() if k.lower() not in {"apikey","key"}}
            text = str(safe_payload).lower()
            if any(x in text for x in ("rate limit","requests per day","spreading out","premium")):
                raise AlphaVantageQuotaError(f"Alpha Vantage quota/rate limit reached after {self.request_count} requests.")
            raise RuntimeError(f"Alpha Vantage response has no feed for {provider_symbol}: {safe_payload}")
        return payload["feed"] or []

    def _fetch_complete(self, provider_symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        results: list[dict] = []
        cursor = start
        while cursor <= end:
            window_end = min(cursor + pd.Timedelta(days=CHUNK_DAYS) - pd.Timedelta(minutes=1), end)
            feed = self._request(provider_symbol, cursor, window_end)
            if len(feed) >= API_LIMIT:
                midpoint = cursor + (window_end - cursor) / 2
                left = self._request(provider_symbol, cursor, midpoint.floor("min"))
                right_start = midpoint.floor("min") + pd.Timedelta(minutes=1)
                right = self._request(provider_symbol, right_start, window_end)
                results.extend(left + right)
            else:
                results.extend(feed)
            cursor = window_end + pd.Timedelta(minutes=1)
        return results

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        original_symbol = query.symbol
        provider_symbol = self.provider_symbol(original_symbol)
        start, end = self._utc_timestamp(query.start), self._utc_timestamp(query.end)
        feed = self._fetch_complete(provider_symbol, start, end)
        rows=[]
        for item in feed:
            ts=next((x for x in (item.get("ticker_sentiment",[]) or []) if str(x.get("ticker","" )).upper()==provider_symbol.upper()),None)
            score=ts.get("ticker_sentiment_score") if ts else item.get("overall_sentiment_score")
            relevance=ts.get("relevance_score") if ts else None
            score_num=pd.to_numeric(score,errors="coerce")
            rows.append({"published_at":item.get("time_published"),"symbol":original_symbol,"headline":item.get("title",""),"source":item.get("source",""),"url":item.get("url",""),"summary":item.get("summary",""),"category":item.get("category","other"),"sentiment":score_num,"intensity":abs(score_num) if pd.notna(score_num) else None,"relevance":pd.to_numeric(relevance,errors="coerce"),"novelty":None})
        if not rows: return pd.DataFrame(columns=EMPTY_COLUMNS)
        df=pd.DataFrame(rows)
        df["published_at"]=pd.to_datetime(df["published_at"],format="%Y%m%dT%H%M%S",utc=True,errors="coerce")
        return validate_articles(df,query).drop_duplicates(subset=["published_at","headline","url"])
