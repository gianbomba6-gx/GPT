"""GDELT DOC 2.0 news adapter with defensive HTTP/JSON handling."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import time
import requests
import pandas as pd
try:
    from .news_provider import NewsProvider, NewsQuery, validate_articles, NORMALIZED_COLUMNS
except ImportError:
    from news_provider import NewsProvider, NewsQuery, validate_articles, NORMALIZED_COLUMNS

SEARCH_TERMS = {
    "STLAM.MI": '("Stellantis" OR "STLA")',
    "STLA": '("Stellantis" OR "STLA")',
    "SPCX": '("SpaceX" OR "SPCX")',
    "NVDA": '("NVIDIA" OR "NVDA")',
    "TSLA": '("Tesla" OR "TSLA")',
}
API = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_seen_date(value: str) -> pd.Timestamp:
    return pd.to_datetime(str(value), format="%Y%m%d%H%M%S", utc=True, errors="coerce")


class GdeltNewsProvider(NewsProvider):
    name = "gdelt"

    def __init__(self, pause: float = 1.0, max_records: int = 250, timeout: int = 30, retries: int = 4):
        self.pause = max(pause, 0.25)
        self.max_records = min(max_records, 250)
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "market-rebound-model/1.0 (GDELT DOC client)",
            "Accept": "application/json",
        })

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
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(API, params=params, timeout=self.timeout)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "json" not in content_type:
                    body = response.text.strip().replace("\n", " ")[:500]
                    raise RuntimeError(
                        f"GDELT returned non-JSON response: status={response.status_code} "
                        f"content_type={content_type!r} body={body!r}"
                    )
                payload = response.json()
                if not isinstance(payload, dict):
                    raise RuntimeError(f"GDELT returned unexpected JSON type: {type(payload).__name__}")
                return payload.get("articles", []) or []
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"GDELT request failed for {symbol} {start.isoformat()} -> {end.isoformat()}: {last_error}"
        ) from last_error

    def _fetch_window(self, symbol: str, start: datetime, end: datetime, depth: int = 0) -> list[dict]:
        if end <= start:
            return []
        articles = self._request(symbol, start, end)
        time.sleep(self.pause)
        if len(articles) >= self.max_records and depth < 16 and (end - start) > timedelta(hours=1):
            mid = start + (end - start) / 2
            return (
                self._fetch_window(symbol, start, mid, depth + 1)
                + self._fetch_window(symbol, mid + timedelta(seconds=1), end, depth + 1)
            )
        return articles

    def fetch(self, query: NewsQuery) -> pd.DataFrame:
        start = pd.Timestamp(query.start).to_pydatetime()
        end = pd.Timestamp(query.end).to_pydatetime()
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        rows = []
        for article in self._fetch_window(query.symbol, start, end):
            published = _parse_seen_date(article.get("seendate", ""))
            if pd.isna(published):
                continue
            rows.append({
                "published_at": published,
                "symbol": query.symbol,
                "headline": article.get("title", ""),
                "source": article.get("domain", ""),
                "url": article.get("url", ""),
                "summary": article.get("socialimage", "") or "",
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
