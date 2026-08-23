"""Historical GDELT GKG daily news features (2013-present).

The public daily GKG archives can contain layout variations, so rows are
parsed semantically instead of requiring one fixed field count.
"""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import csv
import re
import zipfile

import pandas as pd
import requests

# Public compatibility schema for canonical GKG 2.1 fixtures/tools. The
# production parser below does not require a fixed row width.
GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras",
]

NORMALIZED_COLUMNS = [
    "published_at", "symbol", "headline", "source", "url", "summary",
    "category", "sentiment", "intensity", "relevance", "novelty",
]

SYMBOL_ORGANIZATIONS = {
    "STLAM.MI": ("stellantis", "stla", "fiat chrysler", "fiat chrysler automobiles"),
    "SPCX": ("spacex", "space exploration technologies"),
    "NVDA": ("nvidia", "nvda"),
    "TSLA": ("tesla", "tesla motors", "tesla inc"),
}

GKG_BASE = "https://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"
_TIMESTAMP_RE = re.compile(r"^\d{14}$")
_RECORD_RE = re.compile(r"^\d{14}-(?:T?\d+)$")
_TONE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){2,}")
_URL_RE = re.compile(r"^https?://", re.I)
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?::\d+)?$")


def _tone(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(str(value).split(",", 1)[0])
    except (TypeError, ValueError):
        return None


def _matches(text: object, terms: tuple[str, ...]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    haystack = text.lower()
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", haystack)
        for term in terms
    )


def _extract_row(fields: list[str], terms: tuple[str, ...]) -> dict | None:
    if len(fields) < 4:
        return None
    values = [str(x or "").strip() for x in fields]

    record_idx = next((i for i, v in enumerate(values[:4]) if _RECORD_RE.match(v)), None)
    date_idx = next((i for i, v in enumerate(values[:6]) if _TIMESTAMP_RE.match(v)), None)
    if date_idx is None:
        return None

    url_idx = next((i for i, v in enumerate(values) if _URL_RE.match(v)), None)
    source_idx = next((i for i, v in enumerate(values[:8]) if _DOMAIN_RE.match(v)), None)
    if source_idx is None and url_idx is not None and url_idx > 0:
        source_idx = url_idx - 1

    entity_candidates = [v for v in values if _matches(v, terms)]
    if not entity_candidates:
        return None
    org_value = max(entity_candidates, key=lambda v: (v.count(";"), v.count(","), len(v)))

    tone_value = next((v for v in values if _TONE_RE.match(v)), "")
    themes_candidates = [
        v for v in values
        if ";" in v and any(token in v.upper() for token in ("ECON_", "TAX_", "CRISISLEX", "WB_"))
    ]
    summary = max(themes_candidates, key=len) if themes_candidates else ""

    return {
        "published_at": pd.to_datetime(values[date_idx], format="%Y%m%d%H%M%S", utc=True, errors="coerce"),
        "headline": "",
        "source": values[source_idx] if source_idx is not None else "",
        "url": values[url_idx] if url_idx is not None else "",
        "summary": summary,
        "sentiment": _tone(tone_value),
        "_record_id": values[record_idx] if record_idx is not None else "",
        "_org_value": org_value,
    }


class GkgHistoricalProvider:
    """Download one daily GKG file and extract company-level news metadata."""

    def __init__(self, timeout: int = 90):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "market-rebound-model/1.0 (GDELT GKG client)",
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.8",
        })
        self._day_cache: dict[str, list[list[str]]] = {}

    def _load_day(self, day: date) -> list[list[str]]:
        day_s = day.strftime("%Y%m%d")
        if day_s in self._day_cache:
            return self._day_cache[day_s]
        response = self.session.get(GKG_BASE.format(day=day_s), timeout=self.timeout)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            members = zf.namelist()
            if not members:
                raise RuntimeError(f"Empty GKG archive for {day_s}")
            with zf.open(members[0]) as fh:
                rows = [
                    row
                    for row in csv.reader(
                        (line.decode("utf-8", "replace") for line in fh),
                        delimiter="\t",
                    )
                ]
        if not rows:
            raise RuntimeError(f"Empty GKG data file for {day_s}")
        widths = pd.Series([len(r) for r in rows[:2000]]).value_counts().to_dict()
        dominant = max(widths, key=widths.get)
        if dominant < 4:
            raise RuntimeError(f"GDELT GKG archive {day_s} has invalid row width distribution: {widths}")
        self._day_cache[day_s] = rows
        return rows

    def fetch_day(self, symbol: str, day: date) -> pd.DataFrame:
        symbol = symbol.upper()
        terms = SYMBOL_ORGANIZATIONS.get(symbol)
        if not terms:
            raise ValueError(f"No GKG organization mapping for {symbol}")
        rows = self._load_day(day)
        out_rows = []
        for fields in rows:
            item = _extract_row(fields, terms)
            if item is None or pd.isna(item["published_at"]):
                continue
            item.update({
                "symbol": symbol,
                "category": "gkg",
                "intensity": abs(item["sentiment"]) if item["sentiment"] is not None else None,
                "relevance": 1.0,
                "novelty": pd.NA,
            })
            out_rows.append(item)
        if not out_rows:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        out = pd.DataFrame(out_rows)
        return (
            out[NORMALIZED_COLUMNS]
            .dropna(subset=["published_at"])
            .drop_duplicates(subset=["published_at", "url", "symbol"])
            .reset_index(drop=True)
        )

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if end < start:
            raise ValueError("end must be >= start")
        frames = []
        cursor = start
        while cursor <= end:
            print(f"GKG {symbol}: {cursor.isoformat()}")
            frames.append(self.fetch_day(symbol, cursor))
            cursor += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        return pd.concat(frames, ignore_index=True)
