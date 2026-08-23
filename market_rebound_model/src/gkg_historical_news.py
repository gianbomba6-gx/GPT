"""Historical GDELT GKG daily news features (2013-present)."""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import csv
import re
import zipfile

import pandas as pd
import requests

GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras",
]

COMPACT_COLUMNS = [
    "DATE", "DocumentIdentifier", "SourceCommonName", "V2Counts", "V2Themes",
    "V2Locations", "V2Persons", "V2Organizations", "V2Tone", "AllNames", "Extras",
]

NORMALIZED_COLUMNS = [
    "published_at", "symbol", "headline", "source", "url", "summary",
    "category", "sentiment", "intensity", "relevance", "novelty",
]

SYMBOL_ORGANIZATIONS = {
    "STLAM.MI": ("stellantis", "stellantis nv", "stellantis n.v.", "stla", "fiat chrysler", "fiat chrysler automobiles"),
    "SPCX": ("spacex", "space exploration technologies", "space exploration technologies corp"),
    "NVDA": ("nvidia", "nvidia corporation", "nvidia corp", "nvda"),
    "TSLA": ("tesla", "tesla motors", "tesla inc", "tesla, inc", "tsla"),
}

GKG_BASE = "https://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"
_TIMESTAMP_RE = re.compile(r"^\d{14}$")
_RECORD_RE = re.compile(r"^\d{14}-(?:T?\d+)$")
_TONE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){2,}")
_URL_RE = re.compile(r"^https?://", re.I)


def _tone(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(str(value).split(",", 1)[0])
    except (TypeError, ValueError):
        return None


def _compact_alnum(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _matches(text: object, terms: tuple[str, ...]) -> bool:
    normalized = _compact_alnum(text)
    if not normalized:
        return False
    return any(_compact_alnum(term) and _compact_alnum(term) in normalized for term in terms)


def _extract_row(fields: list[str], terms: tuple[str, ...]) -> dict | None:
    """Extract fields using explicit canonical/compact layouts plus safe fallbacks."""
    values = [str(x or "").strip() for x in fields]
    if len(values) < 4:
        return None

    if len(values) == len(GKG_COLUMNS):
        date_value = values[1]
        source = values[3]
        url = values[4]
        tone_value = values[15]
        summary = values[8]
        record_id = values[0]
        match_values = [values[13], values[14], values[23], "\t".join(values)]
    elif len(values) == len(COMPACT_COLUMNS):
        date_value = values[0]
        url = values[1]
        source = values[2]
        tone_value = values[8]
        summary = values[4]
        record_id = ""
        match_values = [values[7], values[9], "\t".join(values)]
    else:
        date_value = next((v for v in values if _TIMESTAMP_RE.fullmatch(v)), None)
        if date_value is None:
            record = next((v for v in values if _RECORD_RE.fullmatch(v)), None)
            if record:
                date_value = record[:14]
        if date_value is None:
            return None
        url = next((v for v in values if _URL_RE.match(v)), "")
        source = ""
        tone_value = next((v for v in values if _TONE_RE.fullmatch(v)), "")
        summary_candidates = [v for v in values if ";" in v and any(k in v.upper() for k in ("ECON_", "TAX_", "CRISISLEX", "WB_", "GENERAL_", "MEDIA_"))]
        summary = max(summary_candidates, key=len) if summary_candidates else ""
        record_id = next((v for v in values if _RECORD_RE.fullmatch(v)), "")
        match_values = values

    if not date_value or not _TIMESTAMP_RE.fullmatch(date_value):
        return None
    if not any(_matches(v, terms) for v in match_values):
        return None

    if not source and url:
        source = re.sub(r"^https?://", "", url, flags=re.I).split("/", 1)[0]

    return {
        "published_at": pd.to_datetime(date_value, format="%Y%m%d%H%M%S", utc=True, errors="coerce"),
        "headline": "",
        "source": source,
        "url": url,
        "summary": summary,
        "sentiment": _tone(tone_value),
        "_record_id": record_id,
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
                rows = [row for row in csv.reader((line.decode("utf-8", "replace") for line in fh), delimiter="\t")]
        if not rows:
            raise RuntimeError(f"Empty GKG data file for {day_s}")
        widths = pd.Series([len(r) for r in rows[:2000]]).value_counts().to_dict()
        if max(widths) < 4:
            raise RuntimeError(f"Invalid GDELT GKG row widths for {day_s}: {widths}")
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
