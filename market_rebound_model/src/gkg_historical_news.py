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
    "STLAM.MI": (
        "stellantis", "stellantis nv", "stellantis n.v.", "stla",
        "fiat chrysler", "fiat chrysler automobiles",
    ),
    "SPCX": (
        "spacex", "space exploration technologies",
        "space exploration technologies corp",
    ),
    "NVDA": (
        "nvidia", "nvidia corporation", "nvidia corp", "nvda",
    ),
    "TSLA": (
        "tesla", "tesla motors", "tesla inc", "tesla, inc", "tsla",
    ),
}

GKG_BASE = "https://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"
_TIMESTAMP_RE = re.compile(r"^\d{14}$")
_TONE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){2,}")


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
    """Match issuer names robustly across punctuation, suffixes and spacing."""
    normalized = _compact_alnum(text)
    if not normalized:
        return False
    return any(_compact_alnum(term) and _compact_alnum(term) in normalized for term in terms)


def _canonical_from_fields(fields: list[str]) -> dict[str, str] | None:
    if len(fields) < 20:
        return None
    return {
        "record_id": fields[0],
        "date": fields[1],
        "source": fields[3],
        "url": fields[4],
        "themes": fields[8],
        "organizations": fields[13],
        "v2organizations": fields[14],
        "tone": fields[15],
        "all_names": fields[23] if len(fields) > 23 else "",
        "all_fields": "\t".join(fields),
    }


def _compact_from_fields(fields: list[str]) -> dict[str, str] | None:
    if len(fields) != len(COMPACT_COLUMNS):
        return None
    row = dict(zip(COMPACT_COLUMNS, fields))
    return {
        "record_id": "",
        "date": row["DATE"],
        "source": row["SourceCommonName"],
        "url": row["DocumentIdentifier"],
        "themes": row["V2Themes"],
        "organizations": row["V2Organizations"],
        "v2organizations": row["V2Organizations"],
        "tone": row["V2Tone"],
        "all_names": row["AllNames"],
        "all_fields": "\t".join(fields),
    }


def _extract_row(fields: list[str], terms: tuple[str, ...]) -> dict | None:
    values = [str(x or "").strip() for x in fields]
    row = _canonical_from_fields(values) if len(values) >= 20 else _compact_from_fields(values)
    if row is None or not _TIMESTAMP_RE.match(row["date"]):
        return None

    # Primary match: the GDELT organization fields. Fallback: scan the
    # complete record because some current public extracts have layout
    # differences or sparsely populated organization fields.
    primary_values = [row["organizations"], row["v2organizations"], row["all_names"]]
    matched = any(_matches(value, terms) for value in primary_values)
    if not matched:
        matched = _matches(row["all_fields"], terms)
    if not matched:
        return None

    return {
        "published_at": pd.to_datetime(
            row["date"], format="%Y%m%d%H%M%S", utc=True, errors="coerce"
        ),
        "headline": "",
        "source": row["source"],
        "url": row["url"],
        "summary": row["themes"] or "",
        "sentiment": _tone(row["tone"]),
        "_record_id": row["record_id"],
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
                    row for row in csv.reader(
                        (line.decode("utf-8", "replace") for line in fh), delimiter="\t"
                    )
                ]
        if not rows:
            raise RuntimeError(f"Empty GKG data file for {day_s}")
        widths = pd.Series([len(r) for r in rows[:2000]]).value_counts().to_dict()
        supported = {11, 27}
        if not set(widths).intersection(supported):
            raise RuntimeError(f"Unsupported GDELT GKG row widths for {day_s}: {widths}")
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
