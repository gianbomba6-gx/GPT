"""Historical GDELT GKG daily news features (2013-present)."""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
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

SYMBOL_ORGANIZATIONS = {
    "STLAM.MI": ("stellantis", "stla", "fiat chrysler", "fca"),
    "SPCX": ("spacex",),
    "NVDA": ("nvidia", "nvda"),
    "TSLA": ("tesla", "tesla motors", "tesla inc"),
}

GKG_BASE = "https://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"


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
    for term in terms:
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", haystack):
            return True
    return False


class GkgHistoricalProvider:
    """Download one daily GKG file and extract company-level news metadata."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "market-rebound-model/1.0 (GDELT GKG client)"})

    def fetch_day(self, symbol: str, day: date) -> pd.DataFrame:
        symbol = symbol.upper()
        terms = SYMBOL_ORGANIZATIONS.get(symbol)
        if not terms:
            raise ValueError(f"No GKG organization mapping for {symbol}")
        day_s = day.strftime("%Y%m%d")
        response = self.session.get(GKG_BASE.format(day=day_s), timeout=self.timeout)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            members = zf.namelist()
            if not members:
                raise RuntimeError(f"Empty GKG archive for {day_s}")
            with zf.open(members[0]) as fh:
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    names=GKG_COLUMNS,
                    usecols=["DATE", "SourceCommonName", "DocumentIdentifier", "Themes", "V2Themes", "Organizations", "V2Organizations", "V2Tone"],
                    dtype=str,
                    on_bad_lines="skip",
                    low_memory=False,
                )
        org_mask = (
            df["Organizations"].map(lambda x: _matches(x, terms))
            | df["V2Organizations"].map(lambda x: _matches(x, terms))
        )
        out = df.loc[org_mask].copy()
        if out.empty:
            return pd.DataFrame(columns=["published_at", "symbol", "headline", "source", "url", "summary", "category", "sentiment", "intensity", "relevance", "novelty"])
        out["published_at"] = pd.to_datetime(out["DATE"], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
        out["symbol"] = symbol
        out["headline"] = ""
        out["source"] = out["SourceCommonName"].fillna("")
        out["url"] = out["DocumentIdentifier"].fillna("")
        out["summary"] = out["V2Themes"].fillna(out["Themes"]).fillna("")
        out["category"] = "gkg"
        out["sentiment"] = out["V2Tone"].map(_tone)
        out["intensity"] = out["V2Tone"].map(lambda x: _tone(x) if x is not None else None)
        out["relevance"] = 1.0
        out["novelty"] = pd.NA
        return out[["published_at", "symbol", "headline", "source", "url", "summary", "category", "sentiment", "intensity", "relevance", "novelty"]].dropna(subset=["published_at"]).drop_duplicates(subset=["published_at", "url"])

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
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
