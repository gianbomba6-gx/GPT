"""Historical GDELT GKG daily news features."""
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
    "STLAM.MI": ("stellantis", "stellantis nv", "stellantis n.v.", "fiat chrysler", "fiat chrysler automobiles"),
    "SPCX": ("spacex", "space exploration technologies", "space exploration technologies corp"),
    "NVDA": ("nvidia", "nvidia corporation", "nvidia corp"),
    "TSLA": ("tesla", "tesla motors", "tesla inc", "tesla, inc"),
}
GKG_BASE = "https://data.gdeltproject.org/gkg/{day}.gkg.csv.zip"
_TONE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){2,}")


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tone(value: object) -> float | None:
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        return float(str(value).split(",", 1)[0])
    except (TypeError, ValueError):
        return None


def _org_names(value: object) -> list[str]:
    text = str(value or "")
    return [part.split(",", 1)[0].strip() for part in text.split(";") if part.split(",", 1)[0].strip()]


def _matches_org_field(value: object, aliases: tuple[str, ...]) -> bool:
    names = _org_names(value)
    alias_norms = {_norm(a) for a in aliases if _norm(a)}
    return any(any(a == _norm(name) or a in _norm(name) for a in alias_norms) for name in names)


def _matches_row(row: pd.Series, aliases: tuple[str, ...]) -> bool:
    return (
        _matches_org_field(row["V2Organizations"], aliases)
        or _matches_org_field(row["Organizations"], aliases)
        or _matches_org_field(row["AllNames"], aliases)
    )


def _row_to_article(row: pd.Series, aliases: tuple[str, ...], symbol: str) -> dict | None:
    if not _matches_row(row, aliases):
        return None
    published = pd.to_datetime(str(row["DATE"]), format="%Y%m%d%H%M%S", utc=True, errors="coerce")
    if pd.isna(published):
        return None
    tone = _tone(row["V2Tone"])
    return {
        "published_at": published,
        "symbol": symbol,
        "headline": "",
        "source": str(row["SourceCommonName"] or "").strip(),
        "url": str(row["DocumentIdentifier"] or "").strip(),
        "summary": str(row["V2Themes"] or "").strip(),
        "category": "gkg",
        "sentiment": tone,
        "intensity": abs(tone) if tone is not None else None,
        "relevance": 1.0,
        "novelty": pd.NA,
    }


class GkgHistoricalProvider:
    def __init__(self, timeout: int = 90):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "market-rebound-model/1.0 (GDELT GKG client)",
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.8",
        })
        self._day_cache: dict[str, pd.DataFrame] = {}

    def _load_day(self, day: date) -> pd.DataFrame:
        day_s = day.strftime("%Y%m%d")
        if day_s in self._day_cache:
            return self._day_cache[day_s]
        response = self.session.get(GKG_BASE.format(day=day_s), timeout=self.timeout)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            if not members:
                raise RuntimeError(f"Empty GKG archive for {day_s}")
            with zf.open(members[0]) as fh:
                first_line = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
                if not first_line:
                    raise RuntimeError(f"Empty GKG data file for {day_s}")
                width = len(next(csv.reader([first_line], delimiter="\t")))
                fh.seek(0)
                if width == len(GKG_COLUMNS):
                    df = pd.read_csv(
                        fh, sep="\t", header=None, names=GKG_COLUMNS,
                        usecols=[1, 3, 4, 8, 13, 14, 15, 23], dtype=str,
                        keep_default_na=False, na_filter=False,
                    )
                elif width == len(COMPACT_COLUMNS):
                    df = pd.read_csv(
                        fh, sep="\t", header=None, names=COMPACT_COLUMNS,
                        dtype=str, keep_default_na=False, na_filter=False,
                    )
                else:
                    raise RuntimeError(
                        f"Unsupported GDELT GKG row width for {day_s}: found {width}; expected {len(GKG_COLUMNS)} or {len(COMPACT_COLUMNS)}"
                    )
        if df.empty:
            raise RuntimeError(f"Empty GKG data file for {day_s}")
        if list(df.columns) == GKG_COLUMNS[:]:
            pass
        else:
            df = df.rename(columns={
                "V2Organizations": "V2Organizations",
                "V2Tone": "V2Tone",
                "V2Themes": "V2Themes",
                "SourceCommonName": "SourceCommonName",
                "DocumentIdentifier": "DocumentIdentifier",
                "DATE": "DATE",
            })
            for col in ("Organizations", "AllNames"):
                if col not in df.columns:
                    df[col] = ""
        self._day_cache[day_s] = df
        return df

    def fetch_day(self, symbol: str, day: date) -> pd.DataFrame:
        symbol = symbol.upper()
        aliases = SYMBOL_ORGANIZATIONS.get(symbol)
        if not aliases:
            raise ValueError(f"No GKG organization mapping for {symbol}")
        df = self._load_day(day)
        rows = []
        for _, row in df.iterrows():
            item = _row_to_article(row, aliases, symbol)
            if item is not None:
                rows.append(item)
        if not rows:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        out = pd.DataFrame(rows)
        return out[NORMALIZED_COLUMNS].drop_duplicates(subset=["published_at", "url", "symbol"]).reset_index(drop=True)

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if end < start:
            raise ValueError("end must be >= start")
        frames = []
        cursor = start
        while cursor <= end:
            print(f"GKG {symbol}: {cursor.isoformat()}")
            frames.append(self.fetch_day(symbol, cursor))
            cursor += timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=NORMALIZED_COLUMNS)
