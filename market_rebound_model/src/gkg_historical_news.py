"""Historical GDELT GKG news features.

Supports legacy GKG 1.0 daily files (11-column), compact 11-column
fixtures, and GKG 2.x 27-column files. The provider caches each daily archive,
retries transient HTTP failures, and uses vectorized organization matching so
large historical GKG files do not require a full ``iterrows()`` scan.

Historical GDELT archives can have occasional missing days. Those gaps are
reported and skipped so one unavailable archive does not invalidate an entire
year or the full backfill.
"""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import csv
import re
import time
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
GKG1_COLUMNS = [
    "DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS", "PERSONS",
    "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS", "SOURCES", "SOURCEURLS",
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
_URL_RE = re.compile(r"^https?://", re.I)


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _tone(value: object) -> float | None:
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        return float(str(value).split(",", 1)[0])
    except (TypeError, ValueError):
        return None


def _parse_gkg_date(value: object) -> pd.Timestamp:
    text = str(value or "").strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d"):
        parsed = pd.to_datetime(text, format=fmt, utc=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    return pd.NaT


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
    published = _parse_gkg_date(row["DATE"])
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
    def __init__(self, timeout: int = 90, retries: int = 5, pause: float = 1.5):
        self.timeout = timeout
        self.retries = retries
        self.pause = pause
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "market-rebound-model/1.0 (GDELT GKG client)",
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.8",
        })
        self._day_cache: dict[str, pd.DataFrame] = {}
        self._missing_days: set[str] = set()
        self._last_request_at = 0.0

    @property
    def missing_days(self) -> list[str]:
        return sorted(self._missing_days)

    def _request_archive(self, day_s: str) -> bytes:
        url = GKG_BASE.format(day=day_s)
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.pause:
                time.sleep(self.pause - elapsed)
            self._last_request_at = time.monotonic()
            response = self.session.get(url, timeout=self.timeout)
            status = response.status_code
            if status == 200:
                return response.content
            if status == 404:
                raise FileNotFoundError(f"GDELT GKG archive not found for {day_s}: {url}")
            if status == 429 or 500 <= status < 600:
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"GDELT GKG transient HTTP {status} for {day_s} after {self.retries + 1} attempts"
                    )
                retry_after = response.headers.get("Retry-After", "")
                try:
                    wait = max(float(retry_after), 0.0) if retry_after else 0.0
                except ValueError:
                    wait = 0.0
                wait = max(wait, min(60.0, 2.0 ** attempt))
                time.sleep(wait)
                continue
            raise RuntimeError(f"GDELT GKG HTTP {status} for {day_s}")
        raise RuntimeError(f"GDELT GKG request failed for {day_s}")

    def _load_day(self, day: date) -> pd.DataFrame:
        day_s = day.strftime("%Y%m%d")
        if day_s in self._day_cache:
            return self._day_cache[day_s]

        content = self._request_archive(day_s)
        with zipfile.ZipFile(BytesIO(content)) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            if not members:
                raise RuntimeError(f"Empty GKG archive for {day_s}")
            with zf.open(members[0]) as fh:
                first_line = fh.readline().decode("utf-8", "replace").rstrip("\r\n")
                if not first_line:
                    raise RuntimeError(f"Empty GKG data file for {day_s}")
                first_fields = next(csv.reader([first_line], delimiter="\t"))
                width = len(first_fields)
                fh.seek(0)
                if width == len(GKG_COLUMNS):
                    raw = pd.read_csv(
                        fh, sep="\t", header=None, names=GKG_COLUMNS,
                        usecols=[1, 3, 4, 8, 13, 14, 15, 23], dtype=str,
                        keep_default_na=False, na_filter=False,
                    )
                    df = raw
                elif width == len(GKG1_COLUMNS) and len(first_fields) > 1 and _URL_RE.match(first_fields[1].strip()):
                    raw = pd.read_csv(
                        fh, sep="\t", header=None, names=COMPACT_COLUMNS,
                        dtype=str, keep_default_na=False, na_filter=False,
                    )
                    df = raw.assign(Organizations="")[[
                        "DATE", "SourceCommonName", "DocumentIdentifier", "V2Themes",
                        "Organizations", "V2Organizations", "V2Tone", "AllNames",
                    ]]
                elif width == len(GKG1_COLUMNS):
                    raw = pd.read_csv(
                        fh, sep="\t", header=None, names=GKG1_COLUMNS,
                        dtype=str, keep_default_na=False, na_filter=False,
                    )
                    df = pd.DataFrame({
                        "DATE": raw["DATE"],
                        "SourceCommonName": raw["SOURCES"],
                        "DocumentIdentifier": raw["SOURCEURLS"],
                        "V2Themes": raw["THEMES"],
                        "Organizations": raw["ORGANIZATIONS"],
                        "V2Organizations": "",
                        "V2Tone": raw["TONE"],
                        "AllNames": "",
                    })
                else:
                    raise RuntimeError(
                        f"Unsupported GDELT GKG row width for {day_s}: "
                        f"found {width}; expected {len(GKG_COLUMNS)} or {len(GKG1_COLUMNS)}"
                    )

        if df.empty:
            raise RuntimeError(f"Empty GKG data file for {day_s}")
        self._day_cache[day_s] = df
        return df

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        return (
            series.fillna("").astype(str).str.lower()
            .str.replace(r"[^a-z0-9]+", "", regex=True)
        )

    def fetch_day_multi(self, symbols: list[str], day: date) -> pd.DataFrame:
        """Read one GKG day once and match all requested symbols vectorially.

        Missing historical GKG archives are treated as data gaps, not fatal
        workflow errors. The missing date is recorded in ``missing_days`` and
        an empty normalized frame is returned.
        """
        normalized_symbols = [s.upper() for s in symbols]
        aliases = {}
        for symbol in normalized_symbols:
            mapping = SYMBOL_ORGANIZATIONS.get(symbol)
            if not mapping:
                raise ValueError(f"No GKG organization mapping for {symbol}")
            aliases[symbol] = mapping

        day_s = day.strftime("%Y%m%d")
        try:
            df = self._load_day(day)
        except FileNotFoundError:
            self._missing_days.add(day_s)
            print(f"GKG MISSING ARCHIVE {day_s}: skipped")
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)

        if df.empty:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)

        normalized_fields = [self._normalize_series(df[column]) for column in ("V2Organizations", "Organizations", "AllNames")]
        searchable = normalized_fields[0] + "|" + normalized_fields[1] + "|" + normalized_fields[2]
        rows = []
        for symbol, symbol_aliases in aliases.items():
            alias_norms = sorted({_norm(alias) for alias in symbol_aliases if _norm(alias)}, key=len, reverse=True)
            pattern = "|".join(re.escape(alias) for alias in alias_norms)
            if not pattern:
                continue
            mask = searchable.str.contains(pattern, regex=True, na=False)
            matched = df.loc[mask]
            if matched.empty:
                continue
            for row in matched.itertuples(index=False):
                row_dict = row._asdict()
                item = _row_to_article(pd.Series(row_dict), symbol_aliases, symbol)
                if item is not None:
                    rows.append(item)

        if not rows:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        return (
            pd.DataFrame(rows)[NORMALIZED_COLUMNS]
            .drop_duplicates(subset=["published_at", "url", "symbol"])
            .reset_index(drop=True)
        )

    def fetch_day(self, symbol: str, day: date) -> pd.DataFrame:
        return self.fetch_day_multi([symbol], day)

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
