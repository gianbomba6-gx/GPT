from datetime import date
from io import BytesIO
import zipfile

from src.gkg_historical_news import GKG_COLUMNS, GkgHistoricalProvider


def _zip_payload(rows):
    raw = "\n".join("\t".join(row) for row in rows).encode()
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20260820.gkg.csv", raw)
    return buf.getvalue()


class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/zip"}

    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, content):
        self.content = content
        self.headers = {}
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.content)


def _row(org):
    row = [""] * len(GKG_COLUMNS)
    # GKG 2.1 order: GKGRECORDID, then DATE.
    row[0] = "20260820153000-1"
    row[1] = "20260820153000"
    row[2] = "1"
    row[3] = "Example News"
    row[4] = "https://example.com/article"
    row[7] = "ECON"
    row[8] = "ECON_STOCKMARKET"
    row[13] = org
    row[14] = f"{org},123"
    row[15] = "-2.5,1.0,3.0,4.0,5.0,6.0,100"
    return row


def test_gkg_filters_organization_and_parses_tone():
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([_row("NVIDIA"), _row("Microsoft")]))
    out = provider.fetch_day("NVDA", date(2026, 8, 20))
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "NVDA"
    assert out.iloc[0]["source"] == "Example News"
    assert out.iloc[0]["sentiment"] == -2.5
    assert out.iloc[0]["intensity"] == 2.5
    assert provider.session.calls == 1


def test_compact_gkg_layout_is_parsed_without_fixed_27_columns():
    # This layout intentionally contains only the key fields needed by V2.
    row = [
        "20260820153000-9",
        "20260820153000",
        "Example News",
        "https://example.com/nvidia-news",
        "ECON_STOCKMARKET;ECON_TECH",
        "",
        "",
        "NVIDIA,123;Microsoft,456",
        "-3.2,1.0,4.0,5.0,6.0,100",
        "NVIDIA,123",
        "",
    ]
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([row]))
    out = provider.fetch_day("NVDA", date(2026, 8, 20))
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "NVDA"
    assert out.iloc[0]["url"] == "https://example.com/nvidia-news"
    assert out.iloc[0]["source"] == "Example News"
    assert out.iloc[0]["sentiment"] == -3.2


def test_gkg_day_is_cached_across_symbols():
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([_row("NVIDIA"), _row("Tesla")]))
    provider.fetch_day("NVDA", date(2026, 8, 20))
    provider.fetch_day("TSLA", date(2026, 8, 20))
    assert provider.session.calls == 1
