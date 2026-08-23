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
    def __init__(self, content): self.content = content
    def raise_for_status(self): pass


class FakeSession:
    def __init__(self, content): self.content, self.calls = content, 0
    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.content)


def _canonical_row(org):
    row = [""] * len(GKG_COLUMNS)
    row[0] = "20260820153000-1"
    row[1] = "20260820153000"
    row[3] = "example.com"
    row[4] = "https://example.com/article"
    row[8] = "ECON_STOCKMARKET;ECON_TECH"
    row[13] = org
    row[14] = f"{org},123"
    row[15] = "-2.5,1.0,3.0,4.0,5.0,6.0,100"
    row[23] = f"{org},123"
    return row


def _gkg1_row(org="NVIDIA"):
    return [
        "20260820153000",
        "3",
        "",
        "ECON_STOCKMARKET;ECON_TECH",
        "",
        "",
        f"{org};Microsoft",
        "-3.2,1.0,4.0,5.0,6.0",
        "",
        "example.com",
        "https://example.com/nvidia-news",
    ]


def test_gkg_filters_organization_and_parses_tone():
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([_canonical_row("NVIDIA"), _canonical_row("Microsoft")]))
    out = provider.fetch_day("NVDA", date(2026, 8, 20))
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "NVDA"
    assert out.iloc[0]["source"] == "example.com"
    assert out.iloc[0]["sentiment"] == -2.5
    assert out.iloc[0]["intensity"] == 2.5
    assert provider.session.calls == 1


def test_gkg1_layout_is_supported():
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([_gkg1_row()]))
    out = provider.fetch_day("NVDA", date(2026, 8, 20))
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "NVDA"
    assert out.iloc[0]["url"] == "https://example.com/nvidia-news"
    assert out.iloc[0]["source"] == "example.com"
    assert out.iloc[0]["sentiment"] == -3.2


def test_gkg_rejects_unknown_row_width():
    row = [""] * 12
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([row]))
    try:
        provider.fetch_day("NVDA", date(2026, 8, 20))
    except RuntimeError as exc:
        message = str(exc)
        assert "expected 27" in message and "11" in message
    else:
        raise AssertionError("Expected unsupported GKG layout to be rejected")


def test_gkg_day_is_cached_across_symbols():
    provider = GkgHistoricalProvider()
    provider.session = FakeSession(_zip_payload([_canonical_row("NVIDIA"), _canonical_row("Tesla")]))
    provider.fetch_day("NVDA", date(2026, 8, 20))
    provider.fetch_day("TSLA", date(2026, 8, 20))
    assert provider.session.calls == 1
