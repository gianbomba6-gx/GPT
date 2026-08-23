import pandas as pd
from src.gdelt_news import GdeltNewsProvider, _parse_seen_date
from src.news_provider import NewsQuery


def test_gdelt_timestamp_parser_is_utc():
    ts = _parse_seen_date("20260820153000")
    assert str(ts) == "2026-08-20 15:30:00+00:00"


def test_gdelt_fetch_normalizes_and_fences(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"articles": [
                {"seendate":"20260820120000","title":"Tesla beats estimates","domain":"example.com","url":"https://example.com/a"},
                {"seendate":"20260821120000","title":"Future article","domain":"example.com","url":"https://example.com/b"},
            ]}
    monkeypatch.setattr("src.gdelt_news.requests.get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr("src.gdelt_news.time.sleep", lambda *_: None)
    p = GdeltNewsProvider()
    q = NewsQuery("TSLA", pd.Timestamp("2026-08-20", tz="UTC").to_pydatetime(), pd.Timestamp("2026-08-20T23:59:59Z").to_pydatetime())
    out = p.fetch(q)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "TSLA"
    assert out.iloc[0]["headline"] == "Tesla beats estimates"
