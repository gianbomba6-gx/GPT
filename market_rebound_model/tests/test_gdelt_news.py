import pandas as pd

from src.gdelt_news import GdeltNewsProvider, SEARCH_TERMS, _parse_seen_date
from src.news_provider import NewsQuery


class FakeResponse:
    def __init__(self, payload=None, content_type="application/json", text=""):
        self._payload = payload
        self.status_code = 200
        self.headers = {"content-type": content_type}
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        return self.response


def test_gdelt_timestamp_parser_is_utc():
    ts = _parse_seen_date("20260820153000")
    assert str(ts) == "2026-08-20 15:30:00+00:00"


def test_doc_search_terms_are_not_rejected_as_short_phrases():
    assert set(SEARCH_TERMS) >= {"STLAM.MI", "SPCX", "NVDA", "TSLA"}
    for symbol, query in SEARCH_TERMS.items():
        assert len(query.replace('"', '').replace('(', '').replace(')', '').strip()) >= 5, symbol


def test_gdelt_fetch_normalizes_and_fences(monkeypatch):
    class Response:
        headers = {"content-type": "application/json"}
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"articles": [
                {"seendate": "20260820120000", "title": "Tesla beats estimates", "domain": "example.com", "url": "https://example.com/a"},
                {"seendate": "20260821120000", "title": "Future article", "domain": "example.com", "url": "https://example.com/b"},
            ]}

    monkeypatch.setattr("src.gdelt_news.requests.Session.get", lambda self, *a, **k: Response())
    monkeypatch.setattr("src.gdelt_news.time.sleep", lambda *_: None)
    p = GdeltNewsProvider(pause=0, retries=0)
    q = NewsQuery("TSLA", pd.Timestamp("2026-08-20", tz="UTC").to_pydatetime(), pd.Timestamp("2026-08-20T23:59:59Z").to_pydatetime())
    out = p.fetch(q)
    assert len(out) == 1
    assert out.iloc[0]["symbol"] == "TSLA"
    assert out.iloc[0]["headline"] == "Tesla beats estimates"


def test_non_json_response_is_reported_clearly():
    p = GdeltNewsProvider(pause=0, retries=0)
    p.session = FakeSession(FakeResponse(ValueError("not json"), "text/html", "<html>gateway response</html>"))
    start = pd.Timestamp("2026-08-20", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2026-08-21", tz="UTC").to_pydatetime()
    try:
        p._request("NVDA", start, end)
    except RuntimeError as exc:
        msg = str(exc)
        assert "non-JSON" in msg
        assert "text/html" in msg
        assert "gateway response" in msg
    else:
        raise AssertionError("Expected RuntimeError for non-JSON response")


def test_unknown_symbol_is_rejected_before_network():
    p = GdeltNewsProvider(pause=0, retries=0)
    fake = FakeSession(FakeResponse({"articles": []}))
    p.session = fake
    start = pd.Timestamp("2026-08-20", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2026-08-21", tz="UTC").to_pydatetime()
    try:
        p._request("UNKNOWN", start, end)
    except ValueError as exc:
        assert "No GDELT search mapping" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown symbol")
    assert fake.calls == []
