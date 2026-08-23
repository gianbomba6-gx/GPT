import pandas as pd
from src.news_features_v2 import build_news_features, merge_with_market


def test_news_features_are_cutoff_safe():
    raw = pd.DataFrame([
        {"published_at":"2026-08-20T10:00:00Z","symbol":"NVDA","headline":"Company misses earnings and cuts guidance"},
        {"published_at":"2026-08-21T10:00:00Z","symbol":"NVDA","headline":"Strong product launch"},
    ])
    out = build_news_features(raw)
    assert list(out["Date"].astype(str)) == ["2026-08-20", "2026-08-21"]
    assert out.loc[out["Date"].astype(str) == "2026-08-20", "negative_news_share"].iloc[0] == 1.0


def test_market_merge_does_not_create_news_rows():
    market = pd.DataFrame([
        {"Date":"2026-08-20","symbol":"NVDA","ret":-0.04},
        {"Date":"2026-08-21","symbol":"NVDA","ret":0.01},
    ])
    news = pd.DataFrame([
        {"Date":"2026-08-20","symbol":"NVDA","news_count":2,"negative_news_share":1,"material_event_share":1,"event_polarity":-1,"event_intensity":1,"unique_event_types":2},
    ])
    out = merge_with_market(market, news)
    assert len(out) == len(market)
    assert out.loc[out["Date"] == "2026-08-21", "news_count"].iloc[0] == 0
