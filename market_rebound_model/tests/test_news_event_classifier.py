import pandas as pd
from src.news_event_classifier import EVENT_TYPES, add_event_features, build_daily_event_features


def test_negative_earnings_event():
    df = pd.DataFrame([{
        "published_at": "2026-08-20T12:00:00Z",
        "symbol": "NVDA",
        "headline": "Company misses earnings and cuts guidance",
    }])
    out = add_event_features(df)
    assert out.loc[0, "event_type"] in {"earnings", "guidance"}
    assert out.loc[0, "is_negative_event"] == 1


def test_gkg_summary_fallback_when_headline_is_empty():
    df = pd.DataFrame([{
        "published_at": "2026-08-20T12:00:00Z",
        "symbol": "NVDA",
        "headline": "",
        "summary": "earnings guidance downgrade",
    }])
    out = add_event_features(df)
    assert out.loc[0, "classification_text"] == "earnings guidance downgrade  "
    assert out.loc[0, "event_type"] in {"earnings", "guidance", "analyst"}
    assert out.loc[0, "is_negative_event"] == 1


def test_gkg_url_slug_fallback_when_headline_is_empty():
    df = pd.DataFrame([{
        "published_at": "2026-08-20T12:00:00Z",
        "symbol": "NVDA",
        "headline": "",
        "summary": "ECON_STOCKMARKET",
        "url": "https://example.com/nvidia-earnings-miss-cuts-guidance",
    }])
    out = add_event_features(df)
    assert out.loc[0, "event_type"] == "earnings"
    assert out.loc[0, "is_negative_event"] == 1


def test_daily_aggregation_is_date_safe():
    df = pd.DataFrame([
        {"published_at": "2026-08-20T12:00:00Z", "symbol": "TSLA", "headline": "Strong growth"},
        {"published_at": "2026-08-20T18:00:00Z", "symbol": "TSLA", "headline": "Analyst downgrade"},
        {"published_at": "2026-08-21T12:00:00Z", "symbol": "TSLA", "headline": "Product launch"},
    ])
    out = build_daily_event_features(df)
    assert len(out) == 2
    assert out.loc[out["Date"] == "2026-08-20", "news_count"].iloc[0] == 2
    assert out.loc[out["Date"] == "2026-08-21", "news_count"].iloc[0] == 1


def test_daily_event_specific_shares_sum_to_one():
    df = pd.DataFrame([
        {"published_at": "2026-08-20T12:00:00Z", "symbol": "TSLA", "headline": "Analyst downgrade"},
        {"published_at": "2026-08-20T13:00:00Z", "symbol": "TSLA", "headline": "Product recall"},
        {"published_at": "2026-08-20T14:00:00Z", "symbol": "TSLA", "headline": "Fed rates rise"},
        {"published_at": "2026-08-20T15:00:00Z", "symbol": "TSLA", "headline": "General market commentary"},
    ])
    out = build_daily_event_features(df)
    cols = [f"event_{event}_share" for event in EVENT_TYPES] + ["event_other_share"]
    assert set(cols).issubset(out.columns)
    assert abs(float(out.loc[0, cols].sum()) - 1.0) < 1e-9
    assert out.loc[0, "event_analyst_share"] == 0.25
    assert out.loc[0, "event_product_share"] == 0.25
    assert out.loc[0, "event_macro_share"] == 0.25
    assert out.loc[0, "event_other_share"] == 0.25
