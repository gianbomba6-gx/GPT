import pandas as pd

from src.news_event_conditional_diagnostics import build_conditional_diagnostics


def test_conditional_diagnostics_filters_news_after_market_close():
    oos = pd.DataFrame([
        {"Date": "2026-08-20", "symbol": "TSLA", "next_ret": 0.03, "next_high": 0.04},
    ])
    raw = pd.DataFrame([
        {
            "published_at": "2026-08-20T18:00:00Z",  # 14:00 ET, before close
            "symbol": "TSLA", "candidate_day": "2026-08-20",
            "headline": "Analyst downgrade", "summary": "", "url": "",
        },
        {
            "published_at": "2026-08-20T21:00:00Z",  # 17:00 ET, after close
            "symbol": "TSLA", "candidate_day": "2026-08-20",
            "headline": "Product launch", "summary": "", "url": "",
        },
    ])
    out = build_conditional_diagnostics(oos, raw)
    analyst = out[(out.event == "analyst") & (out.condition == "event_present")].iloc[0]
    product = out[(out.event == "product") & (out.condition == "event_present")].iloc[0]
    assert analyst.n == 1
    assert product.n == 0


def test_negative_event_condition_is_distinct_from_event_presence():
    oos = pd.DataFrame([
        {"Date": "2026-01-02", "symbol": "NVDA", "next_ret": 0.05, "next_high": 0.06},
        {"Date": "2026-01-03", "symbol": "NVDA", "next_ret": -0.01, "next_high": 0.01},
    ])
    raw = pd.DataFrame([
        {
            "published_at": "2026-01-02T16:00:00Z", "symbol": "NVDA", "candidate_day": "2026-01-02",
            "headline": "Analyst upgrade", "summary": "", "url": "",
        },
        {
            "published_at": "2026-01-03T16:00:00Z", "symbol": "NVDA", "candidate_day": "2026-01-03",
            "headline": "Analyst downgrade", "summary": "", "url": "",
        },
    ])
    out = build_conditional_diagnostics(oos, raw)
    present = out[(out.event == "analyst") & (out.condition == "event_present")].iloc[0]
    negative = out[(out.event == "analyst") & (out.condition == "event_negative")].iloc[0]
    assert present.n == 2
    assert negative.n == 1
    assert negative.mean_next_ret == -0.01


def test_dominant_negative_event_is_reported():
    oos = pd.DataFrame([
        {"Date": "2026-01-02", "symbol": "TSLA", "next_ret": 0.04, "next_high": 0.05},
    ])
    raw = pd.DataFrame([
        {
            "published_at": "2026-01-02T15:00:00Z", "symbol": "TSLA", "candidate_day": "2026-01-02",
            "headline": "Earnings miss and cuts guidance", "summary": "", "url": "",
        },
        {
            "published_at": "2026-01-02T16:00:00Z", "symbol": "TSLA", "candidate_day": "2026-01-02",
            "headline": "Market commentary", "summary": "", "url": "",
        },
    ])
    out = build_conditional_diagnostics(oos, raw)
    row = out[(out.event == "earnings") & (out.condition == "event_dominant_negative")].iloc[0]
    assert row.n == 1
    assert row.mean_next_ret == 0.04
