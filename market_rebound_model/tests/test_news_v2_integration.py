from datetime import date
import pandas as pd

from src.gkg_candidate_backfill import close_utc, filter_at_close, aggregate_daily
from src.live_alert import add_market_regime
from src.news_v2_backtest import merge_news


def test_market_close_is_timezone_aware():
    milan = close_utc(date(2026, 8, 19), "STLAM.MI")
    us = close_utc(date(2026, 8, 19), "NVDA")
    assert str(milan.tz) == "UTC"
    assert str(us.tz) == "UTC"
    assert milan.hour == 15
    assert us.hour == 20


def test_filter_at_close_excludes_post_close_news_for_diagnostics():
    df = pd.DataFrame([
        {"published_at": "2026-08-19T19:00:00Z", "symbol": "NVDA"},
        {"published_at": "2026-08-19T20:30:00Z", "symbol": "NVDA"},
    ])
    out = filter_at_close(df, "NVDA", date(2026, 8, 19))
    assert len(out) == 1
    assert out.iloc[0]["published_at"] == pd.Timestamp("2026-08-19T19:00:00Z")


def test_aggregate_daily_keeps_post_close_news_on_candidate_day():
    raw = pd.DataFrame([
        {
            "published_at": "2026-08-19T21:00:00Z", "candidate_day": "2026-08-19",
            "symbol": "NVDA", "headline": "", "summary": "after close guidance cut",
            "sentiment": -4.0, "intensity": 4.0, "relevance": 1.0, "novelty": 0.0,
        }
    ])
    out = aggregate_daily(raw)
    assert out.iloc[0]["Date"] == pd.Timestamp("2026-08-19")
    assert out.iloc[0]["news_count"] == 1


def test_aggregate_daily_uses_candidate_day_and_event_features():
    raw = pd.DataFrame([
        {
            "published_at": "2026-08-20T12:00:00Z", "candidate_day": "2026-08-19",
            "symbol": "NVDA", "headline": "", "summary": "earnings downgrade",
            "sentiment": -3.0, "intensity": 3.0, "relevance": 1.0, "novelty": 0.0,
        }
    ])
    out = aggregate_daily(raw)
    assert out.iloc[0]["Date"] == pd.Timestamp("2026-08-19")
    assert out.iloc[0]["news_count"] == 1
    assert out.iloc[0]["negative_news_share"] == 1
    assert out.iloc[0]["news_available"] == 1


def _base_market_dates(values):
    market = pd.DataFrame({
        "Date": values,
        "Ultimo": [100.0, 98.0], "Apertura": [100.0, 100.0],
        "Massimo": [101.0, 99.0], "Minimo": [99.0, 97.0], "Vol.": [1000, 1100],
    })
    from src.rebound_model import engineer_features
    return engineer_features(market)


def _news_row(day):
    return {
        "Date": day, "symbol": "NVDA",
        "news_sentiment": -2.0, "news_intensity": 2.0, "news_relevance": 1.0,
        "news_novelty": 0.0, "news_count": 4, "negative_news_share": 1.0,
        "material_event_share": 1.0, "event_polarity": -1.0,
        "event_intensity": 1.0, "unique_event_types": 2, "news_available": 1.0,
    }


def test_merge_news_fills_missing_days_without_lookahead():
    market = _base_market_dates(pd.to_datetime(["2026-08-18", "2026-08-19"]))
    news = pd.DataFrame([_news_row(pd.Timestamp("2026-08-19"))])
    out = merge_news(market, news, "NVDA")
    assert out.loc[out["Date"] == pd.Timestamp("2026-08-18"), "news_count"].iloc[0] == 0
    assert out.loc[out["Date"] == pd.Timestamp("2026-08-19"), "news_count"].iloc[0] == 4


def test_merge_news_maps_weekend_news_to_next_market_session():
    market = _base_market_dates(pd.to_datetime(["2026-08-21", "2026-08-24"]))
    news = pd.DataFrame([_news_row(pd.Timestamp("2026-08-22"))])
    news.loc[:, "news_count"] = 2
    news.loc[:, "news_sentiment"] = -3.0
    news.loc[:, "news_intensity"] = 3.0
    out = merge_news(market, news, "NVDA")
    assert out.loc[out["Date"] == pd.Timestamp("2026-08-21"), "news_count"].iloc[0] == 0
    assert out.loc[out["Date"] == pd.Timestamp("2026-08-24"), "news_count"].iloc[0] == 2


def test_merge_news_normalizes_pandas_datetime_resolution_before_asof_merge():
    market_dates = pd.Series(pd.to_datetime(["2026-08-18", "2026-08-19"]).astype("datetime64[ns]"))
    news_dates = pd.Series(pd.to_datetime(["2026-08-19"]).astype("datetime64[s]"))
    market = _base_market_dates(market_dates)
    news = pd.DataFrame([_news_row(news_dates.iloc[0])])
    out = merge_news(market, news, "NVDA")
    assert out.loc[out["Date"] == pd.Timestamp("2026-08-19"), "news_count"].iloc[0] == 4


def _regime_frame(date_values):
    return pd.DataFrame({
        "Date": pd.to_datetime(date_values),
        "Ultimo": [100.0, 98.0], "ret": [0.0, -0.02],
        "ret_5": [0.0, -0.01], "ret_20": [0.0, -0.02],
    })


def test_add_market_regime_normalizes_datetime_resolution_before_asof_merge():
    frames = {"NVDA": _regime_frame(["2026-08-18", "2026-08-19"])}
    benchmark = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-18", "2026-08-19"]).astype("datetime64[s]"),
        "Ultimo": [100.0, 101.0],
    })
    out = add_market_regime(frames, benchmark)
    assert list(out["NVDA"]["Date"]) == list(pd.to_datetime(["2026-08-18", "2026-08-19"]))
    assert out["NVDA"]["mkt_ret"].notna().any()
