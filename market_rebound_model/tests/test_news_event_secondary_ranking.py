import pandas as pd

from src.news_event_secondary_ranking import score_oos


def _row(day, ret, score_share=1.0, symbol="TSLA"):
    return {
        "Date": day,
        "symbol": symbol,
        "next_ret": ret,
        "v1_top20": True,
        "negative_event_product_share": score_share,
        "event_product_share": score_share,
    }


def test_secondary_score_uses_only_prior_v1_candidates():
    rows = pd.DataFrame([
        _row("2026-01-01", -0.08),
        _row("2026-01-02", 0.04, score_share=0.0),
        _row("2026-01-03", -0.08),
        _row("2026-02-01", 0.02),
    ])
    out = score_oos(rows, min_n=2, shrink_k=0)
    assert len(out) == 4
    assert out.iloc[0]["news_rank_score"] == 0.0
    assert out.iloc[-1]["news_rank_score"] < 0


def test_non_candidate_rows_do_not_change_score():
    rows = pd.DataFrame([
        _row("2026-01-01", -0.08),
        _row("2026-01-02", 0.04, score_share=0.0),
        {**_row("2026-01-03", 0.50), "v1_top20": False},
        _row("2026-01-04", -0.08),
        _row("2026-02-01", 0.02),
    ])
    out = score_oos(rows, min_n=2, shrink_k=0)
    last = out.iloc[-1]
    assert last["news_rank_known_events"] == 1
    assert last["news_rank_score"] < 0


def test_daily_rank_is_deterministic():
    rows = pd.DataFrame([
        _row("2026-01-01", -0.05, 1.0, "TSLA"),
        _row("2026-01-02", -0.05, 1.0, "TSLA"),
        _row("2026-02-01", 0.02, 1.0, "TSLA"),
        _row("2026-02-01", 0.03, 0.0, "NVDA"),
    ])
    a = score_oos(rows, min_n=2, shrink_k=0)
    b = score_oos(rows, min_n=2, shrink_k=0)
    pd.testing.assert_frame_equal(a, b)


def test_inverted_score_is_exact_sign_flip():
    rows = pd.DataFrame([
        _row("2026-01-01", -0.08),
        _row("2026-01-02", 0.04, score_share=0.0),
        _row("2026-01-03", -0.08),
        _row("2026-02-01", 0.02),
    ])
    normal = score_oos(rows, min_n=2, shrink_k=0)
    inverted = score_oos(rows, min_n=2, shrink_k=0, invert_score=True)
    pd.testing.assert_series_equal(
        inverted["news_rank_score"],
        -normal["news_rank_score"],
        check_names=False,
    )


def test_raw_gkg_reconstructs_negative_event_share():
    rows = pd.DataFrame([
        {"Date": "2026-01-01", "symbol": "TSLA", "next_ret": -0.08, "v1_top20": True},
        {"Date": "2026-01-02", "symbol": "TSLA", "next_ret": 0.04, "v1_top20": True},
        {"Date": "2026-01-03", "symbol": "TSLA", "next_ret": -0.08, "v1_top20": True},
        {"Date": "2026-02-01", "symbol": "TSLA", "next_ret": 0.02, "v1_top20": True},
    ])
    raw = pd.DataFrame([
        {"candidate_day": "2026-01-01", "published_at": "2026-01-01T15:00:00Z", "symbol": "TSLA", "headline": "product decline news", "url": "https://example.com", "summary": "product decline"},
        {"candidate_day": "2026-01-02", "published_at": "2026-01-02T15:00:00Z", "symbol": "TSLA", "headline": "company update", "url": "https://example.com", "summary": "routine company update"},
        {"candidate_day": "2026-01-03", "published_at": "2026-01-03T15:00:00Z", "symbol": "TSLA", "headline": "product decline news", "url": "https://example.com", "summary": "product decline"},
        {"candidate_day": "2026-02-01", "published_at": "2026-02-01T15:00:00Z", "symbol": "TSLA", "headline": "product decline news", "url": "https://example.com", "summary": "product decline"},
    ])
    out = score_oos(rows, raw=raw, min_n=2, shrink_k=0)
    assert out.iloc[-1]["news_rank_known_events"] == 1
    assert out.iloc[-1]["news_rank_score"] < 0
