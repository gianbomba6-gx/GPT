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
        _row("2026-01-01", -0.05),
        _row("2026-01-02", -0.05),
        _row("2026-02-01", 0.02),
    ])
    out = score_oos(rows, min_n=2, shrink_k=0)
    assert len(out) == 3
    assert out.iloc[0]["news_rank_score"] == 0.0
    assert out.iloc[-1]["news_rank_score"] < 0


def test_non_candidate_rows_do_not_change_score():
    rows = pd.DataFrame([
        _row("2026-01-01", -0.05),
        {**_row("2026-01-02", 0.50), "v1_top20": False},
        _row("2026-01-03", -0.05),
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
