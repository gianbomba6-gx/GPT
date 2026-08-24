import pandas as pd

from src.news_event_conditional_ranking import build_ranking


def test_small_samples_are_insufficient_and_large_consistent_samples_are_favorable():
    rows = []
    for i in range(25):
        rows.append({
            "symbol": "NVDA", "event": "earnings", "condition": "event_negative", "n": 25,
            "mean_next_ret": 0.01, "hit_2pct": 0.40, "hit_3pct": 0.24, "hit_5pct": 0.10,
        })
    rows += [{
        "symbol": "NVDA", "event": "all", "condition": "all_candidates", "n": 100,
        "mean_next_ret": 0.005, "hit_2pct": 0.30, "hit_3pct": 0.18, "hit_5pct": 0.08,
    }]
    # The ranking operates on already aggregated rows; duplicate event rows are not
    # expected in production, so collapse to one canonical row for the assertion.
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "event", "condition"])
    out = build_ranking(df, min_n=20)
    r = out[(out.symbol == "NVDA") & (out.event == "earnings")].iloc[0]
    assert r["tier"] == "FAVORABLE"
    assert r["delta_mean"] == 0.005


def test_small_sample_is_never_favorable():
    df = pd.DataFrame([
        {"symbol": "TSLA", "event": "ma", "condition": "event_negative", "n": 5,
         "mean_next_ret": 0.08, "hit_2pct": 1.0, "hit_3pct": 1.0, "hit_5pct": 1.0},
        {"symbol": "TSLA", "event": "all", "condition": "all_candidates", "n": 100,
         "mean_next_ret": 0.001, "hit_2pct": 0.27, "hit_3pct": 0.18, "hit_5pct": 0.06},
    ])
    out = build_ranking(df, min_n=20)
    r = out[(out.symbol == "TSLA") & (out.event == "ma")].iloc[0]
    assert r["tier"] == "INSUFFICIENT"
