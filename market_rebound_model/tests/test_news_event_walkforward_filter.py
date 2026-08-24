import pandas as pd

from src.news_event_walkforward_filter import walkforward_filter


def _raw_row(day: str, published: str, event_type: str, negative: int = 1):
    return {
        "candidate_day": day,
        "published_at": published,
        "symbol": "TSLA",
        "headline": f"{event_type} news",
        "url": "https://example.com/story",
        "summary": event_type,
        "event_type": event_type,
        "is_negative_event": negative,
    }


def test_filter_never_uses_same_day_to_learn_rule():
    oos = pd.DataFrame([
        {"Date": "2026-01-01", "symbol": "TSLA", "next_ret": -0.01, "v1_top20": True},
        {"Date": "2026-01-02", "symbol": "TSLA", "next_ret": 0.03, "v1_top20": True},
    ])
    raw = pd.DataFrame([
        _raw_row("2026-01-01", "2026-01-01T18:00:00Z", "earnings"),
        _raw_row("2026-01-02", "2026-01-02T18:00:00Z", "earnings"),
    ])
    out = walkforward_filter(oos, raw, min_n=1)
    assert len(out) == 2
    assert out.iloc[0]["event_filter_status"] == "NEUTRAL"


def test_small_samples_are_never_used_as_veto_or_favorable_rule():
    oos = pd.DataFrame([
        {"Date": "2026-01-01", "symbol": "TSLA", "next_ret": 0.10, "v1_top20": True},
        {"Date": "2026-01-02", "symbol": "TSLA", "next_ret": 0.10, "v1_top20": True},
        {"Date": "2026-01-03", "symbol": "TSLA", "next_ret": 0.10, "v1_top20": True},
    ])
    raw = pd.DataFrame([
        _raw_row("2026-01-01", "2026-01-01T18:00:00Z", "guidance"),
        _raw_row("2026-01-02", "2026-01-02T18:00:00Z", "guidance"),
        _raw_row("2026-01-03", "2026-01-03T18:00:00Z", "guidance"),
    ])
    out = walkforward_filter(oos, raw, min_n=20)
    assert (out["event_filter_status"] == "NEUTRAL").all()
    assert not out["event_veto"].any()


def test_negative_event_can_trigger_prior_learned_veto():
    rows = []
    raw_rows = []
    for i in range(25):
        day = f"2026-01-{i+1:02d}"
        ret = -0.05
        rows.append({"Date": day, "symbol": "TSLA", "next_ret": ret, "v1_top20": True})
        raw_rows.append(_raw_row(day, f"2026-01-{i+1:02d}T18:00:00Z", "product"))
    rows.append({"Date": "2026-02-01", "symbol": "TSLA", "next_ret": 0.02, "v1_top20": True})
    raw_rows.append(_raw_row("2026-02-01", "2026-02-01T18:00:00Z", "product"))
    out = walkforward_filter(pd.DataFrame(rows), pd.DataFrame(raw_rows), min_n=20)
    last = out.iloc[-1]
    assert last["event_filter_status"] == "AVOID"
    assert bool(last["event_veto"])
