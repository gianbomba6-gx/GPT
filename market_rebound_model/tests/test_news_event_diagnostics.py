import pandas as pd

from src.news_event_diagnostics import build_diagnostics


def test_event_diagnostics_reports_present_and_dominant_events():
    df = pd.DataFrame([
        {
            "Date": "2026-01-02", "symbol": "NVDA", "next_ret": 0.03, "next_high": 0.04,
            "event_earnings_share": 1.0, "event_guidance_share": 0.0,
        },
        {
            "Date": "2026-01-03", "symbol": "NVDA", "next_ret": -0.01, "next_high": 0.01,
            "event_earnings_share": 0.0, "event_guidance_share": 1.0,
        },
    ])
    out = build_diagnostics(df)

    earnings = out[(out.symbol == "NVDA") & (out.event == "earnings") & (out.condition == "event_present")].iloc[0]
    guidance = out[(out.symbol == "NVDA") & (out.event == "guidance") & (out.condition == "event_present")].iloc[0]
    assert earnings.n == 1
    assert earnings.mean_next_ret == 0.03
    assert guidance.n == 1
    assert guidance.mean_next_ret == -0.01


def test_event_diagnostics_uses_only_oos_columns():
    df = pd.DataFrame([
        {
            "Date": "2026-01-02", "symbol": "TSLA", "next_ret": 0.02, "next_high": 0.03,
            "event_macro_share": 1.0,
        },
    ])
    out = build_diagnostics(df)
    assert set(out["symbol"]) == {"TSLA"}
    assert out[(out.event == "macro") & (out.condition == "dominant")].iloc[0].n == 1
