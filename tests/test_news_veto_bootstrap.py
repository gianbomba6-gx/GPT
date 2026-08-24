import pandas as pd

from src.news_veto_bootstrap import bootstrap_delta


def test_bootstrap_zero_veto_has_zero_delta():
    df = pd.DataFrame({"next_ret": [0.01, 0.02, -0.01], "event_filter_keep": [True, True, True]})
    out = bootstrap_delta(df, n_boot=200, seed=1)
    assert out["n_all"] == 3
    assert out["n_keep"] == 3
    assert out["delta_mean"] == 0.0


def test_bootstrap_reports_positive_delta_when_veto_removes_bad_rows():
    df = pd.DataFrame({"next_ret": [-0.10, -0.05, 0.02, 0.03], "event_filter_keep": [False, False, True, True]})
    out = bootstrap_delta(df, n_boot=500, seed=1)
    assert out["delta_mean"] > 0
    assert out["ci_high"] > 0
