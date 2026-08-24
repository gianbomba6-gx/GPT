import numpy as np
import pandas as pd

from src.news_veto_bootstrap import bootstrap_delta


def test_bootstrap_reports_keep_vs_all_delta_and_counts():
    rows = pd.DataFrame(
        {
            "next_ret": [0.01, 0.03, -0.01, 0.05],
            "event_filter_keep": [True, True, False, True],
        }
    )
    out = bootstrap_delta(rows, n_boot=2000, seed=7)
    assert out["n_all"] == 4
    assert out["n_keep"] == 3
    assert np.isclose(out["delta_mean"], (0.01 + 0.03 + 0.05) / 3 - 0.02)
    assert out["ci_low"] <= out["delta_mean"] <= out["ci_high"]


def test_bootstrap_is_deterministic_for_fixed_seed():
    rows = pd.DataFrame(
        {
            "next_ret": [0.01, 0.03, -0.01, 0.05, 0.02],
            "event_filter_keep": [True, True, False, True, False],
        }
    )
    a = bootstrap_delta(rows, n_boot=1000, seed=42)
    b = bootstrap_delta(rows, n_boot=1000, seed=42)
    assert a == b


def test_empty_or_no_kept_rows_returns_nan_ci():
    rows = pd.DataFrame(
        {
            "next_ret": [0.01, 0.02],
            "event_filter_keep": [False, False],
        }
    )
    out = bootstrap_delta(rows, n_boot=100, seed=1)
    assert out["n_all"] == 2
    assert out["n_keep"] == 0
    assert np.isnan(out["delta_mean"])
    assert np.isnan(out["ci_low"])
    assert np.isnan(out["ci_high"])
