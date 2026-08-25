import numpy as np
import pandas as pd

from src.news_v3_nvda_selection_robustness import bootstrap_delta, prospective_selection


def test_prospective_selection_uses_only_prior_scores():
    rows = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=6, freq="D"),
        "news_rank_score": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        "next_ret": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    })
    out = prospective_selection(rows, frac=0.5, min_history=3)
    assert not out.loc[:2, "eligible"].any()
    assert out.loc[3:, "eligible"].all()
    # The extreme last score is compared only with prior observations.
    assert bool(out.iloc[-1]["selected"])


def test_bootstrap_delta_is_finite_and_deterministic():
    rows = pd.DataFrame({
        "eligible": [True] * 6,
        "selected": [True, True, False, False, False, False],
        "next_ret": [0.06, 0.04, 0.00, 0.01, -0.01, 0.02],
    })
    a = bootstrap_delta(rows, n_boot=500, seed=7)
    b = bootstrap_delta(rows, n_boot=500, seed=7)
    assert a == b
    assert a["n_eligible"] == 6
    assert a["n_selected"] == 2
    assert np.isfinite(a["delta_mean"])
    assert np.isfinite(a["ci_low"])
    assert np.isfinite(a["ci_high"])
    assert a["ci_low"] <= a["ci_high"]
