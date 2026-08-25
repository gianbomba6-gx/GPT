import numpy as np
import pandas as pd

from src.news_v3_multi_symbol_selection_robustness import bootstrap_delta, prospective_selection


def test_prospective_selection_uses_only_prior_history():
    rows = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5),
        "news_rank_score": [0.1, 0.2, 0.3, 0.9, 0.8],
        "next_ret": [0.0, 0.0, 0.0, 0.1, 0.1],
    })
    out = prospective_selection(rows, frac=0.5, min_history=2)
    assert out.loc[0:1, "eligible"].sum() == 0
    assert out.loc[2:, "eligible"].all()
    assert bool(out.loc[2, "selected"]) is True


def test_bootstrap_delta_is_finite_and_ordered():
    rows = pd.DataFrame({
        "eligible": [True, True, True, True],
        "selected": [True, True, False, False],
        "next_ret": [0.05, 0.03, -0.01, 0.01],
    })
    out = bootstrap_delta(rows, n_boot=1000, seed=7)
    assert out["n_eligible"] == 4
    assert out["n_selected"] == 2
    assert np.isfinite(out["delta_mean"])
    assert np.isfinite(out["ci_low"])
    assert np.isfinite(out["ci_high"])
    assert out["ci_low"] <= out["ci_high"]
