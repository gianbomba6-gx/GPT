import numpy as np
import pandas as pd

from src.news_v3_nvda_selection_bootstrap import _rank_subset, bootstrap_selection


def test_rank_subset_selects_fraction_of_full_sample():
    rows = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=8, freq="D"),
        "news_rank_score": [0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4],
        "next_ret": [0.08, 0.01, 0.07, 0.02, 0.06, 0.03, 0.05, 0.04],
    })
    top25 = _rank_subset(rows, 0.25)
    top50 = _rank_subset(rows, 0.50)
    assert int(top25["selected"].sum()) == 2
    assert int(top50["selected"].sum()) == 4
    assert top25.loc[top25["selected"], "news_rank_score"].tolist() == [0.8, 0.7]


def test_bootstrap_selection_counts_and_delta():
    rows = pd.DataFrame({
        "next_ret": [0.01, 0.03, -0.01, 0.05],
        "selected": [True, True, False, True],
    })
    out = bootstrap_selection(rows, 0.5, n_boot=1000, seed=7)
    assert out["n_all"] == 4
    assert out["n_selected"] == 3
    assert np.isclose(out["delta_mean"], (0.01 + 0.03 + 0.05) / 3 - 0.02)
    assert np.isfinite(out["ci_low"])
    assert np.isfinite(out["ci_high"])
    assert out["ci_low"] <= out["ci_high"]


def test_bootstrap_selection_is_deterministic():
    rows = pd.DataFrame({
        "next_ret": [0.01, 0.03, -0.01, 0.05],
        "selected": [True, True, False, True],
    })
    assert bootstrap_selection(rows, 0.5, n_boot=500, seed=42) == bootstrap_selection(rows, 0.5, n_boot=500, seed=42)
