import numpy as np
import pandas as pd

from src.news_v3_nvda_selection_bootstrap import bootstrap_selection


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
