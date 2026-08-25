import numpy as np
import pandas as pd

from src.news_v3_secondary_ranking_bootstrap import bootstrap_q4_minus_q1, bootstrap_spearman


def _rows():
    return pd.DataFrame({
        "symbol": ["TSLA"] * 8,
        "news_rank_score": np.arange(8, dtype=float),
        "next_ret": [0.00, 0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
    })


def test_q4_minus_q1_has_expected_sign_and_counts():
    out = bootstrap_q4_minus_q1(_rows(), n_boot=1000, seed=7)
    assert out["n_q1"] == 2
    assert out["n_q4"] == 2
    assert out["delta_q4_q1"] > 0
    assert out["ci_low"] <= out["delta_q4_q1"] <= out["ci_high"]


def test_bootstrap_is_deterministic():
    a = bootstrap_spearman(_rows(), n_boot=1000, seed=42)
    b = bootstrap_spearman(_rows(), n_boot=1000, seed=42)
    assert a == b


def test_quartile_bootstrap_rejects_too_few_rows():
    rows = _rows().iloc[:3].copy()
    try:
        bootstrap_q4_minus_q1(rows, n_boot=100, seed=1)
    except ValueError as exc:
        assert "quartile" in str(exc).lower()
    else:
        raise AssertionError("Expected too-small sample to be rejected")
