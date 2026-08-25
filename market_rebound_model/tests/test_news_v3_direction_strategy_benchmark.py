import numpy as np
import pandas as pd

from src.news_v3_direction_strategy_benchmark import _select_calibrated, _select_with_history


def test_fixed_direction_uses_prior_history_only():
    rows = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5),
        "news_rank_score": [0.1, 0.2, 0.3, 0.9, 0.8],
        "next_ret": [0.0, 0.0, 0.0, 0.1, 0.1],
    })
    out = _select_with_history(rows, frac=0.5, direction="normal", min_history=2)
    assert out.loc[0:1, "eligible"].sum() == 0
    assert out.loc[2:, "eligible"].all()


def test_calibrated_direction_has_expected_columns_and_valid_selection():
    rows = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=8),
        "news_rank_score": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5],
        "next_ret": [0.0, 0.0, 0.1, 0.1, -0.01, 0.02, 0.03, 0.01],
    })
    out = _select_calibrated(rows, frac=0.5, min_history=2, min_calibration=2)
    assert {"eligible", "selected", "direction"}.issubset(out.columns)
    assert out["selected"].sum() <= out["eligible"].sum()
    assert out["direction"].isin(["normal", "inverted"]).all()
