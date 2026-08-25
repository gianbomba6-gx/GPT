import numpy as np
import pandas as pd

from src.news_v3_incremental_value import prospective_filter


def test_prospective_filter_uses_only_prior_scores():
    rows = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=5),
        "symbol": ["NVDA"] * 5,
        "news_rank_score": [0.1, 0.2, 0.3, 0.9, 0.8],
        "next_ret": [0.0, 0.0, 0.0, 0.1, 0.1],
        "_row_id": np.arange(5),
    })
    out = prospective_filter(rows, frac=0.5, direction="normal", min_history=2)
    assert out.loc[0:1, "eligible"].sum() == 0
    assert bool(out.loc[2, "eligible"])
    assert bool(out.loc[2, "selected"])
