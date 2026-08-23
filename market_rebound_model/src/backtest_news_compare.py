"""Leakage-safe comparison of V1-like market score and V2 news-enhanced score."""
from __future__ import annotations
import pandas as pd

NEWS_FEATURES = ["negative_news_share", "material_event_share", "event_polarity", "event_intensity"]


def prepare_comparison(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in NEWS_FEATURES:
        if c not in x:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    # News features are only used from the same session's close onward; caller
    # must supply features constructed with a publication cutoff <= market close.
    x["news_pressure"] = (
        0.45 * x["negative_news_share"]
        + 0.30 * x["material_event_share"]
        - 0.15 * x["event_polarity"]
        + 0.10 * x["event_intensity"]
    ).clip(-1, 1)
    return x


def add_next_day_target(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    x = df.sort_values(["symbol", "Date"]).copy()
    x["next_ret"] = x.groupby("symbol")[price_col].shift(-1) / x[price_col] - 1.0
    return x


def summarize_sets(df: pd.DataFrame, score_col: str = "model_p") -> pd.DataFrame:
    x = add_next_day_target(prepare_comparison(df))
    base = x[x.get("ret", 0) <= -0.02].copy()
    if base.empty:
        return pd.DataFrame()
    base["news_adjusted_score"] = base[score_col].fillna(0.0) + 0.20 * base["news_pressure"]
    rows = []
    for name, mask in {
        "baseline -2%": pd.Series(True, index=base.index),
        "V1 top20%": base[score_col] >= base[score_col].quantile(0.80),
        "V2 news top20%": base["news_adjusted_score"] >= base["news_adjusted_score"].quantile(0.80),
    }.items():
        z = base[mask]
        rows.append({
            "set": name, "n": len(z),
            "mean_next_ret": z["next_ret"].mean(),
            "median_next_ret": z["next_ret"].median(),
            "hit_2pct": (z["next_ret"] >= .02).mean(),
            "hit_3pct": (z["next_ret"] >= .03).mean(),
            "hit_5pct": (z["next_ret"] >= .05).mean(),
        })
    return pd.DataFrame(rows)
