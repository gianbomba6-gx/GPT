"""Robust ranking of event-conditioned rebound contexts from OOS diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_CONDITION = "all_candidates"
EVENT_CONDITIONS = ("event_negative", "event_dominant_negative")
MIN_N = 20
BOOTSTRAPS = 2000


def _mean_ci(x: pd.Series, rng: np.random.Generator, bootstraps: int = BOOTSTRAPS) -> tuple[float, float, float]:
    y = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    mean = float(y.mean())
    if len(y) < 2:
        return mean, np.nan, np.nan
    idx = rng.integers(0, len(y), size=(bootstraps, len(y)))
    means = y[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def build_ranking(diag: pd.DataFrame, min_n: int = MIN_N, seed: int = 42) -> pd.DataFrame:
    required = {"symbol", "event", "condition", "n", "mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"}
    missing = required - set(diag.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")

    d = diag.copy()
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    for c in ["n", "mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    baseline = (
        d[d["condition"] == BASELINE_CONDITION]
        .set_index("symbol")[["mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"]]
        .rename(columns=lambda c: f"baseline_{c}")
    )

    candidates = d[d["condition"].isin(EVENT_CONDITIONS) & (d["event"] != "all")].copy()
    if candidates.empty:
        raise ValueError("No event-conditional rows available")
    candidates = candidates.merge(baseline, left_on="symbol", right_index=True, how="left")
    candidates["delta_mean"] = candidates["mean_next_ret"] - candidates["baseline_mean_next_ret"]
    candidates["delta_hit_2pct"] = candidates["hit_2pct"] - candidates["baseline_hit_2pct"]
    candidates["delta_hit_3pct"] = candidates["hit_3pct"] - candidates["baseline_hit_3pct"]
    candidates["delta_hit_5pct"] = candidates["hit_5pct"] - candidates["baseline_hit_5pct"]

    rng = np.random.default_rng(seed)
    rows = []
    for (symbol, event, condition), group in candidates.groupby(["symbol", "event", "condition"], sort=True):
        n = int(group["n"].iloc[0])
        mean = float(group["mean_next_ret"].iloc[0])
        # Diagnostics are aggregated statistics; bootstrap the stored mean is invalid.
        # Use the CI only for ranking uncertainty when raw returns are unavailable.
        ci_half = float(1.96 * np.sqrt(max(mean * mean, 1e-8) / max(n, 1))) if n else np.nan
        lo = mean - ci_half if n else np.nan
        hi = mean + ci_half if n else np.nan
        delta = float(group["delta_mean"].iloc[0])
        d2 = float(group["delta_hit_2pct"].iloc[0])
        d3 = float(group["delta_hit_3pct"].iloc[0])
        d5 = float(group["delta_hit_5pct"].iloc[0])

        if n < min_n:
            tier = "INSUFFICIENT"
        elif lo > 0 and d2 >= 0 and d3 >= 0:
            tier = "FAVORABLE"
        elif hi < 0 and d2 <= 0 and d3 <= 0:
            tier = "AVOID"
        else:
            tier = "WATCH"

        # Confidence score intentionally rewards sample size and consistency,
        # but never turns a small sample into a positive recommendation.
        consistency = (np.sign(delta) == np.sign(d2) == np.sign(d3))
        score = (np.log1p(n) * max(delta, 0.0) * (1.0 if consistency else 0.5))
        rows.append({
            "symbol": symbol,
            "event": event,
            "condition": condition,
            "n": n,
            "mean_next_ret": mean,
            "baseline_mean_next_ret": float(group["baseline_mean_next_ret"].iloc[0]),
            "delta_mean": delta,
            "hit_2pct": float(group["hit_2pct"].iloc[0]),
            "baseline_hit_2pct": float(group["baseline_hit_2pct"].iloc[0]),
            "delta_hit_2pct": d2,
            "hit_3pct": float(group["hit_3pct"].iloc[0]),
            "baseline_hit_3pct": float(group["baseline_hit_3pct"].iloc[0]),
            "delta_hit_3pct": d3,
            "hit_5pct": float(group["hit_5pct"].iloc[0]),
            "baseline_hit_5pct": float(group["baseline_hit_5pct"].iloc[0]),
            "delta_hit_5pct": d5,
            "approx_ci_low": lo,
            "approx_ci_high": hi,
            "tier": tier,
            "score": float(score),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["symbol", "tier", "score", "n"], ascending=[True, True, False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostics")
    ap.add_argument("--out", default="results/news_v3_event_conditional_ranking.csv")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    d = pd.read_csv(args.diagnostics)
    out = build_ranking(d, min_n=args.min_n)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(out.to_string(index=False))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
