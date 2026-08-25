"""Bootstrap OOS significance checks for inverted secondary news ranking."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


QUARTILES = ("Q1_low", "Q2", "Q3", "Q4_high")


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "next_ret", "news_rank_score"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    x = rows[["symbol", "next_ret", "news_rank_score"]].copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["news_rank_score"] = pd.to_numeric(x["news_rank_score"], errors="coerce")
    x = x.dropna(subset=["next_ret", "news_rank_score"]).reset_index(drop=True)
    if len(x) < 4:
        raise ValueError("Not enough rows for quartile bootstrap")
    x["score_q"] = pd.qcut(
        x["news_rank_score"].rank(method="first"),
        q=4,
        labels=list(QUARTILES),
    )
    return x


def bootstrap_q4_minus_q1(rows: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    x = _prepare(rows)
    q1 = x.loc[x["score_q"] == "Q1_low", "next_ret"].to_numpy(dtype=float)
    q4 = x.loc[x["score_q"] == "Q4_high", "next_ret"].to_numpy(dtype=float)
    if len(q1) == 0 or len(q4) == 0:
        return {"n_q1": len(q1), "n_q4": len(q4), "delta_q4_q1": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    observed = float(q4.mean() - q1.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        a = rng.choice(q1, size=len(q1), replace=True)
        b = rng.choice(q4, size=len(q4), replace=True)
        boot[i] = float(b.mean() - a.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_q1": len(q1),
        "n_q4": len(q4),
        "delta_q4_q1": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def bootstrap_spearman(rows: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    x = _prepare(rows)
    if len(x) < 3 or x["news_rank_score"].nunique() < 2 or x["next_ret"].nunique() < 2:
        return {"n": len(x), "spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    def corr(d: pd.DataFrame) -> float:
        return float(d["news_rank_score"].rank().corr(d["next_ret"].rank()))

    observed = corr(x)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = corr(x.iloc[idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": n, "spearman": observed, "ci_low": float(lo), "ci_high": float(hi)}


def _quartile_table(rows: pd.DataFrame) -> pd.DataFrame:
    x = _prepare(rows)
    return (
        x.groupby("score_q", observed=False)["next_ret"]
        .agg(n="count", mean_next_ret="mean", median_next_ret="median")
        .reindex(QUARTILES)
        .reset_index()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_secondary_ranking_bootstrap.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    overall = bootstrap_q4_minus_q1(rows, n_boot=args.n_boot, seed=42)
    spear = bootstrap_spearman(rows, n_boot=args.n_boot, seed=42)

    per_symbol = []
    for symbol, s in rows.groupby("symbol", sort=True):
        try:
            r = bootstrap_q4_minus_q1(s, n_boot=args.n_boot, seed=42)
            sp = bootstrap_spearman(s, n_boot=args.n_boot, seed=42)
        except ValueError:
            r = {"n_q1": 0, "n_q4": 0, "delta_q4_q1": np.nan, "ci_low": np.nan, "ci_high": np.nan}
            sp = {"n": len(s), "spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        per_symbol.append({"symbol": symbol, **r, "spearman": sp["spearman"], "spearman_ci_low": sp["ci_low"], "spearman_ci_high": sp["ci_high"]})

    report = pd.DataFrame(per_symbol)
    overall_row = pd.DataFrame([{"symbol": "ALL", **overall, "spearman": spear["spearman"], "spearman_ci_low": spear["ci_low"], "spearman_ci_high": spear["ci_high"]}])
    report = pd.concat([report, overall_row], ignore_index=True)
    quartiles = _quartile_table(rows)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(p, index=False)
    quartiles.to_csv(p.with_name(p.stem + "_quartiles.csv"), index=False)
    print(report.to_string(index=False))
    print("SCORE QUARTILES")
    print(quartiles.to_string(index=False))
    print(f"Saved {p}")
    print("SECONDARY NEWS BOOTSTRAP PASS")


if __name__ == "__main__":
    main()
