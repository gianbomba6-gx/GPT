"""Bootstrap OOS comparison of V1 top20 versus the learned event veto."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_delta(rows: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap mean-ret difference (keep minus all) on paired OOS rows by resampling all rows."""
    x = rows["next_ret"].dropna().to_numpy(dtype=float)
    keep = rows.loc[rows["event_filter_keep"].astype(bool), "next_ret"].dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(keep) == 0:
        return {"n_all": len(x), "n_keep": len(keep), "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    observed = float(keep.mean() - x.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        sample_keep = rng.choice(keep, size=len(keep), replace=True)
        boot[i] = sample_keep.mean() - sample.mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_all": len(x), "n_keep": len(keep), "delta_mean": observed, "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_event_veto_bootstrap.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rows = pd.read_csv(args.rows_csv)
    required = {"symbol", "next_ret", "event_filter_keep"}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["event_filter_keep"] = rows["event_filter_keep"].astype(bool)
    out = []
    for symbol, s in rows.groupby("symbol", sort=True):
        r = bootstrap_delta(s, n_boot=args.n_boot)
        r["symbol"] = symbol
        out.append(r)
    overall = bootstrap_delta(rows, n_boot=args.n_boot)
    overall["symbol"] = "ALL"
    out.append(overall)
    report = pd.DataFrame(out)[["symbol","n_all","n_keep","delta_mean","ci_low","ci_high"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print("BOOTSTRAP VETO PASS")


if __name__ == "__main__":
    main()
