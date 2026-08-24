"""Bootstrap OOS comparison of V1 top20 versus the learned event veto."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_delta(rows: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap keep-minus-all mean difference on paired OOS rows.

    Each bootstrap resample draws row indices from the original OOS sample and
    recomputes both means on that same resample. This preserves the fact that
    the kept set is a subset of the full set.
    """
    x = rows.loc[rows["next_ret"].notna(), ["next_ret", "event_filter_keep"]].copy()
    if x.empty:
        return {"n_all": 0, "n_keep": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x = x.dropna(subset=["next_ret"]).reset_index(drop=True)
    x["event_filter_keep"] = x["event_filter_keep"].astype(bool)
    keep_mask = x["event_filter_keep"].to_numpy()
    n_all = len(x)
    n_keep = int(keep_mask.sum())
    if n_keep == 0:
        return {"n_all": n_all, "n_keep": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    values = x["next_ret"].to_numpy(dtype=float)
    observed = float(values[keep_mask].mean() - values.mean())

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_all, size=n_all)
        sample_values = values[idx]
        sample_keep = keep_mask[idx]
        keep_count = int(sample_keep.sum())
        if keep_count == 0:
            boot[i] = np.nan
        else:
            boot[i] = float(sample_values[sample_keep].mean() - sample_values.mean())

    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {"n_all": n_all, "n_keep": n_keep, "delta_mean": observed, "ci_low": np.nan, "ci_high": np.nan}

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_all": n_all,
        "n_keep": n_keep,
        "delta_mean": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


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
    report = pd.DataFrame(out)[["symbol", "n_all", "n_keep", "delta_mean", "ci_low", "ci_high"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print("BOOTSTRAP VETO PASS")


if __name__ == "__main__":
    main()
