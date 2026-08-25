from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _rank_subset(s: pd.DataFrame, frac: float) -> pd.DataFrame:
    s = s.sort_values(["Date", "news_rank_score"], ascending=[True, False]).copy()
    # Select the top fraction within each day. A day with one candidate
    # selects that candidate; otherwise at least one candidate is selected.
    keep = np.zeros(len(s), dtype=bool)
    for _, idx in s.groupby("Date", sort=False).groups.items():
        positions = list(idx)
        n = len(positions)
        k = max(1, int(np.ceil(n * frac)))
        top_idx = positions[:k]
        keep[s.index.get_indexer(top_idx)] = True
    out = s.copy()
    out["selected"] = keep
    return out


def bootstrap_selection(rows: pd.DataFrame, frac: float, n_boot: int = 10000, seed: int = 42) -> dict:
    x = rows.loc[rows["next_ret"].notna()].copy().reset_index(drop=True)
    if x.empty:
        return {"n_all": 0, "n_selected": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x = x.dropna(subset=["next_ret"]).reset_index(drop=True)
    selected = x["selected"].astype(bool).to_numpy()
    n_all = len(x)
    n_selected = int(selected.sum())
    if n_selected == 0:
        return {"n_all": n_all, "n_selected": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    values = x["next_ret"].to_numpy(float)
    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_all, size=n_all)
        sv = values[idx]
        ss = selected[idx]
        if not ss.any():
            boot[i] = np.nan
        else:
            boot[i] = float(sv[ss].mean() - sv.mean())
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {"n_all": n_all, "n_selected": n_selected, "delta_mean": observed, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_all": n_all, "n_selected": n_selected, "delta_mean": observed, "ci_low": float(lo), "ci_high": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_nvda_selection_bootstrap.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    rows = pd.read_csv(args.rows_csv)
    rows = rows[rows["symbol"].astype(str).str.upper().eq("NVDA")].copy()
    rows = rows[rows["news_rank_known_events"] > 0].copy()
    rows["Date"] = pd.to_datetime(rows["Date"])
    rows["news_rank_score"] = pd.to_numeric(rows["news_rank_score"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows = rows.dropna(subset=["Date", "news_rank_score", "next_ret"])
    baseline = rows.copy()
    baseline["selected"] = True
    out = []
    for frac, label in [(0.25, "top25"), (0.50, "top50")]:
        selected = _rank_subset(rows, frac)
        r = bootstrap_selection(selected, frac, n_boot=args.n_boot)
        r["selection"] = label
        out.append(r)
    base = {"selection": "baseline", "n_all": len(baseline), "n_selected": len(baseline), "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    out.insert(0, base)
    report = pd.DataFrame(out)[["selection", "n_all", "n_selected", "delta_mean", "ci_low", "ci_high"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print("NVDA NEWS SELECTION BOOTSTRAP PASS")


if __name__ == "__main__":
    main()
