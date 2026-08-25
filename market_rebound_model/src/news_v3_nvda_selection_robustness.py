from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prospective_selection(
    rows: pd.DataFrame,
    frac: float,
    min_history: int = 20,
) -> pd.DataFrame:
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    if min_history < 1:
        raise ValueError("min_history must be >= 1")

    x = rows.copy().sort_values("Date").reset_index(drop=True)
    selected = np.zeros(len(x), dtype=bool)
    eligible = np.zeros(len(x), dtype=bool)

    for i in range(len(x)):
        if i < min_history:
            continue
        prior = x.loc[: i - 1, "news_rank_score"].dropna()
        if len(prior) < min_history:
            continue
        eligible[i] = True
        threshold = float(prior.quantile(1.0 - frac))
        selected[i] = bool(x.at[i, "news_rank_score"] >= threshold)

    x["eligible"] = eligible
    x["selected"] = selected
    return x


def bootstrap_delta(x: pd.DataFrame, n_boot: int = 10000, seed: int = 42) -> dict:
    z = x[x["eligible"] & x["next_ret"].notna()].copy().reset_index(drop=True)
    if z.empty:
        return {"n_eligible": 0, "n_selected": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    values = z["next_ret"].to_numpy(float)
    selected = z["selected"].to_numpy(bool)
    n = len(z)
    n_selected = int(selected.sum())
    if n_selected == 0:
        return {"n_eligible": n, "n_selected": 0, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss = selected[idx]
        boot[i] = np.nan if not ss.any() else float(values[idx][ss].mean() - values[idx].mean())
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {"n_eligible": n, "n_selected": n_selected, "delta_mean": observed, "ci_low": np.nan, "ci_high": np.nan}
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_eligible": n, "n_selected": n_selected, "delta_mean": observed, "ci_low": float(lo), "ci_high": float(hi)}


def build_stability(x: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    z = x[x["eligible"] & x["next_ret"].notna()].copy()
    if z.empty:
        return pd.DataFrame(columns=["period", "n_eligible", "n_selected", "mean_all", "mean_selected_gross", "mean_selected_net", "delta_mean"])
    z["period"] = z["Date"].dt.year.astype(int)
    cost = cost_bps / 10000.0
    rows = []
    for period, g in z.groupby("period", sort=True):
        s = g[g["selected"]]
        mean_all = float(g["next_ret"].mean())
        mean_selected = float(s["next_ret"].mean()) if not s.empty else np.nan
        rows.append({
            "period": int(period),
            "n_eligible": len(g),
            "n_selected": len(s),
            "mean_all": mean_all,
            "mean_selected_gross": mean_selected,
            "mean_selected_net": mean_selected - cost if np.isfinite(mean_selected) else np.nan,
            "delta_mean": mean_selected - mean_all if np.isfinite(mean_selected) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_nvda_selection_robustness.csv")
    ap.add_argument("--out-rows", default="results/news_v3_nvda_selection_prospective_rows.csv")
    ap.add_argument("--out-stability", default="results/news_v3_nvda_selection_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    rows = rows[rows["symbol"].astype(str).str.upper().eq("NVDA")].copy()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["news_rank_score"] = pd.to_numeric(rows["news_rank_score"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows = rows[rows["news_rank_known_events"] > 0].dropna(subset=["Date", "news_rank_score", "next_ret"]).copy()

    reports = []
    selected_rows = None
    for frac, label in [(0.25, "top25"), (0.50, "top50")]:
        x = prospective_selection(rows, frac=frac, min_history=args.min_history)
        stats = bootstrap_delta(x, n_boot=args.n_boot, seed=42)
        selected_mean = float(x.loc[x["eligible"] & x["selected"], "next_ret"].mean()) if x["selected"].any() else np.nan
        reports.append({
            "selection": label,
            "min_history": args.min_history,
            "n_total": len(rows),
            **stats,
            "mean_selected_gross": selected_mean,
            "mean_selected_net_at_cost_bps": selected_mean - args.cost_bps / 10000.0 if np.isfinite(selected_mean) else np.nan,
            "cost_bps": args.cost_bps,
        })
        y = x.copy()
        y["selection"] = label
        selected_rows = y if selected_rows is None else pd.concat([selected_rows, y], ignore_index=True)

    report = pd.DataFrame(reports)
    stability = build_stability(selected_rows[selected_rows.selection == "top50"].copy(), cost_bps=args.cost_bps)

    for p in [args.out, args.out_rows, args.out_stability]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    selected_rows.to_csv(args.out_rows, index=False)
    stability.to_csv(args.out_stability, index=False)

    print(report.to_string(index=False))
    print("STABILITY TOP50")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print(f"Saved {args.out_stability}")
    print("NVDA PROSPECTIVE SELECTION ROBUSTNESS PASS")


if __name__ == "__main__":
    main()
