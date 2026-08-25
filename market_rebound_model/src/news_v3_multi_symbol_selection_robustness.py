from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prospective_selection(rows: pd.DataFrame, frac: float, min_history: int) -> pd.DataFrame:
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    if min_history < 1:
        raise ValueError("min_history must be >= 1")

    x = rows.sort_values("Date").reset_index(drop=True).copy()
    selected = np.zeros(len(x), dtype=bool)
    eligible = np.zeros(len(x), dtype=bool)

    for i in range(len(x)):
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


def run_symbol(rows: pd.DataFrame, symbol: str, frac: float, min_history: int, n_boot: int, seed: int, cost_bps: float) -> tuple[dict, pd.DataFrame]:
    x = prospective_selection(rows, frac=frac, min_history=min_history)
    stats = bootstrap_delta(x, n_boot=n_boot, seed=seed)
    selected = x[x["eligible"] & x["selected"]]
    mean_selected = float(selected["next_ret"].mean()) if not selected.empty else np.nan
    result = {
        "symbol": symbol,
        "selection": f"top{int(frac * 100)}",
        "n_total": len(rows),
        **stats,
        "mean_selected_gross": mean_selected,
        "mean_selected_net_at_cost_bps": mean_selected - cost_bps / 10000.0 if np.isfinite(mean_selected) else np.nan,
        "cost_bps": cost_bps,
    }
    x["symbol"] = symbol
    x["selection"] = f"top{int(frac * 100)}"
    return result, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_multi_symbol_selection_robustness.csv")
    ap.add_argument("--out-rows", default="results/news_v3_multi_symbol_selection_prospective_rows.csv")
    ap.add_argument("--out-stability", default="results/news_v3_multi_symbol_selection_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["news_rank_score"] = pd.to_numeric(rows["news_rank_score"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["news_rank_known_events"] = pd.to_numeric(rows["news_rank_known_events"], errors="coerce").fillna(0)

    valid = rows[(rows["news_rank_known_events"] > 0)].dropna(subset=["Date", "news_rank_score", "next_ret"]).copy()
    coverage = rows.groupby("symbol", sort=True).agg(
        n_rows=("symbol", "size"),
        n_known_events=("news_rank_known_events", lambda s: int((s > 0).sum())),
        n_with_next_ret=("next_ret", lambda s: int(s.notna().sum())),
    ).reset_index()

    reports: list[dict] = []
    row_parts: list[pd.DataFrame] = []
    for symbol, g in valid.groupby("symbol", sort=True):
        for frac in (0.25, 0.50):
            result, selected_rows = run_symbol(g.copy(), symbol, frac, args.min_history, args.n_boot, 42, args.cost_bps)
            reports.append(result)
            row_parts.append(selected_rows)

    report = pd.DataFrame(reports)
    if report.empty:
        raise SystemExit("No symbol has usable ranking rows")

    stability_rows: list[dict] = []
    top50 = pd.concat([x for x in row_parts if (x["selection"].iloc[0] == "top50")], ignore_index=True)
    eligible = top50[top50["eligible"] & top50["next_ret"].notna()].copy()
    cost = args.cost_bps / 10000.0
    for (symbol, period), g in eligible.groupby(["symbol", eligible["Date"].dt.year.astype(int)], sort=True):
        s = g[g["selected"]]
        mean_all = float(g["next_ret"].mean())
        mean_selected = float(s["next_ret"].mean()) if not s.empty else np.nan
        stability_rows.append({
            "symbol": symbol,
            "period": int(period),
            "n_eligible": len(g),
            "n_selected": len(s),
            "mean_all": mean_all,
            "mean_selected_gross": mean_selected,
            "mean_selected_net": mean_selected - cost if np.isfinite(mean_selected) else np.nan,
            "delta_mean": mean_selected - mean_all if np.isfinite(mean_selected) else np.nan,
        })
    stability = pd.DataFrame(stability_rows)

    merged = coverage.merge(report.groupby("symbol")["n_eligible"].max().rename("max_n_eligible"), on="symbol", how="left")
    merged["max_n_eligible"] = merged["max_n_eligible"].fillna(0).astype(int)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    merged.to_csv(args.out.replace(".csv", "_coverage.csv"), index=False)
    pd.concat(row_parts, ignore_index=True).to_csv(args.out_rows, index=False)
    stability.to_csv(args.out_stability, index=False)

    print("MULTI-SYMBOL COVERAGE")
    print(merged.to_string(index=False))
    print("MULTI-SYMBOL PROSPECTIVE ROBUSTNESS")
    print(report.to_string(index=False))
    print("TOP50 STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print(f"Saved {args.out_stability}")
    print("MULTI-SYMBOL NEWS SELECTION ROBUSTNESS PASS")


if __name__ == "__main__":
    main()
