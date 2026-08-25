from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _select_with_history(x: pd.DataFrame, frac: float, direction: str, min_history: int) -> pd.DataFrame:
    x = x.sort_values("Date").reset_index(drop=True).copy()
    x["eligible"] = False
    x["selected"] = False
    scores = x["news_rank_score"].to_numpy(float)
    rets = x["next_ret"].to_numpy(float)
    for i in range(len(x)):
        prior_scores = pd.Series(scores[:i]).dropna()
        if len(prior_scores) < min_history:
            continue
        x.at[i, "eligible"] = True
        hist = prior_scores if direction == "normal" else -prior_scores
        current = scores[i] if direction == "normal" else -scores[i]
        threshold = float(hist.quantile(1.0 - frac))
        x.at[i, "selected"] = bool(current >= threshold)
    return x


def _select_calibrated(x: pd.DataFrame, frac: float, min_history: int, min_calibration: int) -> pd.DataFrame:
    x = x.sort_values("Date").reset_index(drop=True).copy()
    x["eligible"] = False
    x["selected"] = False
    x["direction"] = "normal"
    scores = x["news_rank_score"].to_numpy(float)
    for i in range(len(x)):
        prior = x.loc[: i - 1, ["news_rank_score", "next_ret"]].dropna()
        if len(prior) < min_history:
            continue
        if len(prior) < min_calibration:
            direction = "normal"
        else:
            normal_q = prior["news_rank_score"].quantile(1.0 - frac)
            inverted_q = prior["news_rank_score"].quantile(frac)
            normal_sel = prior["news_rank_score"] >= normal_q
            inverted_sel = prior["news_rank_score"] <= inverted_q
            normal_delta = prior.loc[normal_sel, "next_ret"].mean() - prior["next_ret"].mean() if normal_sel.any() else -np.inf
            inverted_delta = prior.loc[inverted_sel, "next_ret"].mean() - prior["next_ret"].mean() if inverted_sel.any() else -np.inf
            direction = "normal" if normal_delta >= inverted_delta else "inverted"
        hist = prior["news_rank_score"] if direction == "normal" else -prior["news_rank_score"]
        current = scores[i] if direction == "normal" else -scores[i]
        threshold = float(hist.quantile(1.0 - frac))
        x.at[i, "eligible"] = True
        x.at[i, "direction"] = direction
        x.at[i, "selected"] = bool(current >= threshold)
    return x


def _bootstrap(values: np.ndarray, selected: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss = selected[idx]
        boot[i] = np.nan if not ss.any() else float(values[idx][ss].mean() - values[idx].mean())
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return observed, np.nan, np.nan
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def evaluate(x: pd.DataFrame, strategy: str, frac: float, n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    z = x[x["eligible"] & x["next_ret"].notna()].copy().reset_index(drop=True)
    selected = z["selected"].to_numpy(bool)
    values = z["next_ret"].to_numpy(float)
    if len(z) == 0 or not selected.any():
        stats = {"strategy": strategy, "n_total": len(x), "n_eligible": len(z), "n_selected": int(selected.sum()), "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "mean_selected": np.nan}
    else:
        delta, lo, hi = _bootstrap(values, selected, n_boot, seed)
        stats = {"strategy": strategy, "n_total": len(x), "n_eligible": len(z), "n_selected": int(selected.sum()), "delta_mean": delta, "ci_low": lo, "ci_high": hi, "mean_selected": float(values[selected].mean())}
    return stats, z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_direction_strategy_benchmark.csv")
    ap.add_argument("--out-rows", default="results/news_v3_direction_strategy_benchmark_rows.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--min-calibration", type=int, default=8)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["news_rank_score"] = pd.to_numeric(rows["news_rank_score"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["news_rank_known_events"] = pd.to_numeric(rows["news_rank_known_events"], errors="coerce").fillna(0)
    valid = rows[(rows["news_rank_known_events"] > 0)].dropna(subset=["Date", "news_rank_score", "next_ret"]).copy()

    reports = []
    row_parts = []
    for symbol, g in valid.groupby("symbol", sort=True):
        for frac, label in ((0.25, "top25"), (0.50, "top50")):
            strategies = {
                "normal_fixed": _select_with_history(g, frac, "normal", args.min_history),
                "inverted_fixed": _select_with_history(g, frac, "inverted", args.min_history),
                "calibrated": _select_calibrated(g, frac, args.min_history, args.min_calibration),
            }
            for strategy, x in strategies.items():
                stats, z = evaluate(x, strategy, frac, args.n_boot, 42)
                stats.update({"symbol": symbol, "selection": label, "min_history": args.min_history, "min_calibration": args.min_calibration})
                reports.append(stats)
                z = z.copy()
                z["symbol"] = symbol
                z["selection"] = label
                z["strategy"] = strategy
                if "direction" not in z:
                    z["direction"] = "normal" if strategy == "normal_fixed" else ("inverted" if strategy == "inverted_fixed" else pd.NA)
                row_parts.append(z)

    report = pd.DataFrame(reports)[["symbol","selection","strategy","n_total","n_eligible","n_selected","delta_mean","ci_low","ci_high","mean_selected","min_history","min_calibration"]]
    rows_out = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    pivot = report.pivot(index=["symbol","selection"], columns="strategy", values="delta_mean").reset_index()
    for c in ("normal_fixed", "inverted_fixed", "calibrated"):
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot["calibrated_minus_best_fixed"] = pivot["calibrated"] - pivot[["normal_fixed","inverted_fixed"]].max(axis=1)
    pivot = pivot[["symbol","selection","normal_fixed","inverted_fixed","calibrated","calibrated_minus_best_fixed"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    rows_out.to_csv(args.out_rows, index=False)
    pivot_path = Path(args.out).with_name(Path(args.out).stem + "_pivot.csv")
    pivot.to_csv(pivot_path, index=False)
    print("DIRECTION STRATEGY BENCHMARK")
    print(report.to_string(index=False))
    print("BENCHMARK PIVOT")
    print(pivot.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print(f"Saved {pivot_path}")
    print("DIRECTION STRATEGY BENCHMARK PASS")


if __name__ == "__main__":
    main()
