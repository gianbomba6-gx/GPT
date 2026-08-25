from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_delta(values: np.ndarray, selected: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    n = len(values)
    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(seed)
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


def calibrate_direction(rows: pd.DataFrame, frac: float, min_history: int = 20, min_calibration: int = 8) -> pd.DataFrame:
    x = rows.sort_values("Date").reset_index(drop=True).copy()
    x["eligible"] = False
    x["direction"] = "normal"
    x["selected"] = False
    for symbol, idxs in x.groupby("symbol", sort=False).groups.items():
        idxs = list(idxs)
        for pos, i in enumerate(idxs):
            prior = x.loc[idxs[:pos], ["news_rank_score", "next_ret"]].dropna()
            if len(prior) < min_history:
                continue
            # Calibrate direction only from the earliest historical block, then freeze it.
            if len(prior) < min_calibration:
                direction = "normal"
            else:
                q = prior["news_rank_score"].quantile(1.0 - frac)
                normal_sel = prior["news_rank_score"] >= q
                inv_sel = prior["news_rank_score"] <= prior["news_rank_score"].quantile(frac)
                normal_delta = prior.loc[normal_sel, "next_ret"].mean() - prior["next_ret"].mean() if normal_sel.any() else -np.inf
                inv_delta = prior.loc[inv_sel, "next_ret"].mean() - prior["next_ret"].mean() if inv_sel.any() else -np.inf
                direction = "normal" if normal_delta >= inv_delta else "inverted"
            x.at[i, "eligible"] = True
            x.at[i, "direction"] = direction
            score = x.at[i, "news_rank_score"] if direction == "normal" else -x.at[i, "news_rank_score"]
            hist_scores = prior["news_rank_score"] if direction == "normal" else -prior["news_rank_score"]
            threshold = hist_scores.quantile(1.0 - frac)
            x.at[i, "selected"] = bool(score >= threshold)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_symbol_direction_calibration.csv")
    ap.add_argument("--out-rows", default="results/news_v3_symbol_direction_calibration_rows.csv")
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
    rows = rows[(rows["news_rank_known_events"] > 0)].dropna(subset=["Date", "news_rank_score", "next_ret"]).copy()

    reports = []
    row_parts = []
    for symbol, g in rows.groupby("symbol", sort=True):
        for frac, label in [(0.25, "top25"), (0.50, "top50")]:
            x = calibrate_direction(g.copy(), frac=frac, min_history=args.min_history, min_calibration=args.min_calibration)
            z = x[x["eligible"] & x["next_ret"].notna()].copy()
            selected = z["selected"].to_numpy(bool)
            values = z["next_ret"].to_numpy(float)
            if len(z) and selected.any():
                delta, lo, hi = _bootstrap_delta(values, selected, args.n_boot, 42)
                mean_selected = float(values[selected].mean())
            else:
                delta = lo = hi = np.nan
                mean_selected = np.nan
            reports.append({
                "symbol": symbol,
                "selection": label,
                "n_total": len(g),
                "n_eligible": len(z),
                "n_selected": int(selected.sum()),
                "delta_mean": delta,
                "ci_low": lo,
                "ci_high": hi,
                "mean_selected": mean_selected,
                "normal_count": int((z["direction"] == "normal").sum()),
                "inverted_count": int((z["direction"] == "inverted").sum()),
            })
            y = x.copy()
            y["selection"] = label
            row_parts.append(y)

    report = pd.DataFrame(reports)
    rows_out = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    rows_out.to_csv(args.out_rows, index=False)
    print("SYMBOL DIRECTION CALIBRATION")
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print("SYMBOL DIRECTION CALIBRATION PASS")


if __name__ == "__main__":
    main()
