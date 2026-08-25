from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from news_event_secondary_ranking import score_oos
from news_v3_incremental_value import evaluate, prospective_filter


def run_sweep(rows: pd.DataFrame, raw: pd.DataFrame, min_ns: list[int], cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = rows.copy()
    base["symbol"] = base["symbol"].astype(str).str.upper().str.strip()
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    base["next_ret"] = pd.to_numeric(base["next_ret"], errors="coerce")
    base["v1_top20"] = base["v1_top20"].fillna(False).astype(bool)
    base = base.dropna(subset=["Date", "next_ret"]).copy()
    base["_row_id"] = np.arange(len(base))

    reports: list[dict] = []
    stability: list[dict] = []

    for min_n in min_ns:
        scored = score_oos(
            base.drop(columns=["_row_id"], errors="ignore"),
            raw=raw,
            min_n=min_n,
            shrink_k=50.0,
            invert_score=True,
            event_lag_days=1,
        )
        if scored.empty:
            raise RuntimeError(f"No scored rows for min_n={min_n}")

        scored["Date"] = pd.to_datetime(scored["Date"], errors="coerce")
        scored["symbol"] = scored["symbol"].astype(str).str.upper().str.strip()
        scored = scored.merge(
            base[["_row_id", "Date", "symbol", "next_ret", "v1_top20"]],
            on=["Date", "symbol", "next_ret", "v1_top20"],
            how="left",
            validate="many_to_one",
        )
        scored = scored.drop_duplicates(subset=["_row_id"], keep="first")
        if scored["_row_id"].isna().any():
            raise RuntimeError(f"Unmatched row ids for min_n={min_n}")
        scored["_row_id"] = scored["_row_id"].astype(int)

        for symbol, base_symbol in base[base["v1_top20"]].groupby("symbol", sort=True):
            news = scored[(scored["symbol"] == symbol) & (scored["news_rank_known_events"] > 0)].copy()
            mean_base = float(base_symbol["next_ret"].mean()) if len(base_symbol) else 0.0
            if news.empty:
                for frac, label in ((0.25, "news_top25"), (0.50, "news_top50")):
                    reports.append({
                        "min_n": min_n,
                        "symbol": symbol,
                        "filter": label,
                        "n_base": len(base_symbol),
                        "n_news_candidates": 0,
                        "n_eligible": 0,
                        "n_selected": 0,
                        "mean_base": mean_base,
                        "mean_selected_gross": 0.0,
                        "mean_selected_net": 0.0,
                        "delta_mean": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "cost_bps": cost_bps,
                        "status": "INSUFFICIENT_HISTORY",
                    })
                continue

            for frac, label in ((0.25, "news_top25"), (0.50, "news_top50")):
                selected_rows = prospective_filter(news, frac, "inverted", min_history=min_n)
                result = evaluate(
                    base_symbol,
                    selected_rows,
                    f"{symbol}: {label}",
                    n_boot=10000,
                    cost_bps=cost_bps,
                )
                result.update({
                    "min_n": min_n,
                    "symbol": symbol,
                    "filter": label,
                    "n_news_candidates": len(news),
                })
                reports.append(result)

                for year, ybase in base_symbol.groupby(base_symbol["Date"].dt.year, sort=True):
                    ysel = selected_rows[selected_rows["Date"].dt.year == year]
                    yeligible = ysel[ysel["eligible"]]
                    yselected = yeligible[yeligible["selected"]]
                    mean_ybase = float(ybase["next_ret"].mean()) if len(ybase) else 0.0
                    mean_ysel = float(yselected["next_ret"].mean()) if len(yselected) else 0.0
                    stability.append({
                        "min_n": min_n,
                        "symbol": symbol,
                        "year": int(year),
                        "filter": label,
                        "n_base": len(ybase),
                        "n_eligible": len(yeligible),
                        "n_selected": len(yselected),
                        "mean_base": mean_ybase,
                        "mean_selected_gross": mean_ysel,
                        "delta_mean": mean_ysel - mean_ybase if len(yselected) else 0.0,
                        "status": "OK" if len(yselected) else "NO_SELECTED_CASES",
                    })

    return pd.DataFrame(reports), pd.DataFrame(stability)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_csv")
    ap.add_argument("--out", default="results/news_v3_lag1_min_n_sweep.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_min_n_sweep_stability.csv")
    ap.add_argument("--min-ns", default="5,10,15,20")
    ap.add_argument("--cost-bps", type=float, default=20.0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_csv)
    min_ns = [int(x.strip()) for x in args.min_ns.split(",") if x.strip()]
    if not min_ns or any(x <= 0 for x in min_ns):
        raise SystemExit("min-n values must be positive")

    report, stability = run_sweep(rows, raw, min_ns, args.cost_bps)
    if report.empty:
        raise SystemExit("Empty min-n sweep report")

    for col in ["n_base", "n_news_candidates", "n_eligible", "n_selected"]:
        if (report[col] < 0).any():
            raise SystemExit(f"Invalid negative count in {col}")
    if (report["n_selected"] > report["n_base"]).any():
        raise SystemExit("Selected cases exceed baseline cases")
    numeric = ["mean_base", "mean_selected_gross", "mean_selected_net", "delta_mean", "ci_low", "ci_high", "cost_bps"]
    if not np.isfinite(report[numeric].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite min-n sweep result")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    stability.to_csv(args.out_stability, index=False)

    print("NEWS V3 LAG1 MIN-N SWEEP")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 MIN-N STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {out}")
    print(f"Saved {args.out_stability}")
    print("NEWS V3 LAG1 MIN-N SWEEP PASS")


if __name__ == "__main__":
    main()
