from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from news_v3_multi_symbol_selection_robustness import run_symbol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_multi_symbol_direction_check.csv")
    ap.add_argument("--out-rows", default="results/news_v3_multi_symbol_direction_rows.csv")
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

    reports = []
    row_parts = []
    for direction, factor in [("normal", -1.0), ("inverted", 1.0)]:
        scored = valid.copy()
        scored["news_rank_score"] = scored["news_rank_score"] * factor
        for symbol, g in scored.groupby("symbol", sort=True):
            for frac in (0.25, 0.50):
                result, selected_rows = run_symbol(
                    g.copy(), symbol, frac, args.min_history, args.n_boot, 42, args.cost_bps
                )
                result["direction"] = direction
                reports.append(result)
                selected_rows["direction"] = direction
                row_parts.append(selected_rows)

    report = pd.DataFrame(reports)
    rows_out = pd.concat(row_parts, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    rows_out.to_csv(args.out_rows, index=False)

    pivot = report.pivot(index=["symbol", "selection"], columns="direction", values="delta_mean").reset_index()
    for direction in ("normal", "inverted"):
        if direction not in pivot:
            pivot[direction] = pd.NA
    pivot["spread_inverted_minus_normal"] = pivot["inverted"] - pivot["normal"]
    pivot = pivot[["symbol", "selection", "normal", "inverted", "spread_inverted_minus_normal"]]
    pivot_path = Path(args.out).with_name(Path(args.out).stem + "_pivot.csv")
    pivot.to_csv(pivot_path, index=False)

    print("MULTI-SYMBOL SCORE DIRECTION")
    print(report.to_string(index=False))
    print("DIRECTION COMPARISON")
    print(pivot.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print(f"Saved {pivot_path}")
    print("MULTI-SYMBOL SCORE DIRECTION PASS")


if __name__ == "__main__":
    main()
