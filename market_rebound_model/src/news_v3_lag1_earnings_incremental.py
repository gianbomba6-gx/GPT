from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .news_v3_lag1_event_type_diagnostics import score_one_event_type
    from .news_v3_incremental_value import bootstrap_delta, prospective_filter
except ImportError:
    from news_v3_lag1_event_type_diagnostics import score_one_event_type
    from news_v3_incremental_value import bootstrap_delta, prospective_filter


def evaluate(base: pd.DataFrame, selected_rows: pd.DataFrame, label: str, n_boot: int, cost_bps: float) -> dict:
    base = base[base["next_ret"].notna()].copy()
    selected = selected_rows[selected_rows["eligible"] & selected_rows["selected"] & selected_rows["next_ret"].notna()].copy()
    eligible = selected_rows[selected_rows["eligible"] & selected_rows["next_ret"].notna()].copy()
    result = {
        "set": label,
        "n_base": len(base),
        "n_eligible": len(eligible),
        "n_selected": int(len(selected)),
        "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0,
        "mean_selected_gross": 0.0,
        "mean_selected_net": 0.0,
        "delta_mean": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "cost_bps": cost_bps,
        "status": "INSUFFICIENT_HISTORY" if len(eligible) == 0 else ("NO_SELECTED_CASES" if selected.empty else "OK"),
    }
    if selected.empty:
        return result

    base_ids = base["_row_id"].to_numpy()
    selected_ids = set(selected["_row_id"].to_numpy())
    mask = np.array([rid in selected_ids for rid in base_ids], dtype=bool)
    if not mask.any():
        result.update({
            "mean_selected_gross": float("nan"),
            "mean_selected_net": float("nan"),
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "status": "INVALID_SELECTION_ID_ALIGNMENT",
        })
        return result

    values = base["next_ret"].to_numpy(float)
    delta, lo, hi = bootstrap_delta(values, mask, n_boot, 42)
    mean_sel = float(values[mask].mean())
    result.update({
        "mean_selected_gross": mean_sel,
        "mean_selected_net": mean_sel - cost_bps / 10000.0,
        "delta_mean": delta,
        "ci_low": lo,
        "ci_high": hi,
        "status": "OK",
    })
    return result


def annual_rows(base: pd.DataFrame, selected_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, b in base.groupby("symbol", sort=True):
        yb = b.copy()
        yb["year"] = pd.to_datetime(yb["Date"]).dt.year
        ys = selected_rows[selected_rows["symbol"] == symbol].copy()
        ys["year"] = pd.to_datetime(ys["Date"]).dt.year
        for year, by in yb.groupby("year", sort=True):
            by = by[by["next_ret"].notna()].copy()
            sy = ys[(ys["year"] == year) & ys["eligible"] & ys["selected"] & ys["next_ret"].notna()].copy()
            mean_base = float(by["next_ret"].mean()) if len(by) else 0.0
            mean_sel = float(sy["next_ret"].mean()) if len(sy) else 0.0
            out.append({
                "symbol": symbol,
                "year": int(year),
                "n_base": len(by),
                "n_selected": len(sy),
                "mean_base": mean_base,
                "mean_selected_gross": mean_sel,
                "delta_mean": mean_sel - mean_base if len(sy) else 0.0,
                "status": "OK" if len(sy) else "NO_SELECTED_CASES",
            })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_earnings_incremental.csv")
    ap.add_argument("--out-rows", default="results/news_v3_lag1_earnings_incremental_rows.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_earnings_stability.csv")
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--shrink-k", type=float, default=50.0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    args = ap.parse_args()
    if args.min_history < 1 or args.shrink_k < 0 or args.n_boot < 100 or args.cost_bps < 0:
        raise SystemExit("Invalid earnings incremental parameters")

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)
    base = rows[rows["v1_top20"]].dropna(subset=["Date"]).copy()
    base["_row_id"] = np.arange(len(base))

    scored = score_one_event_type(base, raw, "earnings", args.min_history, args.shrink_k)
    if scored.empty:
        raise SystemExit("No V1 top20 rows available for earnings-only test")
    scored["_row_id"] = base["_row_id"].to_numpy()
    scored["news_rank_score"] = pd.to_numeric(scored["event_score"], errors="coerce")
    scored["news_rank_known_events"] = pd.to_numeric(scored["event_known"], errors="coerce").fillna(0)

    reports = []
    parts = []
    for symbol, b in base.groupby("symbol", sort=True):
        s = scored[scored["symbol"] == symbol].copy()
        news = s[(s["event_known"] > 0) & s["news_rank_score"].notna()].copy()
        mean_base = float(b["next_ret"].dropna().mean()) if b["next_ret"].notna().any() else 0.0
        for frac, sel_label in ((0.25, "earnings_top25"), (0.50, "earnings_top50")):
            for direction in ("normal", "inverted"):
                if news.empty:
                    result = {
                        "set": f"{symbol}: {sel_label}: {direction}", "n_base": len(b[b["next_ret"].notna()]),
                        "n_news_candidates": 0, "n_eligible": 0, "n_selected": 0, "mean_base": mean_base,
                        "mean_selected_gross": 0.0, "mean_selected_net": 0.0, "delta_mean": 0.0,
                        "ci_low": 0.0, "ci_high": 0.0, "cost_bps": args.cost_bps, "status": "INSUFFICIENT_HISTORY",
                    }
                    reports.append(result)
                    continue
                x = prospective_filter(news, frac, direction, args.min_history)
                result = evaluate(b, x, f"{symbol}: {sel_label}: {direction}", args.n_boot, args.cost_bps)
                result["n_news_candidates"] = len(news)
                reports.append(result)
                y = x.copy()
                y["filter"] = sel_label
                y["direction"] = direction
                parts.append(y)

        reports.append({
            "set": f"{symbol}: V1 top20 baseline", "n_base": len(b[b["next_ret"].notna()]),
            "n_news_candidates": len(news), "n_eligible": len(b[b["next_ret"].notna()]),
            "n_selected": len(b[b["next_ret"].notna()]), "mean_base": mean_base,
            "mean_selected_gross": mean_base, "mean_selected_net": mean_base - args.cost_bps / 10000.0,
            "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "cost_bps": args.cost_bps, "status": "OK",
        })

    report = pd.DataFrame(reports)
    rows_out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    # Stability is focused on the target variant: earnings-only, inverted, top25/top50.
    target = rows_out[rows_out["direction"] == "inverted"].copy()
    stability_parts = []
    for filter_name, g in target.groupby("filter", sort=True):
        stability_parts.append(annual_rows(base, g))
    stability = pd.concat(stability_parts, ignore_index=True) if stability_parts else pd.DataFrame()
    if not stability.empty:
        stability["filter"] = [
            target.groupby("filter", sort=True).groups.keys().__iter__().__next__()
        ] * 0 if False else stability.get("filter", pd.Series(dtype=object))
        # Rebuild filter labels deterministically from each grouped frame.
        frames = []
        for filter_name, g in target.groupby("filter", sort=True):
            z = annual_rows(base, g)
            z["filter"] = filter_name
            frames.append(z)
        stability = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    rows_out.to_csv(args.out_rows, index=False)
    stability.to_csv(args.out_stability, index=False)
    print("NEWS V3 LAG1 EARNINGS INCREMENTAL VALUE")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 EARNINGS STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print(f"Saved {args.out_stability}")
    print("NEWS V3 LAG1 EARNINGS INCREMENTAL PASS")


if __name__ == "__main__":
    main()
