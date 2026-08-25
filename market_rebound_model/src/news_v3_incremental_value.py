from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prospective_filter(rows: pd.DataFrame, frac: float, direction: str, min_history: int) -> pd.DataFrame:
    x = rows.sort_values(["Date", "symbol", "_row_id"]).reset_index(drop=True).copy()
    x["eligible"] = False
    x["selected"] = False
    for symbol, idxs in x.groupby("symbol", sort=False).groups.items():
        idxs = list(idxs)
        for pos, i in enumerate(idxs):
            prior = x.loc[idxs[:pos], "news_rank_score"].dropna()
            if len(prior) < min_history:
                continue
            x.at[i, "eligible"] = True
            hist = prior if direction == "normal" else -prior
            current = x.at[i, "news_rank_score"] if direction == "normal" else -x.at[i, "news_rank_score"]
            # Use deterministic rank-with-ties instead of a quantile cutoff.
            # This prevents a large block of equal zero scores from being
            # selected en masse and leaking the intended top-fraction semantics.
            combined = pd.concat([hist.reset_index(drop=True), pd.Series([current])], ignore_index=True)
            current_rank = int(combined.rank(method="first", ascending=False).iloc[-1])
            cutoff = max(1, int(np.ceil(frac * len(combined))))
            x.at[i, "selected"] = current_rank <= cutoff
    return x


def bootstrap_delta(values: np.ndarray, selected: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
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


def evaluate(candidates: pd.DataFrame, selected_rows: pd.DataFrame, label: str, n_boot: int, cost_bps: float) -> dict:
    base = candidates[candidates["next_ret"].notna()].copy()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_incremental_value.csv")
    ap.add_argument("--out-rows", default="results/news_v3_incremental_value_rows.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--min-history", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=20.0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["news_rank_score"] = pd.to_numeric(rows["news_rank_score"], errors="coerce")
    rows["news_rank_known_events"] = pd.to_numeric(rows.get("news_rank_known_events", 0), errors="coerce").fillna(0)
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)

    # V1 baseline uses all V1 top20 candidates. News selection is evaluated
    # only on rows where the news layer actually has an observed event score.
    base_rows = rows[rows["v1_top20"]].dropna(subset=["Date", "next_ret"]).copy()
    news_rows = base_rows[base_rows["news_rank_known_events"] > 0].dropna(subset=["news_rank_score"]).copy()
    base_rows["_row_id"] = np.arange(len(base_rows))
    news_rows["_row_id"] = np.arange(len(news_rows))

    reports = []
    parts = []
    for symbol, base in base_rows.groupby("symbol", sort=True):
        news = news_rows[news_rows["symbol"] == symbol].copy()
        if news.empty:
            reports.extend([
                {
                    "set": f"{symbol}: news_top25: normal", "n_base": len(base), "n_news_candidates": 0, "n_eligible": 0, "n_selected": 0,
                    "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0, "mean_selected_gross": 0.0, "mean_selected_net": 0.0,
                    "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "cost_bps": args.cost_bps, "status": "INSUFFICIENT_HISTORY",
                },
            ])
            for frac, label in ((0.25, "news_top25"), (0.50, "news_top50")):
                if label != "news_top25":
                    reports.append({
                        "set": f"{symbol}: {label}: normal", "n_base": len(base), "n_news_candidates": 0, "n_eligible": 0, "n_selected": 0,
                        "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0, "mean_selected_gross": 0.0, "mean_selected_net": 0.0,
                        "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "cost_bps": args.cost_bps, "status": "INSUFFICIENT_HISTORY",
                    })
            continue

        for frac, label in ((0.25, "news_top25"), (0.50, "news_top50")):
            for direction in ("normal", "inverted"):
                x = prospective_filter(news, frac, direction, args.min_history)
                result = evaluate(base, x, f"{symbol}: {label}: {direction}", args.n_boot, args.cost_bps)
                result["n_news_candidates"] = len(news)
                reports.append(result)
                y = x.copy()
                y["symbol"] = symbol
                y["filter"] = label
                y["direction"] = direction
                parts.append(y)

        # Explicit baseline row.
        mean_base = float(base["next_ret"].mean()) if len(base) else 0.0
        reports.append({
            "set": f"{symbol}: V1 top20 baseline",
            "n_base": len(base), "n_news_candidates": len(news), "n_eligible": len(news), "n_selected": len(base),
            "mean_base": mean_base, "mean_selected_gross": mean_base, "mean_selected_net": mean_base - args.cost_bps / 10000.0,
            "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "cost_bps": args.cost_bps, "status": "OK",
        })

    report = pd.DataFrame(reports)
    rows_out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    rows_out.to_csv(args.out_rows, index=False)

    print("NEWS V3 INCREMENTAL VALUE")
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_rows}")
    print("NEWS V3 INCREMENTAL VALUE PASS")


if __name__ == "__main__":
    main()
