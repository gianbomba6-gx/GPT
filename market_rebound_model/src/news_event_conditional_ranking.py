"""Robust ranking of event-conditioned rebound contexts from OOS diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASELINE_CONDITION = "all_candidates"
EVENT_CONDITIONS = ("event_negative", "event_dominant_negative")
MIN_N = 20


def build_ranking(diag: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    required = {"symbol", "event", "condition", "n", "mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"}
    missing = required - set(diag.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")

    d = diag.copy()
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    for c in ["n", "mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    baseline = (
        d[d["condition"] == BASELINE_CONDITION]
        .set_index("symbol")[["mean_next_ret", "hit_2pct", "hit_3pct", "hit_5pct"]]
        .rename(columns=lambda c: f"baseline_{c}")
    )
    candidates = d[d["condition"].isin(EVENT_CONDITIONS) & (d["event"] != "all")].copy()
    if candidates.empty:
        raise ValueError("No event-conditional rows available")
    candidates = candidates.merge(baseline, left_on="symbol", right_index=True, how="left")
    candidates["delta_mean"] = candidates["mean_next_ret"] - candidates["baseline_mean_next_ret"]
    candidates["delta_hit_2pct"] = candidates["hit_2pct"] - candidates["baseline_hit_2pct"]
    candidates["delta_hit_3pct"] = candidates["hit_3pct"] - candidates["baseline_hit_3pct"]
    candidates["delta_hit_5pct"] = candidates["hit_5pct"] - candidates["baseline_hit_5pct"]

    rows = []
    for _, r in candidates.iterrows():
        n = int(r["n"]) if pd.notna(r["n"]) else 0
        delta = float(r["delta_mean"]) if pd.notna(r["delta_mean"]) else float("nan")
        d2 = float(r["delta_hit_2pct"]) if pd.notna(r["delta_hit_2pct"]) else float("nan")
        d3 = float(r["delta_hit_3pct"]) if pd.notna(r["delta_hit_3pct"]) else float("nan")
        d5 = float(r["delta_hit_5pct"]) if pd.notna(r["delta_hit_5pct"]) else float("nan")

        if n < min_n:
            tier = "INSUFFICIENT"
        elif delta > 0 and d2 >= 0 and d3 >= 0 and d5 >= 0:
            tier = "FAVORABLE"
        elif delta < 0 and d2 <= 0 and d3 <= 0 and d5 <= 0:
            tier = "AVOID"
        else:
            tier = "WATCH"

        consistency = int(delta > 0 and d2 >= 0 and d3 >= 0) + int(delta < 0 and d2 <= 0 and d3 <= 0)
        score = float(max(delta, 0.0) * (1 + 0.25 * consistency) * min(n / max(min_n, 1), 5.0)) if pd.notna(delta) else 0.0
        rows.append({
            "symbol": r["symbol"], "event": r["event"], "condition": r["condition"], "n": n,
            "mean_next_ret": r["mean_next_ret"], "baseline_mean_next_ret": r["baseline_mean_next_ret"], "delta_mean": delta,
            "hit_2pct": r["hit_2pct"], "baseline_hit_2pct": r["baseline_hit_2pct"], "delta_hit_2pct": d2,
            "hit_3pct": r["hit_3pct"], "baseline_hit_3pct": r["baseline_hit_3pct"], "delta_hit_3pct": d3,
            "hit_5pct": r["hit_5pct"], "baseline_hit_5pct": r["baseline_hit_5pct"], "delta_hit_5pct": d5,
            "tier": tier, "score": score,
        })

    return pd.DataFrame(rows).sort_values(["symbol", "tier", "score", "n"], ascending=[True, True, False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostics")
    ap.add_argument("--out", default="results/news_v3_event_conditional_ranking.csv")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    out = build_ranking(pd.read_csv(args.diagnostics), min_n=args.min_n)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    print(out.to_string(index=False))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
