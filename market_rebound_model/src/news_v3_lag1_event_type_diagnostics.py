from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.news_event_secondary_ranking import (
    EVENT_TYPES,
    _event_features_lagged,
    _event_score_rules,
)


def score_one_event_type(rows: pd.DataFrame, raw: pd.DataFrame, event_type: str, min_n: int, shrink_k: float) -> pd.DataFrame:
    x = rows.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["v1_top20"] = x["v1_top20"].fillna(False).astype(bool)
    x = x.dropna(subset=["Date", "symbol"]).sort_values(["Date", "symbol"]).reset_index(drop=True)

    events = _event_features_lagged(raw, 1, x[["Date", "symbol"]])
    col = f"negative_event_{event_type}_share"
    if col not in events.columns:
        events[col] = 0.0
    events = events[["Date", "symbol", col]]
    x = x.merge(events, on=["Date", "symbol"], how="left")
    x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    candidates = x[x["v1_top20"]].copy()
    parts = []
    for day in sorted(candidates["Date"].unique()):
        prior = candidates[candidates["Date"] < pd.Timestamp(day)].copy()
        test = candidates[candidates["Date"] == pd.Timestamp(day)].copy()
        if test.empty:
            continue
        hist = prior.loc[prior[col] > 0, ["symbol", "next_ret"]].copy()
        if hist.empty:
            test["event_score"] = 0.0
            test["event_known"] = 0
        else:
            hist["event_type"] = event_type
            rules = _event_score_rules(hist, prior, min_n=min_n, shrink_k=shrink_k)
            test["event_score"] = test.apply(
                lambda r: float(r[col]) * rules.get((str(r["symbol"]), event_type), 0.0), axis=1
            )
            test["event_known"] = test["symbol"].map(
                lambda s: int((str(s), event_type) in rules)
            )
        parts.append(test)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def evaluate(scored: pd.DataFrame, frac: float, invert: bool) -> pd.DataFrame:
    out = []
    for symbol, s in scored.groupby("symbol", sort=True):
        base = s[s["next_ret"].notna()].copy()
        elig = s[(s["event_known"] > 0) & s["next_ret"].notna()].copy()
        if elig.empty:
            out.append(dict(symbol=symbol, n_base=len(base), n_eligible=0, n_selected=0,
                            mean_base=float(base.next_ret.mean()) if len(base) else 0.0,
                            mean_selected_gross=0.0, delta_mean=0.0, status="NO_ELIGIBLE_CASES"))
            continue
        vals = -elig["event_score"] if invert else elig["event_score"]
        elig = elig.copy()
        elig["rank"] = vals.rank(method="first", ascending=False)
        cutoff = max(1, int(np.ceil(frac * len(elig))))
        sel = elig[elig["rank"] <= cutoff]
        mean_base = float(base.next_ret.mean())
        mean_sel = float(sel.next_ret.mean()) if len(sel) else 0.0
        out.append(dict(symbol=symbol, n_base=len(base), n_eligible=len(elig), n_selected=len(sel),
                        mean_base=mean_base, mean_selected_gross=mean_sel,
                        delta_mean=mean_sel - mean_base if len(sel) else 0.0,
                        status="OK" if len(sel) else "NO_SELECTED_CASES"))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_event_type_diagnostics.csv")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--shrink-k", type=float, default=50.0)
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    reports = []
    for event_type in EVENT_TYPES:
        scored = score_one_event_type(rows, raw, event_type, args.min_n, args.shrink_k)
        for frac, label in ((0.25, "top25"), (0.50, "top50")):
            for invert in (False, True):
                r = evaluate(scored, frac, invert)
                r["event_type"] = event_type
                r["selection"] = label
                r["direction"] = "inverted" if invert else "normal"
                reports.append(r)

    report = pd.concat(reports, ignore_index=True)
    cols = ["event_type", "symbol", "selection", "direction", "n_base", "n_eligible", "n_selected",
            "mean_base", "mean_selected_gross", "delta_mean", "status"]
    report = report[cols].sort_values(["event_type", "symbol", "selection", "direction"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print("NEWS V3 LAG1 EVENT TYPE DIAGNOSTICS")
    print(report.to_string(index=False))
    print(f"Saved {args.out}")
    print("NEWS V3 LAG1 EVENT TYPE DIAGNOSTICS PASS")


if __name__ == "__main__":
    main()
