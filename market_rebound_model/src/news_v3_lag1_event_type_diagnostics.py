from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .news_event_secondary_ranking import EVENT_TYPES, _event_features_lagged, _event_score_rules
    from .news_v3_incremental_value import bootstrap_delta, prospective_filter
except ImportError:
    from news_event_secondary_ranking import EVENT_TYPES, _event_features_lagged, _event_score_rules
    from news_v3_incremental_value import bootstrap_delta, prospective_filter


def score_one_event_type(rows, raw, event_type, min_n, shrink_k):
    x = rows.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["v1_top20"] = x["v1_top20"].fillna(False).astype(bool)
    x = x.dropna(subset=["Date", "symbol"]).sort_values(["Date", "symbol", "_row_id"]).reset_index(drop=True)

    col = f"negative_event_{event_type}_share"
    events = _event_features_lagged(raw, 1, x[["Date", "symbol"]]).copy()
    if col not in events.columns:
        events[col] = 0.0
    events = events[["Date", "symbol", col]]
    x = x.drop(columns=[col], errors="ignore").merge(events, on=["Date", "symbol"], how="left", validate="many_to_one")
    x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    candidates = x[x["v1_top20"]].copy()
    parts = []
    for day in sorted(candidates["Date"].unique()):
        test = candidates[candidates["Date"] == day].copy()
        prior = candidates[candidates["Date"] < day].copy()
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
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty and set(out["_row_id"]) != set(candidates["_row_id"]):
        raise SystemExit(f"Row-id coverage changed for event type {event_type}")
    if not out.empty:
        out["news_rank_score"] = pd.to_numeric(out["event_score"], errors="coerce")
        out["news_rank_known_events"] = pd.to_numeric(out["event_known"], errors="coerce").fillna(0)
    return out


def evaluate(base, scored, frac, direction, min_history, n_boot):
    news = scored[(scored["event_known"] > 0) & scored["news_rank_score"].notna()].copy()
    base = base[base["next_ret"].notna()].copy()
    if news.empty:
        return len(base), 0, 0, float(base["next_ret"].mean()) if len(base) else 0.0, 0.0, 0.0, 0.0, 0.0, "NO_ELIGIBLE_CASES"
    selected_rows = prospective_filter(news, frac=frac, direction=direction, min_history=min_history)
    eligible = selected_rows[selected_rows["eligible"] & selected_rows["next_ret"].notna()].copy()
    selected = eligible[eligible["selected"]].copy()
    mean_base = float(base["next_ret"].mean()) if len(base) else 0.0
    if selected.empty:
        return len(base), len(eligible), 0, mean_base, 0.0, 0.0, 0.0, 0.0, "NO_SELECTED_CASES"
    ids = set(selected["_row_id"])
    mask = np.array([rid in ids for rid in base["_row_id"]], dtype=bool)
    if not mask.any():
        return len(base), len(eligible), len(selected), mean_base, np.nan, np.nan, np.nan, np.nan, "INVALID_SELECTION_ID_ALIGNMENT"
    values = base["next_ret"].to_numpy(float)
    delta, lo, hi = bootstrap_delta(values, mask, n_boot, 42)
    return len(base), len(eligible), len(selected), mean_base, float(values[mask].mean()), delta, lo, hi, "OK"


def annual_stability(base, scored, frac, direction, min_history):
    news = scored[(scored["event_known"] > 0) & scored["news_rank_score"].notna()].copy()
    if news.empty:
        return pd.DataFrame()
    sel = prospective_filter(news, frac=frac, direction=direction, min_history=min_history)
    out = []
    for symbol, b in base.groupby("symbol", sort=True):
        b = b.copy()
        b["year"] = pd.to_datetime(b["Date"]).dt.year
        s = sel[sel["symbol"] == symbol].copy()
        s["year"] = pd.to_datetime(s["Date"]).dt.year
        for year, by in b.groupby("year", sort=True):
            by = by[by["next_ret"].notna()]
            sy = s[(s["year"] == year) & s["eligible"] & s["selected"] & s["next_ret"].notna()]
            mean_base = float(by["next_ret"].mean()) if len(by) else 0.0
            mean_sel = float(sy["next_ret"].mean()) if len(sy) else 0.0
            out.append({
                "symbol": symbol,
                "year": int(year),
                "n_base": len(by),
                "n_eligible": int(((s["year"] == year) & s["eligible"] & s["next_ret"].notna()).sum()),
                "n_selected": len(sy),
                "mean_base": mean_base,
                "mean_selected_gross": mean_sel,
                "delta_mean": mean_sel - mean_base if len(sy) else 0.0,
                "status": "OK" if len(sy) else "NO_SELECTED_CASES",
            })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_event_type_diagnostics.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_event_type_stability.csv")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--shrink-k", type=float, default=50.0)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.min_n < 1 or args.shrink_k < 0 or args.n_boot < 100:
        raise SystemExit("Invalid diagnostic parameters")

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)
    base = rows[rows["v1_top20"]].dropna(subset=["Date"]).copy()
    base["_row_id"] = np.arange(len(base))

    reports, stability_parts = [], []
    for event_type in EVENT_TYPES:
        scored = score_one_event_type(base, raw, event_type, args.min_n, args.shrink_k)
        for symbol, b in base.groupby("symbol", sort=True):
            ss = scored[scored["symbol"] == symbol].copy()
            for frac, label in ((0.25, "top25"), (0.50, "top50")):
                for direction in ("normal", "inverted"):
                    n_base, n_eligible, n_selected, mean_base, mean_sel, delta, lo, hi, status = evaluate(
                        b, ss, frac, direction, args.min_n, args.n_boot
                    )
                    reports.append({
                        "event_type": event_type,
                        "symbol": symbol,
                        "selection": label,
                        "direction": direction,
                        "n_base": n_base,
                        "n_eligible": n_eligible,
                        "n_selected": n_selected,
                        "mean_base": mean_base,
                        "mean_selected_gross": mean_sel,
                        "delta_mean": delta,
                        "ci_low": lo,
                        "ci_high": hi,
                        "status": status,
                    })
                st = annual_stability(b, ss, frac, "inverted", args.min_n)
                if not st.empty:
                    st["event_type"] = event_type
                    st["selection"] = label
                    st["direction"] = "inverted"
                    stability_parts.append(st)

    report = pd.DataFrame(reports).sort_values(["event_type", "symbol", "selection", "direction"])
    stability = pd.concat(stability_parts, ignore_index=True) if stability_parts else pd.DataFrame()
    ok = report[report["status"] == "OK"]
    if not ok.empty and not np.isfinite(ok[["n_base", "n_eligible", "n_selected", "mean_base", "mean_selected_gross", "delta_mean", "ci_low", "ci_high"]].to_numpy(float)).all():
        raise SystemExit("Invalid prospective event-type diagnostic result")
    if (report["n_selected"] > report["n_eligible"]).any():
        raise SystemExit("Invalid prospective event-type selection counts")
    if not stability.empty:
        ok_st = stability[stability["status"] == "OK"]
        if not ok_st.empty and not np.isfinite(ok_st[["n_base", "n_eligible", "n_selected", "mean_base", "mean_selected_gross", "delta_mean"]].to_numpy(float)).all():
            raise SystemExit("Invalid prospective event-type stability result")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    stability.to_csv(args.out_stability, index=False)
    print("NEWS V3 LAG1 EVENT TYPE DIAGNOSTICS")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 EVENT TYPE DIAGNOSTICS PASS")
    print("NEWS V3 LAG1 EVENT TYPE STABILITY")
    print(stability.to_string(index=False))
    print("NEWS V3 LAG1 EVENT TYPE STABILITY PASS")
    print(f"Saved {args.out}")
    print(f"Saved {args.out_stability}")


if __name__ == "__main__":
    main()
