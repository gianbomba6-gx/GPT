from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .news_event_secondary_ranking import EVENT_TYPES, _event_features_lagged
    from .news_v3_incremental_value import bootstrap_delta
except ImportError:
    from news_event_secondary_ranking import EVENT_TYPES, _event_features_lagged
    from news_v3_incremental_value import bootstrap_delta


def evaluate(base: pd.DataFrame, event_rows: pd.DataFrame, n_boot: int) -> dict:
    base = base[base["next_ret"].notna()].copy()
    event_rows = event_rows[event_rows["next_ret"].notna()].copy()
    if event_rows.empty:
        return {"n_base": len(base), "n_event_days": 0, "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0, "mean_event_gross": 0.0, "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "status": "NO_EVENT_CASES"}
    base_ids = base["_row_id"].to_numpy()
    event_ids = set(event_rows["_row_id"].to_numpy())
    mask = np.array([rid in event_ids for rid in base_ids], dtype=bool)
    if not mask.any():
        return {"n_base": len(base), "n_event_days": len(event_rows), "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0, "mean_event_gross": np.nan, "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": "INVALID_EVENT_ID_ALIGNMENT"}
    values = base["next_ret"].to_numpy(float)
    delta, lo, hi = bootstrap_delta(values, mask, n_boot, 42)
    return {"n_base": len(base), "n_event_days": int(mask.sum()), "mean_base": float(values.mean()), "mean_event_gross": float(values[mask].mean()), "delta_mean": float(delta), "ci_low": float(lo), "ci_high": float(hi), "status": "OK"}


def annual_stability(base: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, b in base.groupby("symbol", sort=True):
        b = b[b["next_ret"].notna()].copy()
        b["year"] = pd.to_datetime(b["Date"]).dt.year
        e = event_rows[event_rows["symbol"] == symbol].copy()
        e["year"] = pd.to_datetime(e["Date"]).dt.year
        for year, by in b.groupby("year", sort=True):
            ey = e[e["year"] == year]
            mean_base = float(by["next_ret"].mean()) if len(by) else 0.0
            mean_event = float(ey["next_ret"].mean()) if len(ey) else 0.0
            out.append({"symbol": symbol, "year": int(year), "n_base": len(by), "n_event_days": len(ey), "mean_base": mean_base, "mean_event_gross": mean_event, "delta_mean": mean_event - mean_base if len(ey) else 0.0, "status": "OK" if len(ey) else "NO_EVENT_CASES"})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_event_presence_incremental.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_event_presence_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 100:
        raise SystemExit("Invalid presence parameters")

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce").dt.normalize()
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)
    base = rows[rows["v1_top20"]].dropna(subset=["Date"]).copy()
    base = base.sort_values(["Date", "symbol"]).reset_index(drop=True)
    base["_row_id"] = np.arange(len(base))

    reports = []
    stability_parts = []
    for event_type in EVENT_TYPES:
        col = f"negative_event_{event_type}_share"
        features = _event_features_lagged(raw, 1, base[["Date", "symbol"]]).copy()
        if col not in features.columns:
            features[col] = 0.0
        features = features[["Date", "symbol", col]].copy()
        x = base.drop(columns=[col], errors="ignore").merge(features, on=["Date", "symbol"], how="left", validate="many_to_one")
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
        event_rows = x[x[col] > 0].copy()
        if set(event_rows["_row_id"]) - set(base["_row_id"]):
            raise SystemExit(f"Event row-id alignment failure for {event_type}")

        for symbol, b in base.groupby("symbol", sort=True):
            e = event_rows[event_rows["symbol"] == symbol].copy()
            r = evaluate(b, e, args.n_boot)
            r.update({"event_type": event_type, "symbol": symbol})
            reports.append(r)

        st = annual_stability(base, event_rows)
        if not st.empty:
            st["event_type"] = event_type
            stability_parts.append(st)

    report = pd.DataFrame(reports).sort_values(["event_type", "symbol"])
    stability = pd.concat(stability_parts, ignore_index=True) if stability_parts else pd.DataFrame()
    numeric = ["n_base", "n_event_days", "mean_base", "mean_event_gross", "delta_mean", "ci_low", "ci_high"]
    ok = report[report["status"].eq("OK")]
    if not ok.empty and not np.isfinite(ok[numeric].to_numpy(float)).all():
        raise SystemExit("Invalid event presence incremental result")
    if not stability.empty:
        ok_st = stability[stability["status"].eq("OK")]
        cols = ["n_base", "n_event_days", "mean_base", "mean_event_gross", "delta_mean"]
        if not ok_st.empty and not np.isfinite(ok_st[cols].to_numpy(float)).all():
            raise SystemExit("Invalid event presence stability result")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    stability.to_csv(args.out_stability, index=False)
    print("NEWS V3 LAG1 EVENT PRESENCE INCREMENTAL")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 EVENT PRESENCE STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_stability}")
    print("NEWS V3 LAG1 EVENT PRESENCE PASS")


if __name__ == "__main__":
    main()
