from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .news_event_secondary_ranking import _event_features_lagged
    from .news_v3_incremental_value import bootstrap_delta
except ImportError:
    from news_event_secondary_ranking import _event_features_lagged
    from news_v3_incremental_value import bootstrap_delta


def bucket_count(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def evaluate_bucket(base: pd.DataFrame, bucket_rows: pd.DataFrame, n_boot: int) -> dict:
    base = base[base["next_ret"].notna()].copy()
    bucket_rows = bucket_rows[bucket_rows["next_ret"].notna()].copy()
    if bucket_rows.empty:
        return {
            "n_base": len(base),
            "n_bucket": 0,
            "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0,
            "mean_bucket": 0.0,
            "delta_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "status": "NO_BUCKET_CASES",
        }
    base_ids = base["_row_id"].to_numpy()
    bucket_ids = set(bucket_rows["_row_id"].to_numpy())
    mask = np.array([rid in bucket_ids for rid in base_ids], dtype=bool)
    if not mask.any():
        return {
            "n_base": len(base),
            "n_bucket": len(bucket_rows),
            "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0,
            "mean_bucket": np.nan,
            "delta_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "status": "INVALID_BUCKET_ID_ALIGNMENT",
        }
    values = base["next_ret"].to_numpy(float)
    delta, lo, hi = bootstrap_delta(values, mask, n_boot, 42)
    return {
        "n_base": len(base),
        "n_bucket": int(mask.sum()),
        "mean_base": float(values.mean()),
        "mean_bucket": float(values[mask].mean()),
        "delta_mean": float(delta),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "status": "OK",
    }


def annual_stability(base: pd.DataFrame) -> pd.DataFrame:
    out = []
    for symbol, b in base.groupby("symbol", sort=True):
        b = b[b["next_ret"].notna()].copy()
        b["year"] = pd.to_datetime(b["Date"]).dt.year
        for year, by in b.groupby("year", sort=True):
            mean_base = float(by["next_ret"].mean()) if len(by) else 0.0
            for bucket in ("0", "1", "2", "3+"):
                byb = by[by["news_bucket"] == bucket]
                out.append({
                    "symbol": symbol,
                    "year": int(year),
                    "bucket": bucket,
                    "n_base": len(by),
                    "n_bucket": len(byb),
                    "mean_base": mean_base,
                    "mean_bucket": float(byb["next_ret"].mean()) if len(byb) else 0.0,
                    "delta_mean": float(byb["next_ret"].mean() - mean_base) if len(byb) else 0.0,
                    "status": "OK" if len(byb) else "NO_BUCKET_CASES",
                })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_news_intensity.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_news_intensity_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 100:
        raise SystemExit("Invalid intensity parameters")

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce").dt.normalize()
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)
    base = rows[rows["v1_top20"]].dropna(subset=["Date"]).copy()
    base = base.sort_values(["Date", "symbol"]).reset_index(drop=True)
    base["_row_id"] = np.arange(len(base))

    events = _event_features_lagged(raw, 1, base[["Date", "symbol"]]).copy()
    if "news_count" not in events.columns:
        raise SystemExit("Missing news_count from lagged event features")
    events = events[["Date", "symbol", "news_count"]]
    x = base.drop(columns=["news_count"], errors="ignore").merge(
        events, on=["Date", "symbol"], how="left", validate="many_to_one"
    )
    x["news_count"] = pd.to_numeric(x["news_count"], errors="coerce").fillna(0.0)
    x["news_count"] = x["news_count"].astype(int)
    x["news_bucket"] = x["news_count"].map(bucket_count)

    reports = []
    for symbol, b in x.groupby("symbol", sort=True):
        for bucket in ("0", "1", "2", "3+"):
            r = evaluate_bucket(b, b[b["news_bucket"] == bucket], args.n_boot)
            r.update({"symbol": symbol, "bucket": bucket})
            reports.append(r)

    report = pd.DataFrame(reports).sort_values(["symbol", "bucket"])
    stability = annual_stability(x)

    numeric = ["n_base", "n_bucket", "mean_base", "mean_bucket", "delta_mean", "ci_low", "ci_high"]
    ok = report[report["status"].eq("OK")]
    if not ok.empty and not np.isfinite(ok[numeric].to_numpy(float)).all():
        raise SystemExit("Invalid news intensity result")
    if not stability.empty:
        ok_st = stability[stability["status"].eq("OK")]
        cols = ["n_base", "n_bucket", "mean_base", "mean_bucket", "delta_mean"]
        if not ok_st.empty and not np.isfinite(ok_st[cols].to_numpy(float)).all():
            raise SystemExit("Invalid news intensity stability result")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    stability.to_csv(args.out_stability, index=False)
    print("NEWS V3 LAG1 NEWS INTENSITY")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 NEWS INTENSITY STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_stability}")
    print("NEWS V3 LAG1 NEWS INTENSITY PASS")


if __name__ == "__main__":
    main()
