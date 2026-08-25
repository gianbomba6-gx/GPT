from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .news_event_classifier import add_event_features
    from .news_v3_incremental_value import bootstrap_delta
except ImportError:
    from news_event_classifier import add_event_features
    from news_v3_incremental_value import bootstrap_delta


def sentiment_bucket(news_count: int, polarity: float) -> str:
    if news_count <= 0:
        return "no_news"
    if polarity < 0:
        return "negative"
    if polarity > 0:
        return "positive"
    return "neutral"


def map_lag1_daily_sentiment(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    required = {"published_at", "symbol", "candidate_day"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")

    x = raw.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x["candidate_day"] = pd.to_datetime(x["candidate_day"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["published_at", "candidate_day", "symbol"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["Date", "symbol", "news_count", "event_polarity"])

    x = add_event_features(x)
    cal = calendar.copy()
    cal["symbol"] = cal["symbol"].astype(str).str.upper().str.strip()
    cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce").dt.normalize()
    cal = cal.dropna(subset=["Date", "symbol"]).drop_duplicates(["Date", "symbol"])

    target_maps: dict[str, np.ndarray] = {}
    for symbol, s in cal.groupby("symbol", sort=False):
        target_maps[str(symbol)] = np.sort(s["Date"].unique())

    def map_source_day(symbol: str, source_day: pd.Timestamp) -> pd.Timestamp | pd.NaT:
        targets = target_maps.get(str(symbol))
        if targets is None or len(targets) == 0:
            return pd.NaT
        pos = int(np.searchsorted(targets, np.datetime64(source_day), side="right"))
        if pos >= len(targets):
            return pd.NaT
        return pd.Timestamp(targets[pos])

    x["feature_day"] = [
        map_source_day(symbol, day)
        for symbol, day in zip(x["symbol"], x["candidate_day"])
    ]
    x = x.dropna(subset=["feature_day"]).copy()

    daily = (
        x.groupby(["feature_day", "symbol"], as_index=False)
        .agg(news_count=("classification_text", "size"), event_polarity=("event_polarity", "mean"))
        .rename(columns={"feature_day": "Date"})
    )
    return daily


def evaluate(base: pd.DataFrame, bucket_rows: pd.DataFrame, n_boot: int) -> dict:
    base = base[base["next_ret"].notna()].copy()
    bucket_rows = bucket_rows[bucket_rows["next_ret"].notna()].copy()
    if bucket_rows.empty:
        return {
            "n_base": len(base), "n_bucket": 0,
            "mean_base": float(base["next_ret"].mean()) if len(base) else 0.0,
            "mean_bucket": 0.0, "delta_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "status": "NO_BUCKET_CASES",
        }
    ids = set(bucket_rows["_row_id"].to_numpy())
    mask = np.array([rid in ids for rid in base["_row_id"].to_numpy()], dtype=bool)
    if not mask.any():
        return {
            "n_base": len(base), "n_bucket": len(bucket_rows),
            "mean_base": float(base["next_ret"].mean()), "mean_bucket": np.nan,
            "delta_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "status": "INVALID_SENTIMENT_ID_ALIGNMENT",
        }
    values = base["next_ret"].to_numpy(float)
    delta, lo, hi = bootstrap_delta(values, mask, n_boot, 42)
    return {
        "n_base": len(base), "n_bucket": int(mask.sum()),
        "mean_base": float(values.mean()), "mean_bucket": float(values[mask].mean()),
        "delta_mean": float(delta), "ci_low": float(lo), "ci_high": float(hi),
        "status": "OK",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_lag1_news_sentiment.csv")
    ap.add_argument("--out-stability", default="results/news_v3_lag1_news_sentiment_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 100:
        raise SystemExit("Invalid sentiment parameters")

    rows = pd.read_csv(args.rows_csv)
    raw = pd.read_csv(args.raw_gkg)
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce").dt.normalize()
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    rows["next_ret"] = pd.to_numeric(rows["next_ret"], errors="coerce")
    rows["v1_top20"] = rows["v1_top20"].fillna(False).astype(bool)
    base = rows[rows["v1_top20"]].dropna(subset=["Date"]).copy()
    base = base.sort_values(["Date", "symbol"]).reset_index(drop=True)
    base["_row_id"] = np.arange(len(base))

    sentiment = map_lag1_daily_sentiment(raw, base[["Date", "symbol"]])
    base_for_merge = base.drop(columns=["news_count", "event_polarity"], errors="ignore")
    if not sentiment.empty:
        x = base_for_merge.merge(sentiment, on=["Date", "symbol"], how="left", validate="many_to_one")
    else:
        x = base_for_merge.copy()
        x["news_count"] = 0
        x["event_polarity"] = 0.0
    if "event_polarity" not in x.columns or "news_count" not in x.columns:
        raise SystemExit("Lag1 sentiment feature merge missing required columns")
    x["news_count"] = pd.to_numeric(x["news_count"], errors="coerce").fillna(0).astype(int)
    x["event_polarity"] = pd.to_numeric(x["event_polarity"], errors="coerce").fillna(0.0)
    x["sentiment_bucket"] = [sentiment_bucket(n, p) for n, p in zip(x["news_count"], x["event_polarity"])]

    reports = []
    for symbol, b in x.groupby("symbol", sort=True):
        for bucket in ("no_news", "negative", "neutral", "positive"):
            r = evaluate(b, b[b["sentiment_bucket"] == bucket], args.n_boot)
            r.update({"symbol": symbol, "bucket": bucket})
            reports.append(r)

    report = pd.DataFrame(reports).sort_values(["symbol", "bucket"])

    stability_rows = []
    for (symbol, year), by in x.assign(year=pd.to_datetime(x["Date"]).dt.year).groupby(["symbol", "year"], sort=True):
        mean_base = float(by["next_ret"].dropna().mean()) if by["next_ret"].notna().any() else 0.0
        for bucket in ("no_news", "negative", "neutral", "positive"):
            byb = by[(by["sentiment_bucket"] == bucket) & by["next_ret"].notna()]
            stability_rows.append({"symbol": symbol, "year": int(year), "bucket": bucket, "n_base": int(by["next_ret"].notna().sum()), "n_bucket": len(byb), "mean_base": mean_base, "mean_bucket": float(byb["next_ret"].mean()) if len(byb) else 0.0, "delta_mean": float(byb["next_ret"].mean() - mean_base) if len(byb) else 0.0, "status": "OK" if len(byb) else "NO_BUCKET_CASES"})
    stability = pd.DataFrame(stability_rows)

    numeric = ["n_base", "n_bucket", "mean_base", "mean_bucket", "delta_mean", "ci_low", "ci_high"]
    ok = report[report["status"].eq("OK")]
    if not ok.empty and not np.isfinite(ok[numeric].to_numpy(float)).all():
        raise SystemExit("Invalid news sentiment result")
    if not stability.empty:
        ok_st = stability[stability["status"].eq("OK")]
        cols = ["n_base", "n_bucket", "mean_base", "mean_bucket", "delta_mean"]
        if not ok_st.empty and not np.isfinite(ok_st[cols].to_numpy(float)).all():
            raise SystemExit("Invalid news sentiment stability result")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    stability.to_csv(args.out_stability, index=False)
    print("NEWS V3 LAG1 NEWS SENTIMENT")
    print(report.to_string(index=False))
    print("NEWS V3 LAG1 NEWS SENTIMENT STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_stability}")
    print("NEWS V3 LAG1 NEWS SENTIMENT PASS")

if __name__ == "__main__":
    main()
