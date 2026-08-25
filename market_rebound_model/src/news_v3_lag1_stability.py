from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summary(df: pd.DataFrame, label: str) -> dict:
    x = df[df["next_ret"].notna()]
    return {
        "set": label,
        "n": len(x),
        "mean_next_ret": float(x["next_ret"].mean()) if len(x) else 0.0,
        "median_next_ret": float(x["next_ret"].median()) if len(x) else 0.0,
        "hit_2pct": float((x["next_ret"] >= 0.02).mean()) if len(x) else 0.0,
    }


def build_stability(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["year"] = x["Date"].dt.year

    out = []
    for symbol, s in x.groupby("symbol", sort=True):
        for year, y in s.groupby("year", sort=True):
            y = y[y["next_ret"].notna()].copy()
            if y.empty:
                continue
            base = y[y["v1_top20"]]
            if base.empty:
                continue
            for filt in ("news_top25", "news_top50"):
                selected = y[(y["filter"] == filt) & (y["direction"] == "inverted") & y["eligible"] & y["selected"]]
                mean_base = float(base["next_ret"].mean())
                mean_sel = float(selected["next_ret"].mean()) if not selected.empty else 0.0
                out.append({
                    "symbol": symbol,
                    "year": int(year),
                    "filter": filt,
                    "n_base": len(base),
                    "n_eligible": int(((y["filter"] == filt) & (y["direction"] == "inverted") & y["eligible"]).sum()),
                    "n_selected": len(selected),
                    "mean_base": mean_base,
                    "mean_selected_gross": mean_sel,
                    "delta_mean": mean_sel - mean_base if not selected.empty else 0.0,
                    "status": "OK" if not selected.empty else "NO_SELECTED_CASES",
                })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_lag1_stability.csv")
    args = ap.parse_args()
    rows = pd.read_csv(args.rows_csv)
    required = {"Date", "symbol", "next_ret", "v1_top20", "filter", "direction", "eligible", "selected"}
    missing = required - set(rows.columns)
    if missing:
        raise SystemExit(f"Missing stability columns: {sorted(missing)}")
    report = build_stability(rows)
    if report.empty:
        raise SystemExit("No lag1 stability rows available")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(p, index=False)
    print("NEWS V3 LAG1 STABILITY")
    print(report.to_string(index=False))
    print(f"Saved {p}")
    print("NEWS V3 LAG1 STABILITY PASS")


if __name__ == "__main__":
    main()
