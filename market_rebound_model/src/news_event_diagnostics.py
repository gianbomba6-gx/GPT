"""Out-of-sample diagnostics by news event family for rebound research."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EVENTS = (
    "earnings",
    "guidance",
    "analyst",
    "regulatory",
    "ma",
    "product",
    "macro",
    "other",
)

REQUIRED_BASE = {"symbol", "Date", "next_ret", "next_high"}


def stats(x: pd.DataFrame, label: str, symbol: str, event: str, condition: str) -> dict:
    x = x[x["next_ret"].notna()].copy()
    n = len(x)
    return {
        "symbol": symbol,
        "event": event,
        "condition": condition,
        "n": n,
        "mean_next_ret": float(x["next_ret"].mean()) if n else None,
        "median_next_ret": float(x["next_ret"].median()) if n else None,
        "hit_2pct": float((x["next_ret"] >= 0.02).mean()) if n else None,
        "hit_3pct": float((x["next_ret"] >= 0.03).mean()) if n else None,
        "hit_5pct": float((x["next_ret"] >= 0.05).mean()) if n else None,
        "mean_next_high": float(x["next_high"].mean()) if n else None,
    }


def build_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_BASE - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = data.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    for col in ["next_ret", "next_high"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    share_cols = {event: f"event_{event}_share" for event in EVENTS if f"event_{event}_share" in x.columns}
    if not share_cols:
        raise ValueError("No event share columns found in OOS predictions")
    for col in share_cols.values():
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    rows: list[dict] = []
    for symbol, s in x.groupby("symbol", sort=True):
        rows.append(stats(s, "all", symbol, "all", "all_candidates"))
        for event, col in share_cols.items():
            present = s[s[col] > 0]
            rows.append(stats(present, "present", symbol, event, "event_present"))

        share_matrix = s[list(share_cols.values())].copy()
        share_matrix.columns = list(share_cols.keys())
        primary = share_matrix.idxmax(axis=1)
        max_share = share_matrix.max(axis=1)
        for event in share_cols:
            dominant = s[(primary == event) & (max_share > 0)]
            rows.append(stats(dominant, "dominant", symbol, event, "dominant"))

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oos_predictions")
    ap.add_argument("--out", default="results/news_v3_event_diagnostics.csv")
    args = ap.parse_args()

    src = Path(args.oos_predictions)
    if not src.exists():
        raise FileNotFoundError(src)
    data = pd.read_csv(src)
    out = build_diagnostics(data)
    out = out.sort_values(["symbol", "condition", "event"]).reset_index(drop=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
