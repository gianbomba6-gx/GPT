"""Production smoke test for dynamic top10 cutoff.

Does not modify live production code. Rebuilds the same pooled target_3 model
used by live_alert, derives the 90th-percentile probability cutoff from prior
training rebound candidates, and applies it to the latest available row per
symbol. Validates feature completeness, gate behavior, finite values and
current dynamic cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

from rebound_model import BASE_FEATURES, load_yahoo_ohlcv
from live_alert import add_market_regime, REGIME_FEATURES

FEATURES = BASE_FEATURES + REGIME_FEATURES
TARGET = "target_3"
DOWN_THRESHOLD = -0.02
TOP10_Q = 0.90
SEED = 42


def fetch(symbol: str, period: str = "max") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close": "Ultimo", "Open": "Apertura", "High": "Massimo", "Low": "Minimo", "Volume": "Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize().astype("datetime64[ns]")
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def fit(train: pd.DataFrame):
    tr = train.dropna(subset=FEATURES + [TARGET]).copy()
    if len(tr) < 500 or tr[TARGET].nunique() < 2:
        raise RuntimeError("Insufficient pooled training history")
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=2, random_state=SEED
    )
    model.fit(tr[FEATURES], tr[TARGET].astype(int))
    return model, tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_dynamic_top10_live_smoke.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "tickers.json").read_text())
    frames: dict[str, pd.DataFrame] = {}
    benchmark = None
    benchmark_symbol = None

    for item in config["tickers"]:
        d = load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"] == "benchmark":
            benchmark = d
            benchmark_symbol = item["symbol"]
        else:
            frames[item["symbol"]] = d

    if benchmark is None:
        raise RuntimeError(f"Missing benchmark {benchmark_symbol}")

    frames = add_market_regime(frames, benchmark)
    all_data = pd.concat([d.assign(symbol=s) for s, d in frames.items()], ignore_index=True)
    all_data["Date"] = pd.to_datetime(all_data["Date"]).dt.normalize()
    latest_year = int(all_data["Date"].dt.year.max())
    train = all_data[all_data["Date"].dt.year < latest_year].copy()
    model, trainable = fit(train)

    candidates = trainable[trainable["ret"] <= DOWN_THRESHOLD].copy()
    if len(candidates) < 20:
        raise RuntimeError("Insufficient training rebound candidates for dynamic cutoff")
    train_p = model.predict_proba(candidates[FEATURES])[:, 1]
    cutoff = float(np.quantile(train_p, TOP10_Q))
    if not (0.0 < cutoff < 1.0):
        raise RuntimeError(f"Invalid dynamic cutoff: {cutoff}")

    rows = []
    for symbol, d in frames.items():
        latest = d.sort_values("Date").iloc[-1].copy()
        if latest[FEATURES].isna().any():
            raise RuntimeError(f"Missing live features for {symbol}")
        probability = float(model.predict_proba(pd.DataFrame([latest])[FEATURES])[:, 1][0])
        ret = float(latest["ret"])
        gate = ret <= DOWN_THRESHOLD
        qualifies = bool(gate and probability >= cutoff)
        rows.append({
            "symbol": symbol,
            "date": pd.Timestamp(latest["Date"]).date().isoformat(),
            "ret": ret,
            "probability": probability,
            "dynamic_cutoff": cutoff,
            "gate_down_2pct": gate,
            "qualifies_dynamic_top10": qualifies,
        })

    report = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    numeric = ["ret", "probability", "dynamic_cutoff"]
    if not np.isfinite(report[numeric].to_numpy(float)).all():
        raise SystemExit("Invalid dynamic live smoke numeric result")
    if report["dynamic_cutoff"].nunique() != 1:
        raise SystemExit("Cutoff is not unique across live symbols")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print("V1 DYNAMIC TOP10 LIVE SMOKE")
    print(f"latest_year={latest_year} n_train_rows={len(trainable)} n_train_candidates={len(candidates)}")
    print(f"dynamic_cutoff={cutoff:.6f}")
    print(report.to_string(index=False))
    print(f"Saved {out}")
    print("V1 DYNAMIC TOP10 LIVE SMOKE PASS")


if __name__ == "__main__":
    main()
