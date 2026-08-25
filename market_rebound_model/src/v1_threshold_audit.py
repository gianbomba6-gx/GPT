"""Walk-forward audit of V1 training-derived selection fractions.

Keeps the production V1 target_3 model and -2% rebound gate unchanged.
Only the training-derived top selection fraction changes: 10%, 20%, 30%, 40%.
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
FRACTIONS = (0.10, 0.20, 0.30, 0.40)
COST = 0.002
SEED = 42


def fit_model(train: pd.DataFrame):
    tr = train.dropna(subset=FEATURES + ["target_3"]).copy()
    if len(tr) < 500 or tr["target_3"].nunique() < 2:
        return None, tr
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=2, random_state=SEED
    )
    model.fit(tr[FEATURES], tr["target_3"].astype(int))
    return model, tr


def training_threshold(train: pd.DataFrame, fraction: float) -> float:
    model, tr = fit_model(train)
    if model is None:
        return float("nan")
    base = tr["ret"] <= -0.02
    if int(base.sum()) < 20:
        return float("nan")
    p = model.predict_proba(tr[FEATURES])[:, 1]
    # Top fraction of training rebound candidates.
    return float(pd.Series(p, index=tr.index)[base].quantile(1.0 - fraction))


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        test = test.copy()
        test["baseline_signal"] = test["ret"] <= -0.02
        test["year"] = int(year)
        te = test.dropna(subset=FEATURES).copy()
        if te.empty:
            continue
        model, _ = fit_model(train)
        if model is None:
            continue
        te["score"] = model.predict_proba(te[FEATURES])[:, 1]
        for frac in FRACTIONS:
            thr = training_threshold(train, frac)
            if not np.isfinite(thr):
                continue
            te[f"selected_{int(frac*100)}"] = te["baseline_signal"] & (te["score"] >= thr)
            te[f"fraction_{int(frac*100)}"] = frac
        cols = ["Date", "symbol", "next_ret", "baseline_signal", "score", "year"] + [f"selected_{int(f*100)}" for f in FRACTIONS]
        parts.append(te[cols].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bootstrap_delta(values: np.ndarray, selected: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values = values[valid]
    selected = selected[valid]
    if len(values) == 0 or selected.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    mask = selected[idx]
    counts = mask.sum(axis=1)
    ok = counts > 0
    deltas = np.full(n_boot, np.nan)
    deltas[ok] = (sample * mask).sum(axis=1)[ok] / counts[ok] - sample.mean(axis=1)[ok]
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summarize(scored: pd.DataFrame, n_boot: int):
    summary = []
    for symbol in ["ALL"] + sorted(scored["symbol"].unique()):
        x = scored if symbol == "ALL" else scored[scored["symbol"] == symbol]
        x = x[x["baseline_signal"] & x["next_ret"].notna()].copy()
        if x.empty:
            continue
        for pct in (10, 20, 30, 40):
            sel = x[f"selected_{pct}"].fillna(False).astype(bool).to_numpy()
            vals = x["next_ret"].to_numpy(float)
            delta, lo, hi = bootstrap_delta(vals, sel, n_boot)
            n = int(sel.sum())
            gross = float(vals[sel].mean()) if n else np.nan
            summary.append({
                "symbol": symbol,
                "selection_pct": pct,
                "n_baseline": len(x),
                "n_selected": n,
                "mean_baseline": float(vals.mean()),
                "mean_selected_gross": gross,
                "mean_selected_net": gross - COST if n else np.nan,
                "delta_mean_gross": delta,
                "ci_low": lo,
                "ci_high": hi,
                "delta_mean_net": delta - COST if np.isfinite(delta) else np.nan,
                "cost_bps": 20.0,
                "status": "OK" if n else "NO_SELECTED_CASES",
            })
    stability = []
    for (symbol, year), x in scored.groupby(["symbol", "year"], sort=True):
        x = x[x["baseline_signal"] & x["next_ret"].notna()].copy()
        if x.empty:
            continue
        base = float(x["next_ret"].mean())
        for pct in (10, 20, 30, 40):
            sel = x[f"selected_{pct}"].fillna(False).astype(bool)
            n = int(sel.sum())
            mean_sel = float(x.loc[sel, "next_ret"].mean()) if n else np.nan
            stability.append({
                "symbol": symbol, "year": int(year), "selection_pct": pct,
                "n_baseline": len(x), "n_selected": n,
                "mean_baseline": base, "mean_selected_gross": mean_sel,
                "delta_mean": mean_sel - base if n else np.nan,
                "status": "OK" if n else "NO_SELECTED_CASES",
            })
    return pd.DataFrame(summary), pd.DataFrame(stability)


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize()
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_threshold_audit.csv")
    ap.add_argument("--out-stability", default="results/v1_threshold_audit_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise SystemExit("Invalid bootstrap count")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "tickers.json").read_text())
    frames = {}; benchmark = None
    for item in config["tickers"]:
        d = load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"] == "benchmark": benchmark = d
        else: frames[item["symbol"]] = d
    if benchmark is None: raise RuntimeError("Missing benchmark")
    frames = add_market_regime(frames, benchmark)
    data = pd.concat([d.assign(symbol=s) for s,d in frames.items()], ignore_index=True)
    scored = walk_forward(data)
    if scored.empty: raise RuntimeError("No OOS threshold rows generated")
    report, stability = summarize(scored, args.n_boot)
    if report.empty: raise RuntimeError("No threshold summary generated")
    ok = report[report.status.eq("OK")]
    numeric = ["n_baseline","n_selected","mean_baseline","mean_selected_gross","mean_selected_net","delta_mean_gross","ci_low","ci_high","delta_mean_net","cost_bps"]
    if not ok.empty and not np.isfinite(ok[numeric].to_numpy(float)).all(): raise SystemExit("Invalid threshold audit result")
    report.to_csv(root / args.out, index=False)
    stability.to_csv(root / args.out_stability, index=False)
    print("V1 THRESHOLD AUDIT"); print(report.to_string(index=False))
    print("V1 THRESHOLD STABILITY"); print(stability.to_string(index=False))
    print(f"Saved {root / args.out}"); print(f"Saved {root / args.out_stability}"); print("V1 THRESHOLD AUDIT PASS")

if __name__ == "__main__": main()
