"""Confirmatory OOS comparison of V1 top10% vs top20% selection.

Keeps target_3, production V1 features, pooled model, -2% gate and
expanding annual walk-forward training unchanged. Only the training-derived
selection fraction differs. Adds direct bootstrap comparison 10% vs 20%.
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
FRACTIONS = (0.10, 0.20)
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
    return float(pd.Series(p, index=tr.index)[base].quantile(1.0 - fraction))


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        test = test.dropna(subset=["Date", "symbol"]).copy()
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
            pct = int(frac * 100)
            te[f"selected_{pct}"] = te["baseline_signal"] & (te["score"] >= thr)
        parts.append(te[["Date","symbol","next_ret","baseline_signal","score","year","selected_10","selected_20"]].copy())
    return pd.concat(parts, ignore_index=True).sort_values(["symbol","Date"]).reset_index(drop=True) if parts else pd.DataFrame()


def bootstrap_mean(values: np.ndarray, selected: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values = values[valid]
    selected = selected[valid]
    if len(values) == 0 or selected.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[selected].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    mask = selected[idx]
    counts = mask.sum(axis=1)
    ok = counts > 0
    means = np.full(n_boot, np.nan)
    means[ok] = (sample * mask).sum(axis=1)[ok] / counts[ok]
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return observed, float(lo), float(hi)


def bootstrap_direct_difference(values: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values, a, b = values[valid], a[valid], b[valid]
    if len(values) == 0 or a.sum() == 0 or b.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[a].mean() - values[b].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    ma, mb = a[idx], b[idx]
    ca, cb = ma.sum(axis=1), mb.sum(axis=1)
    ok = (ca > 0) & (cb > 0)
    diffs = np.full(n_boot, np.nan)
    diffs[ok] = (sample * ma).sum(axis=1)[ok] / ca[ok] - (sample * mb).sum(axis=1)[ok] / cb[ok]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summarize(df: pd.DataFrame, n_boot: int):
    rows = []
    for symbol in ["ALL"] + sorted(df.symbol.dropna().unique().tolist()):
        x = df if symbol == "ALL" else df[df.symbol == symbol]
        x = x[x.baseline_signal & x.next_ret.notna()].copy()
        if x.empty:
            continue
        vals = x.next_ret.to_numpy(float)
        a = x.selected_10.fillna(False).astype(bool).to_numpy()
        b = x.selected_20.fillna(False).astype(bool).to_numpy()
        for pct, sel in ((10, a), (20, b)):
            mean, lo, hi = bootstrap_mean(vals, sel, n_boot)
            n = int(sel.sum())
            rows.append({
                "symbol": symbol, "comparison": f"top{pct}", "n_baseline": len(x), "n_selected": n,
                "mean_selected_gross": mean, "mean_selected_net": mean - COST if n else np.nan,
                "ci_low": lo, "ci_high": hi, "cost_bps": 20.0,
                "status": "OK" if n else "NO_SELECTED_CASES",
            })
        diff, dlo, dhi = bootstrap_direct_difference(vals, a, b, n_boot)
        rows.append({
            "symbol": symbol, "comparison": "top10_minus_top20", "n_baseline": len(x),
            "n_selected": int(a.sum()), "n_selected_top20": int(b.sum()),
            "mean_selected_gross": diff, "mean_selected_net": diff,
            "ci_low": dlo, "ci_high": dhi, "cost_bps": 0.0,
            "status": "OK" if a.sum() and b.sum() else "NO_SELECTED_CASES",
        })
    report = pd.DataFrame(rows)

    stability = []
    for (symbol, year), x in df.groupby(["symbol","year"], sort=True):
        x = x[x.baseline_signal & x.next_ret.notna()].copy()
        if x.empty:
            continue
        mbase = float(x.next_ret.mean())
        m10 = x.loc[x.selected_10.fillna(False), "next_ret"]
        m20 = x.loc[x.selected_20.fillna(False), "next_ret"]
        stability.extend([
            {"symbol":symbol,"year":int(year),"comparison":"top10","n_baseline":len(x),"n_selected":len(m10),"mean_baseline":mbase,"mean_selected_gross":float(m10.mean()) if len(m10) else np.nan,"delta_mean":float(m10.mean()-mbase) if len(m10) else np.nan},
            {"symbol":symbol,"year":int(year),"comparison":"top20","n_baseline":len(x),"n_selected":len(m20),"mean_baseline":mbase,"mean_selected_gross":float(m20.mean()) if len(m20) else np.nan,"delta_mean":float(m20.mean()-mbase) if len(m20) else np.nan},
            {"symbol":symbol,"year":int(year),"comparison":"top10_minus_top20","n_baseline":len(x),"n_selected":min(len(m10),len(m20)),"mean_baseline":mbase,"mean_selected_gross":float(m10.mean()-m20.mean()) if len(m10) and len(m20) else np.nan,"delta_mean":float(m10.mean()-m20.mean()) if len(m10) and len(m20) else np.nan},
        ])
    return report, pd.DataFrame(stability)


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo","Open":"Apertura","High":"Massimo","Low":"Minimo","Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize()
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="results/v1_top10_top20_confirmation.csv"); ap.add_argument("--out-stability",default="results/v1_top10_top20_confirmation_stability.csv"); ap.add_argument("--n-boot",type=int,default=10000); args=ap.parse_args()
    if args.n_boot < 1000: raise SystemExit("Invalid bootstrap count")
    root=Path(__file__).resolve().parents[1]; config=json.loads((root/"config"/"tickers.json").read_text())
    frames={}; benchmark=None
    for item in config["tickers"]:
        d=load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"]=="benchmark": benchmark=d
        else: frames[item["symbol"]]=d
    if benchmark is None: raise RuntimeError("Missing benchmark")
    frames=add_market_regime(frames,benchmark)
    data=pd.concat([d.assign(symbol=s) for s,d in frames.items()],ignore_index=True)
    scored=walk_forward(data)
    if scored.empty: raise RuntimeError("No OOS confirmation rows generated")
    report,stability=summarize(scored,args.n_boot)
    report.to_csv(root/args.out,index=False); stability.to_csv(root/args.out_stability,index=False)
    print("V1 TOP10 VS TOP20 CONFIRMATION"); print(report.to_string(index=False))
    print("V1 TOP10 VS TOP20 STABILITY"); print(stability.to_string(index=False))
    print(f"Saved {root/args.out}"); print(f"Saved {root/args.out_stability}"); print("V1 TOP10 VS TOP20 CONFIRMATION PASS")

if __name__=="__main__": main()
