"""Operational confirmation: current fixed 70% live threshold vs training-derived top10% cutoff.

Keeps target_3, production V1 features, pooled model, -2% gate and expanding
annual walk-forward unchanged. The only change is how the probability threshold
is converted into an operational signal: fixed 70% vs top10% of prior training
rebound candidates.
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
FIXED = 0.70
TOP10 = 0.90
COST = 0.002
SEED = 42


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize()
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]


def fit(train: pd.DataFrame):
    tr = train.dropna(subset=FEATURES + ["target_3"]).copy()
    if len(tr) < 500 or tr["target_3"].nunique() < 2:
        return None, tr
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=2, random_state=SEED
    )
    model.fit(tr[FEATURES], tr["target_3"].astype(int))
    return model, tr


def top10_cutoff(train: pd.DataFrame) -> float:
    model, tr = fit(train)
    if model is None:
        return float("nan")
    candidates = tr[tr["ret"] <= -0.02].copy()
    if len(candidates) < 20:
        return float("nan")
    p = model.predict_proba(candidates[FEATURES])[:, 1]
    return float(pd.Series(p).quantile(TOP10))


def build(data: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        model, _ = fit(train)
        if model is None:
            continue
        cutoff = top10_cutoff(train)
        if not np.isfinite(cutoff):
            continue
        te = test.dropna(subset=FEATURES).copy()
        if te.empty:
            continue
        te["score"] = model.predict_proba(te[FEATURES])[:, 1]
        te["baseline_signal"] = te["ret"] <= -0.02
        te["selected_fixed70"] = te["baseline_signal"] & (te["score"] >= FIXED)
        te["selected_top10"] = te["baseline_signal"] & (te["score"] >= cutoff)
        te["year"] = int(year)
        te["top10_cutoff"] = cutoff
        pieces.append(te[["Date","symbol","ret","next_ret","target_3","baseline_signal","score","selected_fixed70","selected_top10","year","top10_cutoff"]])
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol","Date"]).reset_index(drop=True)


def bootstrap_mean(values: np.ndarray, mask: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values, mask = values[valid], mask[valid]
    if len(values) == 0 or mask.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[mask].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    s = values[idx]
    m = mask[idx]
    c = m.sum(axis=1)
    ok = c > 0
    out = np.full(n_boot, np.nan)
    out[ok] = (s * m).sum(axis=1)[ok] / c[ok]
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return observed, float(lo), float(hi)


def bootstrap_diff(values: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values, a, b = values[valid], a[valid], b[valid]
    if len(values) == 0 or a.sum() == 0 or b.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[a].mean() - values[b].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    s = values[idx]; ma = a[idx]; mb = b[idx]
    ca, cb = ma.sum(axis=1), mb.sum(axis=1)
    ok = (ca > 0) & (cb > 0)
    out = np.full(n_boot, np.nan)
    out[ok] = (s * ma).sum(axis=1)[ok] / ca[ok] - (s * mb).sum(axis=1)[ok] / cb[ok]
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summarize(df: pd.DataFrame, n_boot: int):
    rows = []
    symbols = ["ALL"] + sorted(df["symbol"].dropna().unique().tolist())
    for symbol in symbols:
        x = df if symbol == "ALL" else df[df.symbol == symbol]
        x = x[x.baseline_signal & x.next_ret.notna()].copy()
        if x.empty:
            continue
        vals = x.next_ret.to_numpy(float)
        hit = x.target_3.astype(int).to_numpy()
        a = x.selected_fixed70.fillna(False).astype(bool).to_numpy()
        b = x.selected_top10.fillna(False).astype(bool).to_numpy()
        for name, mask in (("fixed70", a), ("top10", b)):
            mean, lo, hi = bootstrap_mean(vals, mask, n_boot)
            n = int(mask.sum())
            hit_rate = float(hit[mask].mean()) if n else np.nan
            rows.append({"symbol":symbol,"strategy":name,"n_baseline":len(x),"n_selected":n,"selection_rate":n/len(x),"mean_selected_gross":mean,"mean_selected_net":mean-COST if n else np.nan,"hit3":hit_rate,"ci_low":lo,"ci_high":hi,"cost_bps":20.0,"status":"OK" if n else "NO_SELECTED_CASES"})
        diff, dlo, dhi = bootstrap_diff(vals, b, a, n_boot)
        rows.append({"symbol":symbol,"strategy":"top10_minus_fixed70","n_baseline":len(x),"n_selected":int(b.sum()),"selection_rate":float(b.sum()/len(x)),"mean_selected_gross":diff,"mean_selected_net":diff,"hit3":float(hit[b].mean()-hit[a].mean()) if a.sum() and b.sum() else np.nan,"ci_low":dlo,"ci_high":dhi,"cost_bps":0.0,"status":"OK" if a.sum() and b.sum() else "NO_SELECTED_CASES"})
    report = pd.DataFrame(rows)

    stability=[]
    for (symbol,year),x in df.groupby(["symbol","year"],sort=True):
        x=x[x.baseline_signal & x.next_ret.notna()].copy()
        if x.empty: continue
        for name,col in (("fixed70","selected_fixed70"),("top10","selected_top10")):
            m=x[col].fillna(False).astype(bool)
            n=int(m.sum())
            stability.append({"symbol":symbol,"year":int(year),"strategy":name,"n_baseline":len(x),"n_selected":n,"mean_selected_gross":float(x.loc[m,"next_ret"].mean()) if n else np.nan,"hit3":float(x.loc[m,"target_3"].mean()) if n else np.nan})
    return report,pd.DataFrame(stability)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="results/v1_top10_operational_confirmation.csv"); ap.add_argument("--out-stability",default="results/v1_top10_operational_stability.csv"); ap.add_argument("--n-boot",type=int,default=10000); args=ap.parse_args()
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
    scored=build(data)
    if scored.empty: raise RuntimeError("No OOS operational rows generated")
    report,stability=summarize(scored,args.n_boot)
    nums=["n_baseline","n_selected","selection_rate","mean_selected_gross","mean_selected_net","hit3","ci_low","ci_high","cost_bps"]
    ok=report[report.status.eq("OK")]
    if not ok.empty and not np.isfinite(ok[nums].to_numpy(float)).all(): raise SystemExit("Invalid operational confirmation result")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    report.to_csv(root/args.out,index=False); stability.to_csv(root/args.out_stability,index=False)
    print("V1 TOP10 OPERATIONAL CONFIRMATION"); print(report.to_string(index=False)); print("V1 TOP10 OPERATIONAL STABILITY"); print(stability.to_string(index=False)); print(f"Saved {root/args.out}"); print(f"Saved {root/args.out_stability}"); print("V1 TOP10 OPERATIONAL CONFIRMATION PASS")

if __name__=="__main__": main()
