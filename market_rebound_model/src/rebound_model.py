"""Causal rebound model: shared OHLCV feature engineering and walk-forward scoring."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_FEATURES = ["ret", "gap", "range", "close_loc", "recovery_from_low", "vol_ratio", "ret_3", "dd_3", "vol_3", "ret_5", "dd_5", "vol_5", "ret_10", "dd_10", "vol_10", "ret_20", "dd_20", "vol_20", "ret_60", "dd_60", "vol_60"]
NEWS_FEATURES = ["news_sentiment", "news_intensity", "news_relevance", "news_novelty", "news_count"]

def _italian_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).str.replace("%", "", regex=False).str.replace("M", "", regex=False), errors="coerce")

def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    required = ["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
    for c in ["Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna(subset=required).sort_values("Date").reset_index(drop=True)

def engineer_features(d: pd.DataFrame, keep_targets: bool = True) -> pd.DataFrame:
    d = prepare_ohlcv(d)
    d["ret"] = d["Ultimo"].pct_change()
    d["gap"] = d["Apertura"] / d["Ultimo"].shift(1) - 1
    d["range"] = (d["Massimo"] - d["Minimo"]) / d["Ultimo"].shift(1)
    d["close_loc"] = (d["Ultimo"] - d["Minimo"]) / (d["Massimo"] - d["Minimo"]).replace(0, np.nan)
    d["recovery_from_low"] = d["Ultimo"] / d["Minimo"] - 1
    d["vol20"] = d["Vol."].rolling(20).mean()
    d["vol_ratio"] = d["Vol."] / d["vol20"]
    for w in [3, 5, 10, 20, 60]:
        d[f"ret_{w}"] = d["Ultimo"].pct_change(w)
        d[f"dd_{w}"] = d["Ultimo"] / d["Ultimo"].rolling(w).max() - 1
        d[f"vol_{w}"] = d["ret"].rolling(w).std()
    if keep_targets:
        d["next_ret"] = d["Ultimo"].shift(-1) / d["Ultimo"] - 1
        d["next_high"] = d["Massimo"].shift(-1) / d["Ultimo"] - 1
        d["target_2"] = np.where(d["next_ret"].notna(), (d["next_ret"] >= .02).astype(int), np.nan)
        d["target_3"] = np.where(d["next_ret"].notna(), (d["next_ret"] >= .03).astype(int), np.nan)
        d["target_5"] = np.where(d["next_ret"].notna(), (d["next_ret"] >= .05).astype(int), np.nan)
        d["target_high_3"] = np.where(d["next_high"].notna(), (d["next_high"] >= .03).astype(int), np.nan)
    return d

def load_market_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    for c in ["Ultimo", "Apertura", "Massimo", "Minimo", "Var. %", "Vol."]:
        if c not in raw.columns:
            raise ValueError(f"Missing column in historical CSV: {c}")
        raw[c] = _italian_num(raw[c])
    raw = raw.rename(columns={"Data": "Date"})
    return engineer_features(raw)

def load_yahoo_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df)

def walk_forward_score(df: pd.DataFrame, target: str = "target_3", signal_return: float = -.02):
    features = BASE_FEATURES + [c for c in NEWS_FEATURES if c in df.columns]
    x = df.dropna(subset=features + [target]).copy()
    predictions = []
    for year in sorted(x.Date.dt.year.unique()):
        train = x[x.Date.dt.year < year]
        test = x[x.Date.dt.year == year].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        model = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, learning_rate=.05, l2_regularization=2, random_state=42)
        model.fit(train[features], train[target].astype(int))
        test["probability"] = model.predict_proba(test[features])[:, 1]
        test["signal"] = test["ret"] <= signal_return
        predictions.append(test)
    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), features

def report(scored: pd.DataFrame) -> dict:
    s = scored[scored.signal].copy()
    out = {"signals": int(len(s))}
    for p in [.80, .90, .95]:
        if len(s) == 0: continue
        threshold = s.probability.quantile(p)
        z = s[s.probability >= threshold]
        out[f"top_{int((1-p)*100)}pct_count"] = int(len(z))
        out[f"top_{int((1-p)*100)}pct_mean_next_return"] = float(z.next_ret.mean()) if len(z) else None
        out[f"top_{int((1-p)*100)}pct_hit_3pct"] = float((z.next_ret >= .03).mean()) if len(z) else None
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("market_csv")
    ap.add_argument("--out", default="rebound_v1_output.csv")
    args = ap.parse_args()
    df = load_market_csv(args.market_csv)
    scored, features = walk_forward_score(df)
    scored.to_csv(args.out, index=False)
    print(report(scored))
    print("features:", ", ".join(features))
