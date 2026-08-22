"""V1 causal rebound model: feature engineering, walk-forward scoring, and news merge."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_FEATURES = ["ret", "gap", "range", "close_loc", "recovery_from_low", "vol_ratio", "ret_3", "dd_3", "vol_3", "ret_5", "dd_5", "vol_5", "ret_10", "dd_10", "vol_10", "ret_20", "dd_20", "vol_20", "ret_60", "dd_60", "vol_60"]
NEWS_FEATURES = ["news_sentiment", "news_intensity", "news_relevance", "news_novelty", "news_count"]

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).str.replace("%", "", regex=False).str.replace("M", "", regex=False), errors="coerce")

def load_market_csv(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    for c in ["Ultimo", "Apertura", "Massimo", "Minimo", "Var. %", "Vol."]:
        d[c] = _num(d[c])
    d["Date"] = pd.to_datetime(d["Data"], dayfirst=True)
    d = d.sort_values("Date").reset_index(drop=True)
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
    d["next_ret"] = d["Ultimo"].shift(-1) / d["Ultimo"] - 1
    d["next_high"] = d["Massimo"].shift(-1) / d["Ultimo"] - 1
    d["target_2"] = (d["next_ret"] >= .02).astype(int)
    d["target_3"] = (d["next_ret"] >= .03).astype(int)
    d["target_5"] = (d["next_ret"] >= .05).astype(int)
    d["target_high_3"] = (d["next_high"] >= .03).astype(int)
    return d

def load_news_csv(path: str | Path) -> pd.DataFrame:
    n = pd.read_csv(path)
    n["published_at"] = pd.to_datetime(n["published_at"], utc=True)
    for c in ["sentiment", "intensity", "relevance", "novelty"]:
        n[c] = pd.to_numeric(n[c], errors="coerce").fillna(0)
    return n.sort_values("published_at")

def aggregate_news_causally(market: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Conservative daily merge: only prior-day news is visible to a daily bar."""
    m = market.copy()
    n = news.copy()
    n["day"] = n["published_at"].dt.floor("D")
    grouped = n.groupby("day").agg(news_sentiment=("sentiment", "mean"), news_intensity=("intensity", "mean"), news_relevance=("relevance", "mean"), news_novelty=("novelty", "mean"), news_count=("title", "count")).reset_index()
    grouped["news_sentiment"] *= grouped["news_relevance"]
    grouped["news_intensity"] *= grouped["news_relevance"]
    grouped["news_novelty"] *= grouped["news_relevance"]
    grouped["_date"] = grouped["day"]
    # For a daily OHLC observation dated D, only news published on dates < D is used.
    m["_date"] = pd.to_datetime(m["Date"], utc=True).dt.floor("D") - pd.Timedelta(days=1)
    return pd.merge_asof(m.sort_values("_date"), grouped.sort_values("_date"), on="_date", direction="backward").drop(columns=["_date"])

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
        model.fit(train[features], train[target])
        test["probability"] = model.predict_proba(test[features])[:, 1]
        test["signal"] = test["ret"] <= signal_return
        predictions.append(test)
    if not predictions:
        return pd.DataFrame(), features
    return pd.concat(predictions, ignore_index=True), features

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
    ap.add_argument("--news-csv")
    ap.add_argument("--out", default="rebound_v1_output.csv")
    args = ap.parse_args()
    df = load_market_csv(args.market_csv)
    if args.news_csv: df = aggregate_news_causally(df, load_news_csv(args.news_csv))
    scored, features = walk_forward_score(df)
    scored.to_csv(args.out, index=False)
    print(report(scored))
    print("features:", ", ".join(features))
