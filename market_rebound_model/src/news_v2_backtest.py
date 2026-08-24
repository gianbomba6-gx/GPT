"""Walk-forward comparison of V1, V2 and event-aware V3 news features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import yfinance as yf

from rebound_model import BASE_FEATURES, load_yahoo_ohlcv
from live_alert import add_market_regime, REGIME_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
TARGET = "target_3"
SIGNAL_RETURN = -0.02
BASE_NEWS_FEATURES = [
    "news_sentiment", "news_intensity", "news_relevance", "news_novelty",
    "news_count", "negative_news_share", "material_event_share",
    "event_polarity", "event_intensity", "unique_event_types", "news_available",
]
EVENT_SHARE_FEATURES = [
    "event_earnings_share", "event_guidance_share", "event_analyst_share",
    "event_regulatory_share", "event_ma_share", "event_product_share",
    "event_macro_share", "event_other_share",
]
V2_NEWS_FEATURES = BASE_NEWS_FEATURES
V3_EVENT_AWARE_FEATURES = BASE_NEWS_FEATURES + EVENT_SHARE_FEATURES


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize().astype("datetime64[ns]")
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]


def load_news(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    if d.empty:
        return pd.DataFrame(columns=["Date", "symbol", *V3_EVENT_AWARE_FEATURES])
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    for c in V3_EVENT_AWARE_FEATURES:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "news_available" not in d.columns:
        d["news_available"] = 1.0
    return d.dropna(subset=["Date", "symbol"])


def _normalize_merge_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize().astype("datetime64[ns]")


def merge_news(market: pd.DataFrame, news: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = market.copy()
    x["Date"] = _normalize_merge_date(x["Date"])
    x = x.sort_values("Date")
    n = news[news.symbol == symbol].copy()
    cols = ["Date", "symbol"] + [c for c in V3_EVENT_AWARE_FEATURES if c in n.columns]
    n = n[cols].drop_duplicates(["Date", "symbol"])
    n["available_date"] = _normalize_merge_date(n["Date"])
    n = n.dropna(subset=["available_date"]).sort_values("available_date")
    if not n.empty:
        sessions = x[["Date"]].drop_duplicates().rename(columns={"Date": "session_date"})
        mapped = pd.merge_asof(n, sessions, left_on="available_date", right_on="session_date", direction="forward").dropna(subset=["session_date"])
        agg_cols = [c for c in V3_EVENT_AWARE_FEATURES if c in mapped.columns]
        agg_spec = {c: ("sum" if c == "news_count" else "mean") for c in agg_cols}
        mapped = mapped.groupby("session_date", as_index=False).agg(agg_spec)
        mapped = mapped.rename(columns={"session_date": "Date"})
        mapped["Date"] = _normalize_merge_date(mapped["Date"])
        x = x.merge(mapped, on="Date", how="left")
    for c in V3_EVENT_AWARE_FEATURES:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    x["news_available"] = x["news_available"].astype(float)
    return x


def model_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.Series:
    tr = train.dropna(subset=features + [TARGET])
    te = test.dropna(subset=features)
    if len(tr) < 500 or te.empty or tr[TARGET].nunique() < 2:
        return pd.Series(dtype=float)
    model = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, learning_rate=.05, l2_regularization=2, random_state=42)
    model.fit(tr[features], tr[TARGET].astype(int))
    return pd.Series(model.predict_proba(te[features])[:, 1], index=te.index)


def prior_oos_predictions(prior: pd.DataFrame, features: list[str]) -> pd.Series:
    outputs = []
    for year in sorted(prior.Date.dt.year.unique()):
        train = prior[prior.Date.dt.year < year]
        test = prior[prior.Date.dt.year == year]
        if len(train) < 500 or test.empty:
            continue
        p = model_predict(train, test, features)
        if not p.empty:
            outputs.append(p)
    return pd.concat(outputs) if outputs else pd.Series(dtype=float)


def pooled_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    results = []
    v1 = BASE_FEATURES + REGIME_FEATURES
    v2 = v1 + V2_NEWS_FEATURES
    v3 = v1 + V3_EVENT_AWARE_FEATURES
    candidate = data[data["ret"] <= SIGNAL_RETURN].copy()
    for year in sorted(candidate.Date.dt.year.unique()):
        train = candidate[candidate.Date.dt.year < year].copy()
        test = candidate[candidate.Date.dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        p1 = model_predict(train, test, v1)
        p2 = model_predict(train, test, v2)
        p3 = model_predict(train, test, v3)
        test["prob_v1"] = p1.reindex(test.index)
        test["prob_v2"] = p2.reindex(test.index)
        test["prob_v3"] = p3.reindex(test.index)
        test["baseline_signal"] = True
        prior_v1 = prior_oos_predictions(train, v1)
        prior_v2 = prior_oos_predictions(train, v2)
        prior_v3 = prior_oos_predictions(train, v3)
        thr1 = prior_v1.quantile(.80) if len(prior_v1) else float("nan")
        thr2 = prior_v2.quantile(.80) if len(prior_v2) else float("nan")
        thr3 = prior_v3.quantile(.80) if len(prior_v3) else float("nan")
        test["v1_70"] = test["prob_v1"] >= .70
        test["v2_70"] = test["prob_v2"] >= .70
        test["v3_70"] = test["prob_v3"] >= .70
        test["v1_top20"] = test["prob_v1"] >= thr1
        test["v2_top20"] = test["prob_v2"] >= thr2
        test["v3_top20"] = test["prob_v3"] >= thr3
        results.append(test)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def stats(x: pd.DataFrame, label: str) -> dict:
    x = x[x.next_ret.notna()].copy()
    return {
        "set": label,
        "n": len(x),
        "mean_next_ret": float(x.next_ret.mean()) if len(x) else None,
        "median_next_ret": float(x.next_ret.median()) if len(x) else None,
        "hit_2pct": float((x.next_ret >= .02).mean()) if len(x) else None,
        "hit_3pct": float((x.next_ret >= .03).mean()) if len(x) else None,
        "hit_5pct": float((x.next_ret >= .05).mean()) if len(x) else None,
        "mean_next_high": float(x.next_high.mean()) if len(x) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("news_daily")
    ap.add_argument("--out", default="results/latest_news_v3_event_aware_backtest.csv")
    args = ap.parse_args()
    news = load_news(args.news_daily)
    frames = {}
    benchmark = None
    for item in CONFIG["tickers"]:
        d = load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"] == "benchmark":
            benchmark = d
        else:
            frames[item["symbol"]] = merge_news(d, news, item["symbol"])
    if benchmark is None:
        raise RuntimeError("Benchmark data missing")
    frames = add_market_regime(frames, benchmark)
    all_data = pd.concat([d.assign(symbol=s) for s, d in frames.items()], ignore_index=True)
    scored = pooled_walk_forward(all_data)
    if scored.empty:
        raise RuntimeError("No out-of-sample candidate rows were scored")
    rows = []
    for symbol in frames:
        s = scored[scored.symbol == symbol]
        rows += [
            stats(s[s.baseline_signal], f"{symbol}: baseline -2%"),
            stats(s[s.v1_70], f"{symbol}: V1 regime >=70%"),
            stats(s[s.v2_70], f"{symbol}: V2 + news >=70%"),
            stats(s[s.v3_70], f"{symbol}: V3 event-aware >=70%"),
            stats(s[s.v1_top20], f"{symbol}: V1 regime top20%"),
            stats(s[s.v2_top20], f"{symbol}: V2 + news top20%"),
            stats(s[s.v3_top20], f"{symbol}: V3 event-aware top20%"),
        ]
    out = pd.DataFrame(rows)
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)
    scored.to_csv(ROOT / "results/latest_news_v3_oos_predictions.csv", index=False)
    print(out.to_string(index=False))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
