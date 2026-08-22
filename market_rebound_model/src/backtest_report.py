"""Walk-forward validation: baseline vs technical-only vs market-regime model.

All model decisions are made using information available before the test
observation. Ranking thresholds are derived from the training period, never
from the test year.
"""
from __future__ import annotations
import json
from pathlib import Path
import yfinance as yf
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from rebound_model import load_yahoo_ohlcv, BASE_FEATURES
from live_alert import add_market_regime, REGIME_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
TARGET = "target_3"
TECH_FEATURES = BASE_FEATURES
REGIME_FEATURES_ALL = BASE_FEATURES + REGIME_FEATURES


def fetch(symbol: str, period: str = "max") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None)
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.Series, float]:
    tr = train.dropna(subset=features + [TARGET])
    te = test.dropna(subset=features).copy()
    if len(tr) < 500 or te.empty or tr[TARGET].nunique() < 2:
        return pd.Series(dtype=float), float("nan")
    model = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, learning_rate=.05, l2_regularization=2, random_state=42)
    model.fit(tr[features], tr[TARGET].astype(int))
    return pd.Series(model.predict_proba(te[features])[:, 1], index=te.index), float("nan")


def pooled_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    results = []
    for year in sorted(data.Date.dt.year.unique()):
        train = data[data.Date.dt.year < year].copy()
        test = data[data.Date.dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        # Technical-only model.
        p_tech, _ = fit_predict(train, test, TECH_FEATURES)
        # Technical + market regime model.
        p_regime, _ = fit_predict(train, test, REGIME_FEATURES_ALL)
        test["prob_tech"] = p_tech.reindex(test.index)
        test["prob_regime"] = p_regime.reindex(test.index)
        test["baseline_signal"] = test["ret"] <= -.02
        test["tech_signal_70"] = test["baseline_signal"] & (test["prob_tech"] >= .70)
        test["regime_signal_70"] = test["baseline_signal"] & (test["prob_regime"] >= .70)
        # Prospective top-20% threshold: derive it from TRAINING predictions.
        tr_pred, _ = fit_predict(train, train, TECH_FEATURES)
        tr_pred_regime, _ = fit_predict(train, train, REGIME_FEATURES_ALL)
        train_baseline = train["ret"] <= -.02
        tech_thr = tr_pred[train_baseline.reindex(tr_pred.index, fill_value=False)].quantile(.80) if len(tr_pred) else float("nan")
        regime_thr = tr_pred_regime[train_baseline.reindex(tr_pred_regime.index, fill_value=False)].quantile(.80) if len(tr_pred_regime) else float("nan")
        test["tech_top20"] = test["baseline_signal"] & (test["prob_tech"] >= tech_thr)
        test["regime_top20"] = test["baseline_signal"] & (test["prob_regime"] >= regime_thr)
        results.append(test)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def stats(x: pd.DataFrame, label: str) -> dict:
    x = x[x["next_ret"].notna()].copy()
    if x.empty:
        return {"set": label, "n": 0}
    return {
        "set": label,
        "n": len(x),
        "mean_next_ret": x.next_ret.mean(),
        "median_next_ret": x.next_ret.median(),
        "hit_2pct": (x.next_ret >= .02).mean(),
        "hit_3pct": (x.next_ret >= .03).mean(),
        "hit_5pct": (x.next_ret >= .05).mean(),
        "mean_next_high": x.next_high.mean(),
    }


def main():
    frames = {}
    benchmark = None
    for item in CONFIG["tickers"]:
        d = load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"] == "benchmark":
            benchmark = d
        else:
            frames[item["symbol"]] = d
    frames = add_market_regime(frames, benchmark)
    all_data = pd.concat([d.assign(symbol=s) for s, d in frames.items()], ignore_index=True)
    scored = pooled_walk_forward(all_data)
    out = []
    for symbol in frames:
        s = scored[scored.symbol == symbol]
        out += [
            stats(s[s.baseline_signal], f"{symbol}: baseline -2%"),
            stats(s[s.tech_signal_70], f"{symbol}: technical >=70%"),
            stats(s[s.regime_signal_70], f"{symbol}: regime >=70%"),
            stats(s[s.tech_top20], f"{symbol}: technical top20%"),
            stats(s[s.regime_top20], f"{symbol}: regime top20%"),
        ]
    report = pd.DataFrame(out)
    result_path = ROOT / "results" / "latest_backtest_report.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(result_path, index=False)
    scored.to_csv(ROOT / "results" / "latest_oos_predictions.csv", index=False)
    print(report.to_string(index=False))
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
