"""Walk-forward audit of V1 target formulations.

Compares the production target_3 classifier against target_2, target_5 and a
continuous next-day-return regressor. No production code is changed by this
script; all selection thresholds are derived from the training period.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from rebound_model import BASE_FEATURES
from live_alert import add_market_regime, REGIME_FEATURES

FEATURES = BASE_FEATURES + REGIME_FEATURES
VARIANTS = ("target_2", "target_3", "target_5", "regression")


def fit_classifier(train: pd.DataFrame, test: pd.DataFrame, target: str):
    tr = train.dropna(subset=FEATURES + [target]).copy()
    te = test.dropna(subset=FEATURES).copy()
    if len(tr) < 500 or te.empty or tr[target].nunique() < 2:
        return pd.Series(dtype=float), tr, te
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=15, learning_rate=.05,
        l2_regularization=2, random_state=42
    )
    model.fit(tr[FEATURES], tr[target].astype(int))
    return pd.Series(model.predict_proba(te[FEATURES])[:, 1], index=te.index), tr, te


def fit_regressor(train: pd.DataFrame, test: pd.DataFrame):
    tr = train.dropna(subset=FEATURES + ["next_ret"]).copy()
    te = test.dropna(subset=FEATURES).copy()
    if len(tr) < 500 or te.empty:
        return pd.Series(dtype=float), tr, te
    model = HistGradientBoostingRegressor(
        max_iter=250, max_leaf_nodes=15, learning_rate=.05,
        l2_regularization=2, random_state=42, loss="squared_error"
    )
    model.fit(tr[FEATURES], tr["next_ret"].astype(float))
    return pd.Series(model.predict(te[FEATURES]), index=te.index), tr, te


def training_top20_threshold(train: pd.DataFrame, score: pd.Series) -> float:
    baseline = train["ret"] <= -.02
    usable = score.index.intersection(train.index)
    s = score.reindex(usable)
    b = baseline.reindex(usable, fill_value=False)
    v = s[b].dropna()
    return float(v.quantile(.80)) if len(v) >= 20 else float("nan")


def yearly_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        test["baseline_signal"] = test["ret"] <= -.02

        for variant in VARIANTS:
            if variant.startswith("target_"):
                score, tr, _ = fit_classifier(train, test, variant)
            else:
                score, tr, _ = fit_regressor(train, test)
            if score.empty:
                continue
            thr = training_top20_threshold(train, score.reindex(train.index))
            # To avoid accidentally using test labels, the threshold comes from
            # a separate model fitted only on the training period.
            if not np.isfinite(thr):
                # Refit on train and create training scores for the threshold.
                if variant.startswith("target_"):
                    model = HistGradientBoostingClassifier(
                        max_iter=250, max_leaf_nodes=15, learning_rate=.05,
                        l2_regularization=2, random_state=42
                    )
                    tr2 = train.dropna(subset=FEATURES + [variant]).copy()
                    if len(tr2) < 500 or tr2[variant].nunique() < 2:
                        continue
                    model.fit(tr2[FEATURES], tr2[variant].astype(int))
                    train_score = pd.Series(model.predict_proba(tr2[FEATURES])[:, 1], index=tr2.index)
                else:
                    model = HistGradientBoostingRegressor(
                        max_iter=250, max_leaf_nodes=15, learning_rate=.05,
                        l2_regularization=2, random_state=42, loss="squared_error"
                    )
                    tr2 = train.dropna(subset=FEATURES + ["next_ret"]).copy()
                    if len(tr2) < 500:
                        continue
                    model.fit(tr2[FEATURES], tr2["next_ret"].astype(float))
                    train_score = pd.Series(model.predict(tr2[FEATURES]), index=tr2.index)
                thr = training_top20_threshold(train, train_score)
            test[f"score_{variant}"] = score.reindex(test.index)
            test[f"signal_{variant}"] = test["baseline_signal"] & (test[f"score_{variant}"] >= thr)

            selected = test[test[f"signal_{variant}"] & test["next_ret"].notna()].copy()
            baseline = test[test["baseline_signal"] & test["next_ret"].notna()].copy()
            if selected.empty or baseline.empty:
                continue
            auc = float("nan")
            if variant.startswith("target_"):
                y = test.loc[test["baseline_signal"] & test["next_ret"].notna(), variant]
                p = test.loc[y.index, f"score_{variant}"]
                if len(y) >= 10 and y.nunique() > 1:
                    auc = float(roc_auc_score(y.astype(int), p))
            out.append({
                "year": int(year),
                "variant": variant,
                "n_baseline": len(baseline),
                "n_selected": len(selected),
                "mean_baseline": float(baseline["next_ret"].mean()),
                "mean_selected": float(selected["next_ret"].mean()),
                "delta_mean": float(selected["next_ret"].mean() - baseline["next_ret"].mean()),
                "hit3_baseline": float((baseline["next_ret"] >= .03).mean()),
                "hit3_selected": float((selected["next_ret"] >= .03).mean()),
                "auc": auc,
            })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_objective_audit.csv")
    args = ap.parse_args()

    # Reuse the same live data acquisition path as V1/backtest_report.
    import json
    import yfinance as yf
    from rebound_model import load_yahoo_ohlcv
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "tickers.json").read_text())

    def fetch(symbol: str) -> pd.DataFrame:
        x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
        if x.empty:
            raise RuntimeError(f"No Yahoo data for {symbol}")
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
        x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize()
        return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]

    frames: dict[str, pd.DataFrame] = {}
    benchmark = None
    for item in config["tickers"]:
        d = load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"] == "benchmark":
            benchmark = d
        else:
            frames[item["symbol"]] = d
    if benchmark is None:
        raise RuntimeError("Missing benchmark")
    frames = add_market_regime(frames, benchmark)
    data = pd.concat([d.assign(symbol=s) for s, d in frames.items()], ignore_index=True)
    report = yearly_walk_forward(data)
    if report.empty:
        raise RuntimeError("No audit rows generated")
    report.to_csv(root / args.out, index=False)
    print("V1 OBJECTIVE AUDIT")
    print(report.to_string(index=False))
    print(f"Saved {root / args.out}")


if __name__ == "__main__":
    main()
