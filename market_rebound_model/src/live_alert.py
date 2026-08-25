"""Daily Yahoo Finance rebound scanner with Discord alerts.

The live scanner uses a pooled cross-ticker expanding-window model and adds
Euro Stoxx 50 market-regime features. The production threshold can be fixed
(legacy) or derived dynamically as the training 90th percentile of model
probabilities among rebound candidates.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

from rebound_model import BASE_FEATURES, load_yahoo_ohlcv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
TARGET = "target_3"
REGIME_FEATURES = [
    "mkt_ret",
    "mkt_ret_5",
    "mkt_ret_20",
    "mkt_vol_20",
    "rel_ret_1",
    "rel_ret_5",
    "rel_ret_20",
]
FEATURES = BASE_FEATURES + REGIME_FEATURES
SEED = 42


def fetch(symbol: str, period: str = "max") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(
        columns={
            "Close": "Ultimo",
            "Open": "Apertura",
            "High": "Massimo",
            "Low": "Minimo",
            "Volume": "Vol.",
        }
    )
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize().astype("datetime64[ns]")
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def discord(message: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DISCORD_WEBHOOK_URL secret is not configured")
    r = requests.post(url, json={"content": message}, timeout=20)
    r.raise_for_status()


def add_market_regime(
    frames: dict[str, pd.DataFrame], benchmark: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    b = benchmark[["Date", "Ultimo"]].copy()
    b["Date"] = pd.to_datetime(b["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
    b = b.sort_values("Date")
    b["mkt_ret"] = b["Ultimo"].pct_change()
    b["mkt_ret_5"] = b["Ultimo"].pct_change(5)
    b["mkt_ret_20"] = b["Ultimo"].pct_change(20)
    b["mkt_vol_20"] = b["mkt_ret"].rolling(20).std()
    b = b.drop(columns=["Ultimo"])
    out = {}
    for symbol, d in frames.items():
        x = d.copy()
        x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize().astype("datetime64[ns]")
        x = x.sort_values("Date")
        z = pd.merge_asof(x, b, on="Date", direction="backward")
        z["rel_ret_1"] = z["ret"] - z["mkt_ret"]
        z["rel_ret_5"] = z["ret_5"] - z["mkt_ret_5"]
        z["rel_ret_20"] = z["ret_20"] - z["mkt_ret_20"]
        out[symbol] = z
    return out


def fit_pooled_model(all_data: pd.DataFrame):
    trainable = all_data.dropna(subset=FEATURES + [TARGET]).copy()
    years = sorted(trainable.Date.dt.year.unique())
    if len(years) < 2:
        raise RuntimeError("Insufficient calendar history for walk-forward model")
    latest_year = max(years)
    train = trainable[trainable.Date.dt.year < latest_year].copy()
    if len(train) < 500 or train[TARGET].nunique() < 2:
        raise RuntimeError("Insufficient pooled history for latest prediction")
    model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=2,
        random_state=SEED,
    )
    model.fit(train[FEATURES], train[TARGET].astype(int))
    return model, trainable, train, latest_year


def dynamic_top_cutoff(model, train: pd.DataFrame) -> tuple[float, int]:
    signal = CONFIG["signal"]
    quantile = float(signal.get("dynamic_top_quantile", 0.90))
    if not 0.0 < quantile < 1.0:
        raise RuntimeError(f"Invalid dynamic_top_quantile: {quantile}")
    candidates = train[train["ret"] <= float(signal["down_day_threshold"])].dropna(
        subset=FEATURES
    )
    if len(candidates) < 20:
        raise RuntimeError(
            f"Insufficient rebound candidates for dynamic cutoff: {len(candidates)}"
        )
    probabilities = model.predict_proba(candidates[FEATURES])[:, 1]
    cutoff = float(pd.Series(probabilities).quantile(quantile))
    if not 0.0 < cutoff < 1.0:
        raise RuntimeError(f"Invalid dynamic probability cutoff: {cutoff}")
    return cutoff, len(candidates)


def resolve_threshold(model, train: pd.DataFrame) -> tuple[float, str, int | None]:
    signal = CONFIG["signal"]
    mode = str(signal.get("threshold_mode", "fixed")).strip().lower()
    if mode == "fixed":
        return float(signal["alert_probability_threshold"]), "fixed", None
    if mode == "dynamic_top10":
        cutoff, n_candidates = dynamic_top_cutoff(model, train)
        return cutoff, "dynamic_top10", n_candidates
    raise RuntimeError(f"Unknown threshold_mode: {mode}")


def main() -> int:
    alerts = []
    down_threshold = float(CONFIG["signal"]["down_day_threshold"])
    frames: dict[str, pd.DataFrame] = {}
    benchmark_symbol = next(
        x["symbol"] for x in CONFIG["tickers"] if x["type"] == "benchmark"
    )

    for item in CONFIG["tickers"]:
        symbol = item["symbol"]
        try:
            raw = load_yahoo_ohlcv(fetch(symbol))
            if item["type"] == "benchmark":
                benchmark = raw
            else:
                frames[symbol] = raw
                print(f"DATA {symbol}: {len(raw)} rows, latest={raw.iloc[-1]['Date'].date()}")
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")

    if not frames:
        raise RuntimeError("No ticker data could be downloaded")
    if "benchmark" not in locals():
        raise RuntimeError(f"No benchmark data for {benchmark_symbol}")

    frames = add_market_regime(frames, benchmark)
    rows = []
    for symbol, d in frames.items():
        z = d.copy()
        z["symbol"] = symbol
        rows.append(z)
    all_data = pd.concat(rows, ignore_index=True)
    all_data["Date"] = pd.to_datetime(all_data["Date"], errors="coerce").dt.normalize()

    model, trainable, train, latest_year = fit_pooled_model(all_data)
    threshold, threshold_mode, n_candidates = resolve_threshold(model, train)
    print(
        f"THRESHOLD mode={threshold_mode} value={threshold:.6f} "
        f"latest_year={latest_year} train_rows={len(train)}"
        + (f" train_rebound_candidates={n_candidates}" if n_candidates is not None else "")
    )

    latest_by_symbol = {symbol: d.sort_values("Date").iloc[-1].copy() for symbol, d in frames.items()}
    predictions = {}
    for symbol, latest in latest_by_symbol.items():
        if latest[FEATURES].isna().any():
            print(f"SKIP {symbol}: insufficient features for prediction")
            continue
        latest["probability"] = float(
            model.predict_proba(pd.DataFrame([latest])[FEATURES])[:, 1][0]
        )
        predictions[symbol] = latest

    for item in CONFIG["tickers"]:
        if item["type"] == "benchmark":
            continue
        symbol = item["symbol"]
        if symbol not in predictions:
            continue
        latest = predictions[symbol]
        probability = float(latest["probability"])
        ret = float(latest["ret"])
        print(
            f"CHECK {symbol}: date={latest['Date'].date()} ret={ret:.2%} "
            f"model_p={probability:.1%} threshold={threshold:.1%} "
            f"mkt_1d={latest['mkt_ret']:.2%} rel_1d={latest['rel_ret_1']:.2%}"
        )
        if ret <= down_threshold and probability >= threshold:
            alerts.append(
                f"🚨 **REBOUND SIGNAL — {symbol} ({item['name']})**\n"
                f"Close: {latest['Ultimo']:.2f}\n"
                f"Daily move: {ret:.2%}\n"
                f"P(next-day +3%): **{probability:.1%}**\n"
                f"Dynamic cutoff ({threshold_mode}): **{threshold:.1%}**\n"
                f"Volume ratio: {latest['vol_ratio']:.2f}x\n"
                f"20d drawdown: {latest['dd_20']:.2%}\n"
                f"Market 1d: {latest['mkt_ret']:.2%} | Relative: {latest['rel_ret_1']:.2%}"
            )

    if alerts:
        discord("\n\n".join(alerts))
        print(f"Sent {len(alerts)} Discord alert(s).")
    else:
        print("No qualifying rebound signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
