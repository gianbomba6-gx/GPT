"""Daily Yahoo Finance rebound scanner with Discord alerts.

The live scanner uses a pooled cross-ticker expanding-window model. This is
important for short-history instruments such as SPCX: they contribute to the
training set but do not require an independent 500-observation history.

Only completed daily bars are scored. The latest bar is scored by a model
trained exclusively on observations from earlier calendar years.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import requests
import yfinance as yf
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from rebound_model import load_yahoo_ohlcv, BASE_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
TARGET = "target_3"


def fetch(symbol: str, period: str = "10y") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None)
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def discord(message: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DISCORD_WEBHOOK_URL secret is not configured")
    r = requests.post(url, json={"content": message}, timeout=20)
    r.raise_for_status()


def pooled_walk_forward_predict(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return latest-row probabilities using a pooled expanding-window model."""
    rows = []
    for symbol, d in frames.items():
        z = d.copy()
        z["symbol"] = symbol
        rows.append(z)
    all_data = pd.concat(rows, ignore_index=True)
    features = BASE_FEATURES
    trainable = all_data.dropna(subset=features + [TARGET]).copy()
    latest_by_symbol = {s: d.iloc[-1].copy() for s, d in frames.items()}
    years = sorted(trainable.Date.dt.year.unique())
    if len(years) < 2:
        raise RuntimeError("Insufficient calendar history for walk-forward model")
    latest_year = max(years)
    train = trainable[trainable.Date.dt.year < latest_year]
    if len(train) < 500:
        raise RuntimeError("Insufficient pooled history for latest prediction")
    model = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15, learning_rate=.05, l2_regularization=2, random_state=42)
    model.fit(train[features], train[TARGET].astype(int))
    result = {}
    for symbol, latest in latest_by_symbol.items():
        if latest[features].isna().any():
            continue
        latest_df = pd.DataFrame([latest])
        latest["probability"] = float(model.predict_proba(latest_df[features])[:, 1][0])
        result[symbol] = latest
    return result


def main() -> int:
    alerts = []
    threshold = CONFIG["signal"]["alert_probability_threshold"]
    down_threshold = CONFIG["signal"]["down_day_threshold"]
    frames: dict[str, pd.DataFrame] = {}
    for item in CONFIG["tickers"]:
        if item["type"] == "benchmark":
            continue
        symbol = item["symbol"]
        try:
            frames[symbol] = load_yahoo_ohlcv(fetch(symbol))
            print(f"DATA {symbol}: {len(frames[symbol])} rows, latest={frames[symbol].iloc[-1]['Date'].date()}")
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")
    if not frames:
        raise RuntimeError("No ticker data could be downloaded")

    predictions = pooled_walk_forward_predict(frames)
    for item in CONFIG["tickers"]:
        if item["type"] == "benchmark":
            continue
        symbol = item["symbol"]
        if symbol not in predictions:
            print(f"SKIP {symbol}: insufficient features for prediction")
            continue
        latest = predictions[symbol]
        probability = latest["probability"]
        ret = float(latest["ret"])
        print(f"CHECK {symbol}: date={latest['Date'].date()} ret={ret:.2%} model_p={probability:.1%}")
        if ret <= down_threshold and probability >= threshold:
            alerts.append(
                f"🚨 **REBOUND SIGNAL — {symbol} ({item['name']})**\n"
                f"Close: {latest['Ultimo']:.2f}\n"
                f"Daily move: {ret:.2%}\n"
                f"P(next-day +3%): **{probability:.1%}**\n"
                f"Volume ratio: {latest['vol_ratio']:.2f}x\n"
                f"20d drawdown: {latest['dd_20']:.2%}"
            )
    if alerts:
        discord("\n\n".join(alerts))
        print(f"Sent {len(alerts)} Discord alert(s).")
    else:
        print("No qualifying rebound signals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
