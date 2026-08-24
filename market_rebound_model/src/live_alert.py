"""Daily Yahoo Finance rebound scanner with Discord alerts.

The live scanner uses a pooled cross-ticker expanding-window model and adds
Euro Stoxx 50 market-regime features to distinguish idiosyncratic selloffs
from broad market selloffs.
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
REGIME_FEATURES = ["mkt_ret", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "rel_ret_1", "rel_ret_5", "rel_ret_20"]


def fetch(symbol: str, period: str = "10y") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize().astype("datetime64[ns]")
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def discord(message: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DISCORD_WEBHOOK_URL secret is not configured")
    r = requests.post(url, json={"content": message}, timeout=20)
    r.raise_for_status()


def add_market_regime(frames: dict[str, pd.DataFrame], benchmark: pd.DataFrame) -> dict[str, pd.DataFrame]:
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


def pooled_walk_forward_predict(frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    rows = []
    for symbol, d in frames.items():
        z = d.copy()
        z["symbol"] = symbol
        rows.append(z)
    all_data = pd.concat(rows, ignore_index=True)
    features = BASE_FEATURES + REGIME_FEATURES
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
        latest["probability"] = float(model.predict_proba(pd.DataFrame([latest])[features])[:, 1][0])
        result[symbol] = latest
    return result


def main() -> int:
    alerts = []
    threshold = CONFIG["signal"]["alert_probability_threshold"]
    down_threshold = CONFIG["signal"]["down_day_threshold"]
    frames: dict[str, pd.DataFrame] = {}
    benchmark_symbol = next(x["symbol"] for x in CONFIG["tickers"] if x["type"] == "benchmark")
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
        print(f"CHECK {symbol}: date={latest['Date'].date()} ret={ret:.2%} model_p={probability:.1%} mkt_1d={latest['mkt_ret']:.2%} rel_1d={latest['rel_ret_1']:.2%}")
        if ret <= down_threshold and probability >= threshold:
            alerts.append(
                f"🚨 **REBOUND SIGNAL — {symbol} ({item['name']})**\n"
                f"Close: {latest['Ultimo']:.2f}\n"
                f"Daily move: {ret:.2%}\n"
                f"P(next-day +3%): **{probability:.1%}**\n"
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
