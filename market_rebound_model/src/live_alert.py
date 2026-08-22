"""Daily Yahoo Finance scanner with Discord alerts."""
from __future__ import annotations
import json, os
from pathlib import Path
import requests
import yfinance as yf
import pandas as pd
from rebound_model import load_yahoo_ohlcv, predict_latest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())


def fetch(symbol: str, period: str = "10y") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close": "Ultimo", "Open": "Apertura", "High": "Massimo", "Low": "Minimo", "Volume": "Vol."})
    idx = pd.to_datetime(x.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    x["Date"] = idx
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def discord(message: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DISCORD_WEBHOOK_URL secret is not configured")
    r = requests.post(url, json={"content": message}, timeout=20)
    r.raise_for_status()


def main() -> int:
    alerts = []
    threshold = CONFIG["signal"]["alert_probability_threshold"]
    down_threshold = CONFIG["signal"]["down_day_threshold"]
    for item in CONFIG["tickers"]:
        if item["type"] == "benchmark":
            continue
        symbol = item["symbol"]
        try:
            d = load_yahoo_ohlcv(fetch(symbol))
            probability, _ = predict_latest(d)
            latest = d.iloc[-1]
            print(f"CHECK {symbol}: date={latest['Date'].date()} ret={latest['ret']:.2%} model_p={probability:.1%}")
            if latest["ret"] <= down_threshold and probability >= threshold:
                alerts.append(
                    f"🚨 **REBOUND SIGNAL — {symbol} ({item['name']})**\n"
                    f"Close: {latest['Ultimo']:.2f}\n"
                    f"Daily move: {latest['ret']:.2%}\n"
                    f"P(next-day +3%): **{probability:.1%}**\n"
                    f"Volume ratio: {latest['vol_ratio']:.2f}x\n"
                    f"20d drawdown: {latest['dd_20']:.2%}"
                )
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")
    if alerts:
        discord("\n\n".join(alerts))
        print(f"Sent {len(alerts)} Discord alert(s).")
    else:
        print("No qualifying rebound signals.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
