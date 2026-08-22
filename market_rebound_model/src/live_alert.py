"""Daily Yahoo Finance scanner with Discord alerts.

The scanner is intentionally conservative: it only alerts when the latest
completed daily bar is a sufficiently large decline and the walk-forward
model probability clears the configured threshold. Discord credentials are
read only from the DISCORD_WEBHOOK_URL environment variable.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import requests
import yfinance as yf
import pandas as pd
from rebound_model import walk_forward_score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())

def fetch(symbol: str, period: str = "10y") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Data"] = pd.to_datetime(x.index).strftime("%d/%m/%Y")
    return x.reset_index(drop=True)

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
            raw = fetch(symbol)
            from rebound_model import load_market_csv
            # Reuse the model's feature engineering without requiring a physical CSV.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
                raw[["Data","Ultimo","Apertura","Massimo","Minimo","Vol."]].to_csv(f.name, index=False)
                tmp = f.name
            d = load_market_csv(tmp)
            os.unlink(tmp)
            scored, _ = walk_forward_score(d)
            if scored.empty:
                continue
            last = scored.iloc[-1]
            if float(last["ret"]) <= down_threshold and float(last["probability"]) >= threshold:
                alerts.append(
                    f"🚨 **REBOUND SIGNAL — {symbol} ({item['name']})**\n"
                    f"Close: {last['Ultimo']:.2f}\n"
                    f"Daily move: {last['ret']:.2%}\n"
                    f"P(next-day +3%): **{last['probability']:.1%}**\n"
                    f"Volume ratio: {last['vol_ratio']:.2f}x\n"
                    f"20d drawdown: {last['dd_20']:.2%}"
                )
        except Exception as exc:
            print(f"ERROR {symbol}: {exc}")
    if alerts:
        discord("\n\n".join(alerts))
    else:
        print("No qualifying rebound signals.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
