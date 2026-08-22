"""Walk-forward validation report for the pooled rebound model.

Downloads Yahoo daily data, evaluates a simple -2% baseline versus the ML
model, and writes a CSV with out-of-sample signal statistics. The report is
intended for research/validation, not live trading.
"""
from __future__ import annotations
import json
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from rebound_model import load_yahoo_ohlcv, BASE_FEATURES
from live_alert import add_market_regime, REGIME_FEATURES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "tickers.json").read_text())
TARGET = "target_3"
FEATURES = BASE_FEATURES + REGIME_FEATURES

def fetch(symbol: str, period: str = "max") -> pd.DataFrame:
    x = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if x.empty: raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close":"Ultimo", "Open":"Apertura", "High":"Massimo", "Low":"Minimo", "Volume":"Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None)
    return x.reset_index(drop=True)[["Date","Ultimo","Apertura","Massimo","Minimo","Vol."]]

def pooled_walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    results=[]
    years=sorted(data.Date.dt.year.unique())
    for year in years:
        train=data[(data.Date.dt.year < year)].dropna(subset=FEATURES+[TARGET])
        test=data[data.Date.dt.year == year].dropna(subset=FEATURES+[TARGET]).copy()
        if len(train)<500 or test.empty: continue
        model=HistGradientBoostingClassifier(max_iter=250,max_leaf_nodes=15,learning_rate=.05,l2_regularization=2,random_state=42)
        model.fit(train[FEATURES],train[TARGET].astype(int))
        test["probability"]=model.predict_proba(test[FEATURES])[:,1]
        test["baseline_signal"]=test["ret"]<=-.02
        test["model_signal"]=test["baseline_signal"] & (test["probability"]>=.70)
        test["model_top20"]=test["baseline_signal"] & (test["probability"]>=test.loc[test.baseline_signal,"probability"].quantile(.80) if test.baseline_signal.any() else False)
        results.append(test)
    return pd.concat(results,ignore_index=True) if results else pd.DataFrame()

def stats(x: pd.DataFrame, label: str) -> dict:
    if x.empty: return {"set":label,"n":0}
    return {"set":label,"n":len(x),"mean_next_ret":x.next_ret.mean(),"median_next_ret":x.next_ret.median(),"hit_2pct":(x.next_ret>=.02).mean(),"hit_3pct":(x.next_ret>=.03).mean(),"hit_5pct":(x.next_ret>=.05).mean(),"mean_next_high":x.next_high.mean()}

def main():
    frames={}
    benchmark=None
    for item in CONFIG["tickers"]:
        d=load_yahoo_ohlcv(fetch(item["symbol"]))
        if item["type"]=="benchmark": benchmark=d
        else: frames[item["symbol"]]=d
    frames=add_market_regime(frames,benchmark)
    all_data=pd.concat([d.assign(symbol=s) for s,d in frames.items()],ignore_index=True)
    scored=pooled_walk_forward(all_data)
    out=[]
    for symbol in frames:
        s=scored[scored.symbol==symbol]
        out += [stats(s[s.baseline_signal],f"{symbol}: baseline -2%"),stats(s[s.model_signal],f"{symbol}: model >=70%"),stats(s[s.model_top20],f"{symbol}: model top20%")]
    report=pd.DataFrame(out)
    result_path=ROOT/"results"/"latest_backtest_report.csv"
    report.to_csv(result_path,index=False)
    scored.to_csv(ROOT/"results"/"latest_oos_predictions.csv",index=False)
    print(report.to_string(index=False))
    print(f"Saved {result_path}")

if __name__=="__main__": main()
