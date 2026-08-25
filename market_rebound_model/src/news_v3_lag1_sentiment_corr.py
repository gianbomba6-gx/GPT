from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .news_event_classifier import add_event_features
except ImportError:
    from news_event_classifier import add_event_features

def map_lag1(raw, calendar):
    x = raw.copy()
    for c in ("symbol",): x[c] = x[c].astype(str).str.upper().str.strip()
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x["candidate_day"] = pd.to_datetime(x["candidate_day"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["published_at","candidate_day","symbol"])
    x = add_event_features(x)
    cal = calendar.copy(); cal["symbol"] = cal["symbol"].astype(str).str.upper().str.strip(); cal["Date"] = pd.to_datetime(cal["Date"], errors="coerce").dt.normalize()
    maps = {str(k): np.sort(v["Date"].dropna().unique()) for k,v in cal.groupby("symbol", sort=False)}
    def target(sym, day):
        a = maps.get(str(sym), np.array([], dtype="datetime64[ns]"))
        p = int(np.searchsorted(a, np.datetime64(day), side="right"))
        return pd.Timestamp(a[p]) if p < len(a) else pd.NaT
    x["feature_day"] = [target(s,d) for s,d in zip(x["symbol"],x["candidate_day"])]
    x = x.dropna(subset=["feature_day"])
    return x.groupby(["feature_day","symbol"], as_index=False).agg(news_count=("classification_text","size"), event_polarity=("event_polarity","mean")).rename(columns={"feature_day":"Date"})

def spearman(x,y):
    if len(x)<3 or pd.Series(x).nunique()<2 or pd.Series(y).nunique()<2: return np.nan
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("rows_csv"); ap.add_argument("raw_gkg"); ap.add_argument("--out",default="results/news_v3_lag1_sentiment_corr.csv"); args=ap.parse_args()
    rows=pd.read_csv(args.rows_csv); raw=pd.read_csv(args.raw_gkg)
    rows["Date"]=pd.to_datetime(rows["Date"],errors="coerce").dt.normalize(); rows["symbol"]=rows["symbol"].astype(str).str.upper().str.strip(); rows["next_ret"]=pd.to_numeric(rows["next_ret"],errors="coerce"); rows["v1_top20"]=rows["v1_top20"].fillna(False).astype(bool)
    base=rows[rows["v1_top20"]].dropna(subset=["Date"]).copy(); base=base.sort_values(["Date","symbol"]).reset_index(drop=True)
    sent=map_lag1(raw,base[["Date","symbol"]]); x=base.drop(columns=["news_count","event_polarity"],errors="ignore").merge(sent,on=["Date","symbol"],how="left",validate="many_to_one"); x["news_count"]=pd.to_numeric(x.get("news_count",0),errors="coerce").fillna(0); x["event_polarity"]=pd.to_numeric(x.get("event_polarity",0.0),errors="coerce").fillna(0.0)
    out=[]
    for sym,s in x.groupby("symbol",sort=True):
        s=s[(s["next_ret"].notna())&(s["news_count"]>0)]; r=spearman(s["event_polarity"].to_numpy(float),s["next_ret"].to_numpy(float)); out.append({"symbol":sym,"n_base":int(len(x[x.symbol==sym])),"n_news_days":int(len(s)),"spearman_polarity_next_ret":r,"status":"OK" if np.isfinite(r) else "INSUFFICIENT_VARIATION"})
    report=pd.DataFrame(out).sort_values("symbol"); Path(args.out).parent.mkdir(parents=True,exist_ok=True); report.to_csv(args.out,index=False); print("NEWS V3 LAG1 SENTIMENT CORRELATION"); print(report.to_string(index=False)); print("NEWS V3 LAG1 SENTIMENT CORRELATION PASS")
if __name__=="__main__": main()
