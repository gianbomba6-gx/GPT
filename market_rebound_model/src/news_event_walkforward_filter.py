"""Walk-forward, out-of-sample event filter learned only from prior observations."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from .news_event_classifier import EVENT_TYPES, add_event_features
except ImportError:
    from news_event_classifier import EVENT_TYPES, add_event_features

MARKET_META = {
    "STLAM.MI": ("Europe/Rome", time(17, 30)),
    "SPCX": ("America/New_York", time(16, 0)),
    "NVDA": ("America/New_York", time(16, 0)),
    "TSLA": ("America/New_York", time(16, 0)),
}
MIN_N = 20
EVENT_CONDITIONS = ("event_negative", "event_dominant_negative")


def close_utc(day: date, symbol: str) -> pd.Timestamp:
    tz_name, close_time = MARKET_META[symbol]
    return pd.Timestamp(datetime.combine(day, close_time, tzinfo=ZoneInfo(tz_name))).tz_convert("UTC")


def _event_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"published_at", "symbol", "candidate_day"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    x = raw.copy()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    x["candidate_day"] = pd.to_datetime(x["candidate_day"], errors="coerce").dt.normalize()
    x = x.dropna(subset=["published_at", "candidate_day", "symbol"]).copy()
    x["market_close_utc"] = [
        close_utc(day.date(), symbol) if symbol in MARKET_META else pd.NaT
        for symbol, day in zip(x["symbol"], x["candidate_day"])
    ]
    x = x[x["market_close_utc"].notna() & (x["published_at"] <= x["market_close_utc"])].copy()
    if x.empty:
        return pd.DataFrame(columns=["Date", "symbol", "news_count", *[f"event_{e}_share" for e in EVENT_TYPES], *[f"negative_event_{e}_share" for e in EVENT_TYPES]])
    x = add_event_features(x)
    keys = ["candidate_day", "symbol"]
    base = x.groupby(keys, as_index=False).agg(news_count=("event_type", "size"))
    counts = pd.crosstab([x["candidate_day"], x["symbol"]], x["event_type"]).reset_index()
    counts = counts.rename(columns={c: f"event_{c}_count" for c in counts.columns if c not in keys})
    neg = x[x["is_negative_event"] == 1]
    neg_counts = pd.crosstab([neg["candidate_day"], neg["symbol"]], neg["event_type"]).reset_index()
    neg_counts = neg_counts.rename(columns={c: f"negative_event_{c}_count" for c in neg_counts.columns if c not in keys})
    out = base.merge(counts, on=keys, how="left").merge(neg_counts, on=keys, how="left")
    for event in EVENT_TYPES:
        ec, nc = f"event_{event}_count", f"negative_event_{event}_count"
        if ec not in out:
            out[ec] = 0
        if nc not in out:
            out[nc] = 0
        out[ec] = pd.to_numeric(out[ec], errors="coerce").fillna(0)
        out[nc] = pd.to_numeric(out[nc], errors="coerce").fillna(0)
        out[f"event_{event}_share"] = out[ec] / out["news_count"].clip(lower=1)
        out[f"negative_event_{event}_share"] = out[nc] / out["news_count"].clip(lower=1)
    out = out.rename(columns={"candidate_day": "Date"})
    return out


def _rule_from_history(history: pd.DataFrame, symbol: str, event: str, condition: str, min_n: int) -> str:
    """Return a learned rule, treating an empty/no-schema history as insufficient data."""
    if history.empty:
        return "INSUFFICIENT"
    required = {"symbol", "condition", "event", "n", "mean_next_ret", "baseline_mean_next_ret", "delta_hit_2pct", "delta_hit_3pct", "delta_hit_5pct"}
    if not required.issubset(history.columns):
        return "INSUFFICIENT"
    h = history[
        (history["symbol"] == symbol)
        & (history["condition"] == condition)
        & (history["event"] == event)
    ]
    if h.empty:
        return "INSUFFICIENT"
    n = int(h["n"].iloc[0])
    if n < min_n:
        return "INSUFFICIENT"
    mean = float(h["mean_next_ret"].iloc[0])
    base = float(h["baseline_mean_next_ret"].iloc[0])
    d2 = float(h["delta_hit_2pct"].iloc[0])
    d3 = float(h["delta_hit_3pct"].iloc[0])
    d5 = float(h["delta_hit_5pct"].iloc[0])
    delta = mean - base
    if delta > 0 and d2 >= 0 and d3 >= 0 and d5 >= 0:
        return "FAVORABLE"
    if delta < 0 and d2 <= 0 and d3 <= 0 and d5 <= 0:
        return "AVOID"
    return "WATCH"


def _history_ranking(prior: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    baseline = prior.groupby("symbol").agg(
        baseline_mean_next_ret=("next_ret", "mean"),
        baseline_hit_2pct=("next_ret", lambda s: float((s >= .02).mean())),
        baseline_hit_3pct=("next_ret", lambda s: float((s >= .03).mean())),
        baseline_hit_5pct=("next_ret", lambda s: float((s >= .05).mean())),
    )
    for symbol, s in prior.groupby("symbol"):
        for event in EVENT_TYPES:
            neg_share = s[f"negative_event_{event}_share"].fillna(0)
            dom_share = pd.concat([s[f"event_{e}_share"].fillna(0).rename(e) for e in EVENT_TYPES], axis=1)
            primary = dom_share.idxmax(axis=1)
            dominant = (primary == event) & (dom_share.max(axis=1) > 0)
            for condition, mask in (
                ("event_negative", neg_share > 0),
                ("event_dominant_negative", dominant & (neg_share > 0)),
            ):
                q = s.loc[mask, "next_ret"].dropna()
                if q.empty:
                    n = 0
                    mean = float("nan")
                    h2 = h3 = h5 = float("nan")
                else:
                    n = len(q)
                    mean = float(q.mean())
                    h2, h3, h5 = float((q >= .02).mean()), float((q >= .03).mean()), float((q >= .05).mean())
                base = baseline.loc[symbol]
                rows.append({
                    "symbol": symbol, "event": event, "condition": condition, "n": n,
                    "mean_next_ret": mean,
                    "baseline_mean_next_ret": float(base.baseline_mean_next_ret),
                    "delta_mean": mean - float(base.baseline_mean_next_ret) if n else float("nan"),
                    "hit_2pct": h2, "baseline_hit_2pct": float(base.baseline_hit_2pct),
                    "delta_hit_2pct": h2 - float(base.baseline_hit_2pct) if n else float("nan"),
                    "hit_3pct": h3, "baseline_hit_3pct": float(base.baseline_hit_3pct),
                    "delta_hit_3pct": h3 - float(base.baseline_hit_3pct) if n else float("nan"),
                    "hit_5pct": h5, "baseline_hit_5pct": float(base.baseline_hit_5pct),
                    "delta_hit_5pct": h5 - float(base.baseline_hit_5pct) if n else float("nan"),
                })
    return pd.DataFrame(rows)


def _classify_current(row: pd.Series, rules: pd.DataFrame, min_n: int) -> tuple[str, list[str], list[str]]:
    favorable, avoid = [], []
    symbol = row["symbol"]
    dom = pd.Series({e: float(row.get(f"event_{e}_share", 0.0)) for e in EVENT_TYPES})
    primary = str(dom.idxmax()) if float(dom.max()) > 0 else None
    for event in EVENT_TYPES:
        neg_share = float(row.get(f"negative_event_{event}_share", 0.0))
        if neg_share <= 0:
            continue
        status = _rule_from_history(rules, symbol, event, "event_negative", min_n)
        if primary == event:
            dom_status = _rule_from_history(rules, symbol, event, "event_dominant_negative", min_n)
            if dom_status in {"FAVORABLE", "AVOID"}:
                status = dom_status
        if status == "FAVORABLE":
            favorable.append(event)
        elif status == "AVOID":
            avoid.append(event)
    if avoid:
        return "AVOID", favorable, avoid
    if favorable:
        return "FAVORABLE", favorable, avoid
    return "NEUTRAL", favorable, avoid


def walkforward_filter(oos: pd.DataFrame, raw: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    required = {"Date", "symbol", "next_ret", "v1_top20"}
    missing = required - set(oos.columns)
    if missing:
        raise ValueError(f"Missing OOS columns: {sorted(missing)}")
    x = oos.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x = x.dropna(subset=["Date", "symbol"]).sort_values(["Date", "symbol"]).reset_index(drop=True)
    events = _event_features(raw)
    x = x.merge(events, on=["Date", "symbol"], how="left")
    for event in EVENT_TYPES:
        for prefix in ("event", "negative_event"):
            col = f"{prefix}_{event}_share"
            if col not in x:
                x[col] = 0.0
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    out_rows = []
    dates = sorted(pd.to_datetime(x["Date"]).dropna().unique())
    for day in dates:
        day_ts = pd.Timestamp(day)
        prior = x[x["Date"] < day_ts].copy()
        test = x[x["Date"] == day_ts].copy()
        if test.empty:
            continue
        # Rules are learned exclusively from dates strictly before the test date.
        rules = _history_ranking(prior, min_n=min_n) if not prior.empty else pd.DataFrame()
        for idx, row in test.iterrows():
            if not bool(row.get("v1_top20", False)):
                continue
            status, favorable, avoid = _classify_current(row, rules, min_n)
            z = row.to_dict()
            z.update({
                "event_filter_status": status,
                "event_favorable": ";".join(favorable),
                "event_avoid": ";".join(avoid),
                "event_veto": status == "AVOID",
                "event_filter_keep": status != "AVOID",
            })
            out_rows.append(z)
    return pd.DataFrame(out_rows)


def _stats(x: pd.DataFrame, label: str) -> dict:
    y = x[x["next_ret"].notna()]
    return {
        "set": label,
        "n": len(y),
        "mean_next_ret": float(y["next_ret"].mean()) if len(y) else None,
        "median_next_ret": float(y["next_ret"].median()) if len(y) else None,
        "hit_2pct": float((y["next_ret"] >= .02).mean()) if len(y) else None,
        "hit_3pct": float((y["next_ret"] >= .03).mean()) if len(y) else None,
        "hit_5pct": float((y["next_ret"] >= .05).mean()) if len(y) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oos_predictions")
    ap.add_argument("raw_gkg")
    ap.add_argument("--out", default="results/news_v3_event_walkforward_filter.csv")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    oos = pd.read_csv(args.oos_predictions)
    raw = pd.read_csv(args.raw_gkg)
    scored = walkforward_filter(oos, raw, min_n=args.min_n)
    rows = []
    for symbol, s in scored.groupby("symbol", sort=True):
        rows.extend([
            _stats(s, f"{symbol}: V1 top20"),
            _stats(s[s.event_filter_keep], f"{symbol}: V1 top20 + walkforward event veto"),
            _stats(s[s.event_filter_status == "FAVORABLE"], f"{symbol}: V1 top20 + favorable event contexts"),
        ])
    report = pd.DataFrame(rows)
    if report.empty:
        raise RuntimeError("No V1 top20 OOS rows available for event-filter evaluation")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(p, index=False)
    scored.to_csv(p.with_name("news_v3_event_walkforward_filter_rows.csv"), index=False)
    print(report.to_string(index=False))
    print(f"VETOED ROWS: {int(scored.event_veto.sum())}")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
