"""Validate the production-style dynamic top10 probability cutoff OOS.

For each test year, fit the pooled target_3 model on all prior years, compute the
90th percentile of training probabilities among rebound candidates (ret <= -2%),
and apply that single cutoff to the full test year. The resulting mask must match
the ranking-based top10 selection computed from the same test scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

from rebound_model import BASE_FEATURES, load_yahoo_ohlcv
from live_alert import add_market_regime, REGIME_FEATURES

FEATURES = BASE_FEATURES + REGIME_FEATURES
COST = 0.002
SEED = 42
TOP10_Q = 0.90


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close": "Ultimo", "Open": "Apertura", "High": "Massimo", "Low": "Minimo", "Volume": "Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize().astype("datetime64[ns]")
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def fit(train: pd.DataFrame):
    tr = train.dropna(subset=FEATURES + ["target_3"]).copy()
    if len(tr) < 500 or tr["target_3"].nunique() < 2:
        return None, tr
    model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=2,
        random_state=SEED,
    )
    model.fit(tr[FEATURES], tr["target_3"].astype(int))
    return model, tr


def cutoff_from_training(model, train: pd.DataFrame) -> tuple[float, int]:
    candidates = train[train["ret"] <= -0.02].dropna(subset=FEATURES).copy()
    if len(candidates) < 20:
        return float("nan"), len(candidates)
    p = model.predict_proba(candidates[FEATURES])[:, 1]
    return float(np.quantile(p, TOP10_Q)), len(candidates)


def build(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    yearly = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        model, tr = fit(train)
        if model is None:
            continue
        cutoff, n_train_candidates = cutoff_from_training(model, tr)
        if not np.isfinite(cutoff):
            continue
        te = test.dropna(subset=FEATURES).copy()
        if te.empty:
            continue
        te["score"] = model.predict_proba(te[FEATURES])[:, 1]
        te["baseline_signal"] = te["ret"] <= -0.02
        te["selected_dynamic"] = te["baseline_signal"] & (te["score"] >= cutoff)
        te["selected_rank_top10"] = False
        for symbol, x in te.groupby("symbol", sort=False):
            idx = x.index[x["baseline_signal"]]
            if len(idx):
                k = max(1, int(np.ceil(len(idx) * 0.10)))
                chosen = x.loc[idx].nlargest(k, "score").index
                te.loc[chosen, "selected_rank_top10"] = True
        # Candidate universe is pooled; therefore the ranking reference must also be pooled.
        idx = te.index[te["baseline_signal"]]
        if len(idx):
            k = max(1, int(np.ceil(len(idx) * 0.10)))
            chosen = te.loc[idx].nlargest(k, "score").index
            te["selected_rank_top10"] = False
            te.loc[chosen, "selected_rank_top10"] = True
        te["year"] = int(year)
        te["dynamic_cutoff"] = cutoff
        te["n_train_candidates"] = int(n_train_candidates)
        pieces.append(te[["Date", "symbol", "next_ret", "target_3", "baseline_signal", "score", "selected_dynamic", "selected_rank_top10", "year", "dynamic_cutoff", "n_train_candidates"]])
        cand = te[te["baseline_signal"] & te["next_ret"].notna()].copy()
        yearly.append({
            "year": int(year),
            "n_train_candidates": int(n_train_candidates),
            "dynamic_cutoff": cutoff,
            "n_test_candidates": len(cand),
            "n_dynamic_selected": int(cand["selected_dynamic"].sum()),
            "n_rank_top10": int(cand["selected_rank_top10"].sum()),
            "selection_rate_dynamic": float(cand["selected_dynamic"].mean()) if len(cand) else np.nan,
            "selection_masks_match": bool((cand["selected_dynamic"] == cand["selected_rank_top10"]).all()) if len(cand) else True,
        })
    if not pieces:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(yearly)


def bootstrap_mean(values: np.ndarray, mask: np.ndarray, n_boot: int):
    valid = np.isfinite(values)
    values, mask = values[valid], mask[valid]
    if len(values) == 0 or mask.sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[mask].mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    s, m = values[idx], mask[idx]
    c = m.sum(axis=1)
    ok = c > 0
    out = np.full(n_boot, np.nan)
    out[ok] = (s * m).sum(axis=1)[ok] / c[ok]
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return observed, float(lo), float(hi)


def summarize(scored: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    x = scored[scored["baseline_signal"] & scored["next_ret"].notna()].copy()
    for strategy, col in (("dynamic_top10", "selected_dynamic"), ("rank_top10", "selected_rank_top10")):
        vals = x["next_ret"].to_numpy(float)
        mask = x[col].to_numpy(bool)
        mean, lo, hi = bootstrap_mean(vals, mask, n_boot)
        n = int(mask.sum())
        hit = float(x.loc[mask, "target_3"].mean()) if n else np.nan
        rows.append({
            "strategy": strategy,
            "n_baseline": len(x),
            "n_selected": n,
            "selection_rate": n / len(x) if len(x) else np.nan,
            "mean_selected_gross": mean,
            "mean_selected_net": mean - COST if n else np.nan,
            "hit3": hit,
            "ci_low": lo,
            "ci_high": hi,
            "cost_bps": 20.0,
            "status": "OK" if n else "NO_SELECTED_CASES",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_dynamic_top10_validation.csv")
    ap.add_argument("--out-yearly", default="results/v1_dynamic_top10_validation_yearly.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise SystemExit("Invalid bootstrap count")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "tickers.json").read_text())
    frames, benchmark = {}, None
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
    scored, yearly = build(data)
    if scored.empty or yearly.empty:
        raise RuntimeError("No OOS dynamic top10 rows generated")
    if not yearly["selection_masks_match"].all():
        raise SystemExit("Dynamic cutoff does not reproduce top10 ranking")
    report = summarize(scored, args.n_boot)
    ok = report[report.status.eq("OK")]
    nums = ["n_baseline", "n_selected", "selection_rate", "mean_selected_gross", "mean_selected_net", "hit3", "ci_low", "ci_high", "cost_bps"]
    if not ok.empty and not np.isfinite(ok[nums].to_numpy(float)).all():
        raise SystemExit("Invalid dynamic top10 numeric result")
    if (yearly["dynamic_cutoff"] <= 0).any() or (yearly["dynamic_cutoff"] >= 1).any():
        raise SystemExit("Invalid dynamic top10 cutoff")
    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(root / args.out, index=False)
    yearly.to_csv(root / args.out_yearly, index=False)
    print("V1 DYNAMIC TOP10 VALIDATION")
    print(report.to_string(index=False))
    print("V1 DYNAMIC TOP10 YEARLY")
    print(yearly.to_string(index=False))
    print(f"Saved {root / args.out}")
    print(f"Saved {root / args.out_yearly}")
    print("V1 DYNAMIC TOP10 VALIDATION PASS")

if __name__ == "__main__":
    main()
