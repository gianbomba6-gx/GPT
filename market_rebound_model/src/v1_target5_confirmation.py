"""Confirmatory OOS comparison of V1 target_3 vs target_5.

Keeps the production V1 feature set, pooled model, -2% rebound gate,
expanding annual walk-forward training and training-derived top-20% threshold.
Adds bootstrap confidence intervals, per-symbol results and annual stability.
No production code is modified by this script.
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
VARIANTS = ("target_3", "target_5")
COST = 0.002  # 20 bps
SEED = 42


def fit_model(train: pd.DataFrame, features_target: str) -> tuple[HistGradientBoostingClassifier | None, pd.DataFrame]:
    tr = train.dropna(subset=FEATURES + [features_target]).copy()
    if len(tr) < 500 or tr[features_target].nunique() < 2:
        return None, tr
    model = HistGradientBoostingClassifier(
        max_iter=250,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=2,
        random_state=SEED,
    )
    model.fit(tr[FEATURES], tr[features_target].astype(int))
    return model, tr


def training_threshold(train: pd.DataFrame, target: str) -> float:
    model, tr = fit_model(train, target)
    if model is None:
        return float("nan")
    base = tr["ret"] <= -0.02
    if int(base.sum()) < 20:
        return float("nan")
    p = model.predict_proba(tr[FEATURES])[:, 1]
    return float(pd.Series(p, index=tr.index)[base].quantile(0.80))


def walk_forward(data: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in sorted(data["Date"].dt.year.unique()):
        train = data[data["Date"].dt.year < year].copy()
        test = data[data["Date"].dt.year == year].copy()
        if len(train) < 500 or test.empty:
            continue
        test = test.copy()
        test["baseline_signal"] = test["ret"] <= -0.02
        for target in VARIANTS:
            model, _ = fit_model(train, target)
            if model is None:
                continue
            thr = training_threshold(train, target)
            if not np.isfinite(thr):
                continue
            te = test.dropna(subset=FEATURES).copy()
            if te.empty:
                continue
            te[f"score_{target}"] = model.predict_proba(te[FEATURES])[:, 1]
            te[f"selected_{target}"] = te["baseline_signal"] & (te[f"score_{target}"] >= thr)
            cols = ["Date", "symbol", "ret", "next_ret", "baseline_signal", f"score_{target}", f"selected_{target}"]
            piece = te[cols].copy()
            piece["year"] = year
            parts.append(piece)
    if not parts:
        return pd.DataFrame()
    # Merge the per-target OOS decisions on the same test observations.
    out = parts[0]
    for p in parts[1:]:
        keep = ["Date", "symbol", "ret", "next_ret", "baseline_signal", "year", f"score_{p.columns[-2].removeprefix('selected_')}", p.columns[-1]]
        out = out.merge(p[keep], on=["Date", "symbol", "ret", "next_ret", "baseline_signal", "year"], how="outer")
    return out.sort_values(["symbol", "Date"]).reset_index(drop=True)


def bootstrap_delta(values: np.ndarray, selected: np.ndarray, n_boot: int) -> tuple[float, float, float]:
    valid = np.isfinite(values)
    values = values[valid]
    selected = selected[valid]
    if len(values) == 0 or selected.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    observed = float(values[selected].mean() - values.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    masks = selected[idx]
    counts = masks.sum(axis=1)
    valid_boot = counts > 0
    deltas = np.full(n_boot, np.nan, dtype=float)
    deltas[valid_boot] = (sample * masks).sum(axis=1)[valid_boot] / counts[valid_boot] - sample.mean(axis=1)[valid_boot]
    ci = np.nanpercentile(deltas, [2.5, 97.5])
    return observed, float(ci[0]), float(ci[1])


def summarize(df: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for symbol in ["ALL"] + sorted(df["symbol"].dropna().unique().tolist()):
        x = df if symbol == "ALL" else df[df["symbol"] == symbol]
        x = x[x["next_ret"].notna() & x["baseline_signal"]].copy()
        if x.empty:
            continue
        base_mean = float(x["next_ret"].mean())
        for target in VARIANTS:
            col = f"selected_{target}"
            if col not in x.columns:
                continue
            sel = x[col].fillna(False).astype(bool).to_numpy()
            vals = x["next_ret"].to_numpy(float)
            delta, lo, hi = bootstrap_delta(vals, sel, n_boot)
            n_sel = int(sel.sum())
            mean_sel = float(vals[sel].mean()) if n_sel else float("nan")
            rows.append({
                "symbol": symbol,
                "variant": target,
                "n_baseline": len(x),
                "n_selected": n_sel,
                "mean_baseline": base_mean,
                "mean_selected_gross": mean_sel,
                "mean_selected_net": mean_sel - COST if n_sel else float("nan"),
                "delta_mean_gross": delta,
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "delta_mean_net": delta - COST if np.isfinite(delta) else float("nan"),
                "cost_bps": 20.0,
                "status": "OK" if n_sel else "NO_SELECTED_CASES",
            })
    report = pd.DataFrame(rows)

    stab = []
    for (symbol, year), x in df.groupby(["symbol", "year"], sort=True):
        x = x[x["next_ret"].notna() & x["baseline_signal"]].copy()
        if x.empty:
            continue
        base_mean = float(x["next_ret"].mean())
        for target in VARIANTS:
            sel = x[f"selected_{target}"].fillna(False).astype(bool)
            n_sel = int(sel.sum())
            mean_sel = float(x.loc[sel, "next_ret"].mean()) if n_sel else float("nan")
            stab.append({
                "symbol": symbol,
                "year": int(year),
                "variant": target,
                "n_baseline": len(x),
                "n_selected": n_sel,
                "mean_baseline": base_mean,
                "mean_selected_gross": mean_sel,
                "delta_mean": mean_sel - base_mean if n_sel else float("nan"),
                "status": "OK" if n_sel else "NO_SELECTED_CASES",
            })
    return report, pd.DataFrame(stab)


def fetch(symbol: str) -> pd.DataFrame:
    x = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if x.empty:
        raise RuntimeError(f"No Yahoo data for {symbol}")
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    x = x.rename(columns={"Close": "Ultimo", "Open": "Apertura", "High": "Massimo", "Low": "Minimo", "Volume": "Vol."})
    x["Date"] = pd.to_datetime(x.index).tz_localize(None).normalize()
    return x.reset_index(drop=True)[["Date", "Ultimo", "Apertura", "Massimo", "Minimo", "Vol."]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/v1_target5_confirmation.csv")
    ap.add_argument("--out-stability", default="results/v1_target5_confirmation_stability.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    if args.n_boot < 1000:
        raise SystemExit("Invalid bootstrap count")

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "tickers.json").read_text())
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
    scored = walk_forward(data)
    if scored.empty:
        raise RuntimeError("No OOS confirmation rows generated")
    report, stability = summarize(scored, args.n_boot)
    if report.empty:
        raise RuntimeError("No confirmation summary generated")
    numeric = ["n_baseline", "n_selected", "mean_baseline", "mean_selected_gross", "mean_selected_net", "delta_mean_gross", "delta_ci_low", "delta_ci_high", "delta_mean_net", "cost_bps"]
    ok = report[report["status"].eq("OK")]
    if not ok.empty and not np.isfinite(ok[numeric].to_numpy(float)).all():
        raise SystemExit("Invalid Target_5 confirmation result")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(root / args.out, index=False)
    stability.to_csv(root / args.out_stability, index=False)
    print("V1 TARGET5 CONFIRMATION")
    print(report.to_string(index=False))
    print("V1 TARGET5 STABILITY")
    print(stability.to_string(index=False))
    print(f"Saved {root / args.out}")
    print(f"Saved {root / args.out_stability}")
    print("V1 TARGET5 CONFIRMATION PASS")


if __name__ == "__main__":
    main()
