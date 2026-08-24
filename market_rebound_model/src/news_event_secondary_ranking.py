"""OOS secondary news ranking for existing V1 top20 candidates.

The V1 candidate set is untouched.  The news score is learned strictly from
prior V1 top20 observations and is used only to rank/diagnose candidates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = (
    "earnings",
    "guidance",
    "analyst",
    "regulatory",
    "ma",
    "product",
    "macro",
)
MIN_N = 20
SHRINK_K = 50.0


def _event_score_rules(prior: pd.DataFrame, min_n: int = MIN_N, shrink_k: float = SHRINK_K) -> dict[tuple[str, str], float]:
    rules: dict[tuple[str, str], float] = {}
    for (symbol, event), s in prior.groupby(["symbol", "event_type"], sort=False):
        if len(s) < min_n:
            continue
        q = s["next_ret"].dropna()
        if len(q) < min_n:
            continue
        baseline = prior.loc[prior["symbol"] == symbol, "next_ret"].dropna()
        if baseline.empty:
            continue
        delta = float(q.mean() - baseline.mean())
        shrink = len(q) / (len(q) + shrink_k)
        rules[(str(symbol), str(event))] = float(delta * shrink)
    return rules


def _score_row(row: pd.Series, rules: dict[tuple[str, str], float]) -> float:
    symbol = str(row["symbol"])
    score = 0.0
    for event in EVENT_TYPES:
        share = float(row.get(f"negative_event_{event}_share", 0.0) or 0.0)
        if share <= 0:
            continue
        score += share * rules.get((symbol, event), 0.0)
    return float(score)


def score_oos(rows: pd.DataFrame, min_n: int = MIN_N, shrink_k: float = SHRINK_K) -> pd.DataFrame:
    required = {"Date", "symbol", "next_ret", "v1_top20"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for event in EVENT_TYPES:
        for prefix in ("event", "negative_event"):
            col = f"{prefix}_{event}_share"
            if col not in rows.columns:
                rows = rows.copy()
                rows[col] = 0.0

    x = rows.copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["next_ret"] = pd.to_numeric(x["next_ret"], errors="coerce")
    x["v1_top20"] = x["v1_top20"].fillna(False).astype(bool)
    x = x.dropna(subset=["Date", "symbol"]).sort_values(["Date", "symbol"]).reset_index(drop=True)
    candidates = x[x["v1_top20"]].copy()

    out_parts: list[pd.DataFrame] = []
    for day in sorted(candidates["Date"].unique()):
        day_ts = pd.Timestamp(day)
        prior = candidates[candidates["Date"] < day_ts].copy()
        test = candidates[candidates["Date"] == day_ts].copy()
        if test.empty:
            continue
        rules = _event_score_rules(prior, min_n=min_n, shrink_k=shrink_k) if not prior.empty else {}
        test["news_rank_score"] = test.apply(lambda r: _score_row(r, rules), axis=1)
        test["news_rank_known_events"] = test.apply(
            lambda r: sum(
                1 for e in EVENT_TYPES
                if float(r.get(f"negative_event_{e}_share", 0.0) or 0.0) > 0
                and (str(r["symbol"]), e) in rules
            ),
            axis=1,
        )
        test["news_daily_rank"] = test.groupby("Date")["news_rank_score"].rank(method="first", ascending=False).astype(int)
        test["news_daily_candidates"] = test.groupby("Date")["symbol"].transform("size").astype(int)
        out_parts.append(test)
    if not out_parts:
        return pd.DataFrame()
    return pd.concat(out_parts, ignore_index=True)


def _spearman(x: pd.DataFrame) -> float:
    y = x[["news_rank_score", "next_ret"]].dropna()
    if len(y) < 3 or y["news_rank_score"].nunique() < 2 or y["next_ret"].nunique() < 2:
        return float("nan")
    return float(y["news_rank_score"].rank().corr(y["next_ret"].rank()))


def _stats(x: pd.DataFrame, label: str) -> dict:
    y = x["next_ret"].dropna()
    return {
        "set": label,
        "n": len(y),
        "mean_next_ret": float(y.mean()) if len(y) else np.nan,
        "median_next_ret": float(y.median()) if len(y) else np.nan,
        "hit_2pct": float((y >= .02).mean()) if len(y) else np.nan,
        "hit_3pct": float((y >= .03).mean()) if len(y) else np.nan,
        "hit_5pct": float((y >= .05).mean()) if len(y) else np.nan,
    }


def build_report(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = []
    for symbol, s in scored.groupby("symbol", sort=True):
        s = s.sort_values("news_rank_score")
        reports.append({
            "symbol": symbol,
            "n": len(s),
            "spearman_score_next_ret": _spearman(s),
            "multi_candidate_days": int((s["news_daily_candidates"] > 1).sum()),
            "rank1_days": int(((s["news_daily_candidates"] > 1) & (s["news_daily_rank"] == 1)).sum()),
            "known_event_rate": float((s["news_rank_known_events"] > 0).mean()) if len(s) else np.nan,
        })

    if len(scored):
        scored = scored.copy()
        scored["score_q"] = pd.qcut(
            scored["news_rank_score"].rank(method="first"),
            q=4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        )
        q = (
            scored.groupby("score_q", observed=False)["next_ret"]
            .agg(n="count", mean_next_ret="mean")
            .reset_index()
        )
    else:
        q = pd.DataFrame(columns=["score_q", "n", "mean_next_ret"])

    return pd.DataFrame(reports), q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_csv")
    ap.add_argument("--out", default="results/news_v3_secondary_ranking.csv")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--shrink-k", type=float, default=SHRINK_K)
    args = ap.parse_args()
    rows = pd.read_csv(args.rows_csv)
    scored = score_oos(rows, min_n=args.min_n, shrink_k=args.shrink_k)
    if scored.empty:
        raise SystemExit("No V1 top20 rows available for secondary news ranking")
    report, quartiles = build_report(scored)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(p, index=False)
    quartiles.to_csv(p.with_name("news_v3_secondary_ranking_quartiles.csv"), index=False)
    scored.to_csv(p.with_name("news_v3_secondary_ranking_rows.csv"), index=False)
    print("SYMBOL REPORT")
    print(report.to_string(index=False))
    print("SCORE QUARTILES")
    print(quartiles.to_string(index=False))
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
