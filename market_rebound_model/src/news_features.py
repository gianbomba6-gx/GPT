"""News feature schema and deterministic helpers for the rebound model.

V2 deliberately keeps news ingestion separate from the model. Historical news
must be time-bounded to the signal cutoff to avoid look-ahead bias.
"""
from __future__ import annotations
import pandas as pd

NEWS_COLUMNS = [
    "Date", "symbol", "news_sentiment", "news_intensity", "news_relevance",
    "news_novelty", "news_count", "news_category"
]

CATEGORIES = [
    "market", "earnings", "guidance", "analyst", "regulatory", "legal",
    "m_and_a", "macro", "geopolitical", "product", "insider", "other"
]

def empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)

def aggregate_news(news: pd.DataFrame, signal_dates: pd.Series) -> pd.DataFrame:
    """Aggregate news available by each signal date.

    A row dated D may only use articles timestamped no later than D's market
    close. The caller is responsible for supplying publication timestamps and
    applying the exact exchange/news cutoff appropriate to the market.
    """
    if news.empty:
        return pd.DataFrame(columns=["Date", "symbol"] + NEWS_COLUMNS[2:6] + ["news_count"])
    n = news.copy()
    n["Date"] = pd.to_datetime(n["Date"], errors="coerce").dt.normalize()
    n = n.dropna(subset=["Date", "symbol"])
    agg = (n.groupby(["Date", "symbol"], as_index=False)
        .agg(news_sentiment=("news_sentiment", "mean"),
             news_intensity=("news_intensity", "mean"),
             news_relevance=("news_relevance", "mean"),
             news_novelty=("news_novelty", "mean"),
             news_count=("symbol", "size"),
             news_category=("news_category", lambda x: ",".join(sorted(set(x.dropna().astype(str)))))))
    return agg
