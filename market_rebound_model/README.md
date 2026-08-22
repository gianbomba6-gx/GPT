# Market Rebound Model

Research framework for forecasting next-session rebounds after large down moves using historical OHLCV data plus timestamp-aligned economic/news features.

## Research design
- Strictly causal features: only information available by the signal timestamp.
- Targets: next-day close return, next-day high return, and threshold rebound classification.
- Features: drawdown, multi-horizon returns, intraday range, close location, gap, volume shock, volatility, market/sector relative strength, and news/event sentiment.
- Validation: walk-forward / expanding-window evaluation; no random train/test split.
- Metrics: precision@signal, hit rate, average next-day return, max adverse excursion, calibration, and strategy-level return after costs.

## News integration
News should be stored with publication timestamp, source, entity/ticker, event category, polarity, surprise/novelty, and relevance. The key anti-leakage rule is to use only articles whose publication timestamp precedes the signal time. Intraday timestamps are preferable to end-of-day article dates.

## Current data insight
STLAM has a long history (4,226 daily rows), while SPCX has only 49 rows in the supplied file. SPCX is therefore useful for a recent event study but is far too small for an independent robust ML model. It should be evaluated jointly with regime-aware features or kept as an out-of-sample case study.

## Example recent event
For STLAM, the 19 Aug 2026 rebound coincided with easing U.S.-Canada tariff tensions; Reuters reported a proposed reduction in auto tariffs and a temporary pause in new Canadian tariffs, while ANSA reported the stock rebounded about 6% with doubled trading volume. This is a useful example of a potentially news-driven rebound regime.

For SPCX, the 20 Aug 2026 decline coincided with a second post-IPO lockup/share-unlock event; the supplied data show -4.05% on 119.8M shares, followed by +2.22% on 21 Aug. This illustrates why event-type and supply-overhang features may be valuable.
