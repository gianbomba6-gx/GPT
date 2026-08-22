# V2 News Dataset

## Objective
Build a historical dataset linking a signal-day selloff to the news that was publicly available by the signal cutoff, then measure whether news improves next-session rebound prediction.

## Required raw fields
- `published_at` (UTC timestamp)
- `symbol`
- `title`
- `source`
- `url`
- `summary`

## Derived fields
- `news_sentiment`: -1..1
- `news_intensity`: 0..1
- `news_relevance`: 0..1
- `news_novelty`: 0..1
- `news_count`
- `news_category`

## Leakage rule
For a prediction made after the signal session, only articles with `published_at` at or before the signal cutoff are eligible. Articles first published after the cutoff are excluded even if they discuss the same event retrospectively.

## Validation design
Compare out-of-sample:
1. Technical only
2. Technical + market regime
3. Technical + market regime + news

The same walk-forward dates and signal definitions must be used for all three models. News is an additional information set, not a replacement for price/volume features.

## Initial universe
STLAM.MI, SPCX, NVDA, TSLA.

Once V2 is validated, expand to FTSE MIB + NASDAQ-100.
