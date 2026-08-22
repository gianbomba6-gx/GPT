# V1 initial backtest — STLAM

Dataset: 4,226 daily observations supplied by the user.

Signal definition: previous/current session return <= -2%; target is next-session close return >= +3%.

## Baseline event study
- 708 qualifying down days.
- Mean next-day close return: +0.078%.
- Median next-day close return: +0.141%.
- Probability next-day close return >= +2%: 23.16%.
- Probability next-day close return >= +3%: 13.28%.
- Probability next-day close return >= +5%: 4.38%.
- Mean next-day high relative to signal-day close: +2.04%.

## Walk-forward model
An expanding-window HistGradientBoostingClassifier was tested with technical/price-volume features and a +3% next-day close target. The top 20/10/5% model-score subsets among >=2% down days had respectively approximately 19.8%, 18.9%, and 25.9% hit rates for +3% next-day close returns in this initial run.

These results are NOT yet evidence of a tradable edge: the model is uncalibrated, the feature set is deliberately generic, transaction costs/slippage are not modeled, and the news component has not yet been fitted to a timestamped historical news dataset. The next research gate is a true ablation study: technical-only vs technical+news, both evaluated with identical walk-forward periods and leakage controls.

## Important limitation
The current daily files do not contain intraday timestamps, so news must be merged conservatively. Same-day news cannot be assumed to have been known before the signal. The V1 news interface therefore uses prior-day information unless intraday market/news timestamps are supplied.
