# Production Plan

Current status: research-grade model with broker-data preprocessing, richer features, time-series validation, and managed-trade backtesting.

It is not ready for live money until the remaining production controls are added and paper-tested.

## Current Strategy

- Timeframe: H1 candles resampled from broker M1 export
- Entry: next bar open after model signal
- Exit: ATR stop, ATR take-profit, or max holding time
- Risk: fixed fraction of equity per trade, capped notional
- Positioning: one open trade at a time
- Model: soft-voting ensemble over linear, random forest, and gradient boosting classifiers
- Regime gate: required before ML entries; current research version reduces weak-regime losses but is not yet strong enough for live use

## Minimum Live Requirements

- Broker adapter for live candles, spreads, order placement, fills, rejections, and account state
- Paper trading mode that uses the same code path as live mode
- Kill switches:
  - max daily loss
  - max weekly loss
  - max consecutive losses
  - max spread
  - no-trade window around high-impact events
- Persistent logging for every candle, feature vector, prediction, order, fill, and exit
- Model registry with immutable artifact versions and config snapshots
- Daily reconciliation against broker account history
- Monitoring alerts for missing candles, stale prices, order failures, and unusual drawdown

## Research Needed Before Live Use

- Walk-forward parameter selection instead of manually choosing the best holdout settings
- Broker-native historical spread and commission model
- Out-of-sample test over more than one market regime
- M15 and H1 comparison using the same execution simulator
- Baseline comparison:
  - EMA trend + ATR exit
  - Donchian breakout + ATR exit
  - always-flat model
- News/event filter for CPI, NFP, FOMC, rate decisions, and major geopolitical shock periods

## Promotion Rule

Do not promote a model to live unless it passes all of these:

- Positive net return after realistic spread/slippage
- Profit factor above `1.20`
- Max drawdown below your personal hard limit
- At least `300` out-of-sample trades or multiple clean walk-forward windows
- No single month accounts for most of the profit
- Paper trading matches backtest behavior closely for at least `4` weeks
