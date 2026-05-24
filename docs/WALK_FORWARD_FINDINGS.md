# Walk-Forward Findings

Long-range hourly walk-forward testing does not support live deployment yet.

## Test Design

- Dataset: `data/xauusd_1h.csv`
- Coverage: `2004-06-11` to `2025-09-30`
- Training window: `7` years
- Validation window for parameter selection: `1` year
- Unseen test window: `2` years
- Test folds: `2014-2025`
- Execution: next-bar entry, ATR stop, ATR target, max holding time, spread and slippage

## Combined Result

- Total return: `-7.00%`
- Max drawdown: `-22.20%`
- Trades: `1,466`
- Win rate: `46.38%`
- Profit factor: `0.97`
- Average trade PnL: `-$0.48`

## Interpretation

The model is not robust across regimes. It had positive folds in `2014-2017` and `2024-2025`, but failed in `2018-2023`.

A low win rate is not automatically fatal when average winners are larger than average losers, but this walk-forward result shows the current edge is not stable enough. The profit factor below `1.0` means the strategy lost money after costs across the full walk-forward test.

## Recommendation

Do not trade this version live.

## Barrier Label Experiment

I also tested a stricter label design where the model learns whether an ATR take-profit is hit before an ATR stop-loss. This better matches the actual execution model, but it still did not pass walk-forward testing.

- Total return: `-6.15%`
- Max drawdown: `-19.05%`
- Trades: `1,134`
- Win rate: `44.97%`
- Profit factor: `0.97`

The result improved trade count and some individual folds, but did not solve regime instability. The `2018-2020` fold remained the main failure period.

## Regime Gate Experiment

I added a first-layer regime gate that chooses between:

- no regime filter
- trend plus normal volatility
- breakout plus higher volatility
- calmer trend regime

Each walk-forward fold selects the regime only from its validation window, then tests it on unseen data.

- Total return: `1.64%`
- Max drawdown: `-9.32%`
- Trades: `785`
- Win rate: `45.73%`
- Profit factor: `1.01`

This is a real improvement over the ungated barrier model. The worst fold improved materially:

- Barrier model `2018-2020`: `-14.97%`
- Regime-gated model `2018-2020`: `-3.12%`

However, the profit factor is still too close to break-even. This is useful risk reduction, not yet a production edge.

## Offline RL Experiment

I added a small offline reinforcement-learning research path:

- Environment: `XAUUSDTradingEnv`
- Actions: flat, long, short
- Policy: linear softmax policy
- Trainer: cross-entropy method
- Reward: PnL after costs, drawdown penalty, turnover penalty
- Train window: `2007-2016`
- Test window: `2016-2025`

The first unpenalized version failed, but the drawdown/turnover-penalized version improved:

- Total return: `1.43%`
- Max drawdown: `-7.29%`
- Trades: `1,423`
- Win rate: `47.86%`
- Profit factor: `1.04`

This is the best broad-holdout result so far, but it is still not production-grade. It shows the RL framing has promise, especially for position management, but it needs walk-forward RL validation before any live use.

### RL Walk-Forward

I then tested RL with rolling train/test folds.

- Total return: `-50.38%`
- Max drawdown: `-52.52%`
- Trades: `8,126`
- Win rate: `45.26%`
- Profit factor: `0.91`

This invalidates the one-split RL result. The simple linear CEM policy overtrades and does not generalize. Do not use this RL version live.

The useful lesson: RL may still help later, but it needs a stronger environment design before a heavier library like Ray RLlib is worth the complexity. The current priority should remain robust regime and direction filters, then a stricter execution model.

## Baseline Experiment

Transparent non-ML baselines also failed after costs:

- EMA trend: profit factor `0.86`
- Donchian breakout: profit factor `0.99`
- Hybrid trend-breakout: profit factor `0.98`

This suggests the current hourly edge is not strong enough as a standalone system under the current spread/slippage and ATR-exit assumptions.

## Meta-Label Experiment

I implemented the López de Prado-style idea of separating trade generation from trade filtering:

- Primary model: transparent candidate setups
  - trend pullback
  - trend follow
  - Donchian breakout
  - Donchian reversal
  - session momentum
  - volatility expansion
- Secondary model: ML accept/reject classifier
- Label: ATR target hit before ATR stop
- Threshold selected on validation window only

Walk-forward result:

- Total return: `-5.09%`
- Max drawdown: `-10.44%`
- Trades: `600`
- Win rate: `45.00%`
- Profit factor: `0.95`

This is better structured than raw prediction, but still not tradable. The `trend_follow` setup dominates the trade count, and the `2018-2020` fold remains the main failure period. The next useful improvement is per-setup model/threshold selection instead of one global meta-label threshold.

### Per-Setup Selection

I then moved from one global threshold to per-setup validation thresholds. This improved most individual folds, but the `trend_follow` setup still caused a severe `2018-2020` loss.

I added an exclusion experiment for `trend_follow`:

- Total return: `0.79%`
- Max drawdown: `-11.92%`
- Trades: `610`
- Win rate: `46.39%`
- Profit factor: `1.01`

This is the closest to breakeven so far and removes the catastrophic trend-follow failure, but it is still not strong enough for live trading. The second half of the test is healthier than the first half, suggesting the remaining setups may need a market-era/regime gate.

Next research should focus on:

- Regime filters to avoid weak periods
- Baseline comparison against simple trend/breakout systems
- Better labels based on stop/target hit probability, not only forward return
- Broker-native spread history
- M15 and H4 multi-timeframe features
- Walk-forward model selection with a holdout month untouched by tuning
