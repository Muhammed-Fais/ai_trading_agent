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

### Setup Regime Gate

I tested setup-specific regime gates on the no-`trend_follow` meta-label system.

- Total return: `-5.49%`
- Max drawdown: `-10.85%`
- Trades: `397`
- Win rate: `43.83%`
- Profit factor: `0.92`

This made the result worse than the ungated no-`trend_follow` version. The gate helped some later folds but damaged earlier folds too much. Current best research baseline remains the ungated no-`trend_follow` meta-label variant.

## Macro Feature Experiment

I added daily macro proxy features and forward-filled them into the H1 XAUUSD candles:

- VIX
- US 10Y yield proxy
- US dollar index proxy
- S&P 500 risk sentiment proxy

The macro-enriched no-`trend_follow` meta-label model is the first result to pass the rough research threshold:

- Total return: `24.80%`
- Max drawdown: `-8.31%`
- Trades: `784`
- Win rate: `49.49%`
- Profit factor: `1.21`

Fold-level behavior improved materially, including the previously weak `2018-2020` period. A setup-regime gate on top of macro features still hurt performance:

- Total return: `-5.63%`
- Profit factor: `0.85`

Optimizing validation thresholds for quality rather than raw return improved the trade profile:

- Total return: `22.26%`
- Max drawdown: `-7.44%`
- Trades: `621`
- Win rate: `50.89%`
- Profit factor: `1.25`

Current best paper-trading candidate: `xauusd_meta_label_macro_quality_no_trend_follow`.

## Higher-Timeframe Context Experiment

I added H4 and D1 context features to the hourly model. The features are shifted by one completed higher-timeframe bar before being forward-filled into H1 rows, so the H1 model only sees completed H4/D1 information.

The macro-enriched no-`trend_follow` meta-label model with H4/D1 context is now the strongest research candidate:

- Total return: `30.45%`
- Max drawdown: `-9.97%`
- Trades: `794`
- Win rate: `50.00%`
- Profit factor: `1.26`

This improved return versus the previous macro-quality variant and slightly improved profit factor, but the win rate is back near `50%` and the drawdown is larger. It still has weak early folds:

- Fold 1 return: `-2.86%`
- Fold 1 max drawdown: `-9.97%`
- Fold 1 profit factor: `0.95`
- Fold 2 return: `-0.11%`
- Fold 2 profit factor: `0.97`

That means the model is improving, but it is still not proven enough for live money.

## Stress Test

I then stress-tested the higher-timeframe candidate by rerunning the same walk-forward process under worse execution assumptions.

| Variant | Return | Max DD | Trades | Win Rate | Profit Factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base costs | `30.45%` | `-9.97%` | `794` | `50.00%` | `1.26` |
| 2x spread and slippage | `6.47%` | `-16.37%` | `839` | `49.11%` | `1.05` |
| 3x spread and slippage | `-5.82%` | `-21.00%` | `759` | `50.07%` | `0.95` |
| Half risk | `15.34%` | `-5.10%` | `859` | `49.71%` | `1.24` |

The useful conclusion is mixed: the base model is positive, and half-risk keeps drawdown near `5%`, but doubled costs reduce profit factor to `1.05` and tripled costs fail. This is not a live-trading approval. It is a candidate for paper trading with real broker spreads and execution logs.

Current best research candidate: `xauusd_meta_label_macro_htf_quality_no_trend_follow`.

The paper-trading artifact trained with the final two years as validation selected a stricter recent profile after the H4/D1 feature update:

- Validation return: `10.83%`
- Validation max drawdown: `-2.23%`
- Validation trades: `136`
- Validation win rate: `53.68%`
- Validation profit factor: `1.61`

This is encouraging but should be treated as a recent validation result, not live proof. It is suitable for paper logging, not live money yet.

This is still research, not live-ready, but it is now worth moving into paper-trading infrastructure after additional checks:

- broker-native spread/fill model
- untouched final holdout
- paper trading on the same execution path
- kill switches and monitoring

Next research should focus on:

- Regime filters to avoid weak periods
- Baseline comparison against simple trend/breakout systems
- Better labels based on stop/target hit probability, not only forward return
- Broker-native spread history
- M15 and H4 multi-timeframe features
- Walk-forward model selection with a holdout month untouched by tuning

## M5 VWAP Cross Experiment

I tested the user-specified M5 VWAP idea:

- After `6:00 AM IST`
- Wait for price to touch/cross IST daily VWAP
- If price closes above VWAP, use the crossing candle open as stop
- Enter after a bullish confirmation candle closes above VWAP without touching VWAP
- Target `1:3` reward/risk

The direct version was not tradable:

- Long-only return: `-3.09%`
- Long-only max drawdown: `-4.92%`
- Long-only trades: `233`
- Long-only profit factor: `0.88`

I then added a train/test sweep over practical filters:

- long-only
- stop taking new entries after a selected IST hour
- minimum close distance above VWAP
- confirmation must happen within a limited number of M5 bars
- reward/risk and max-hold variations

The train-selected filtered variant was:

- Long-only
- Trade cutoff: `16:00 IST`
- Minimum confirmation close distance from VWAP: `8 bps`
- Max confirmation wait: `3` M5 bars
- Target: `3R`
- Max hold: `24` M5 bars

Full sample:

- Total return: `3.38%`
- Max drawdown: `-0.87%`
- Trades: `65`
- Win rate: `49.23%`
- Profit factor: `1.72`

Chronological split:

- Train return: `2.05%`
- Train max drawdown: `-0.55%`
- Train trades: `38`
- Train profit factor: `1.74`
- Test return: `1.31%`
- Test max drawdown: `-0.87%`
- Test trades: `27`
- Test profit factor: `1.69`
- Test doubled-cost profit factor: `1.35`

This is a real improvement over the raw strategy, but the sample is too small for live use. It is a watchlist/paper-trading candidate only. The next requirement is more M5 history with real bid/ask spread, then a forward paper test.
