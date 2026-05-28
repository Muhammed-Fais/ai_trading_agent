# XAUUSD ML Trading Agent

Production-oriented starter for a personal XAUUSD spot trading model. It predicts whether forward return clears a configurable threshold, then applies confidence and risk gates before emitting a trade signal.

This is not financial advice and it does not guarantee profitable trades. Treat every model as untrusted until it has survived out-of-sample testing, paper trading, and live monitoring.

## What It Builds

- Leakage-safe technical features from OHLCV candles
- Triple-class labels: `short`, `flat`, `long`
- Time-series cross-validation
- Ensemble model:
  - calibrated logistic regression
  - random forest
  - histogram gradient boosting
- Backtest with spread, slippage, and position sizing assumptions
- Prediction CLI with confidence thresholding
- Config-driven workflow

## Data

Use your broker or market data provider's XAUUSD spot candles where possible. Good production choices include:

- MetaTrader 5 broker history via `MetaTrader5.copy_rates_from`
- OANDA REST v20 candles/pricing APIs
- Paid institutional/retail data providers with survivable uptime and clear licensing

Expected CSV columns:

```csv
timestamp,open,high,low,close,volume
2024-01-02T00:00:00Z,2063.1,2064.2,2061.8,2062.7,1234
```

`volume` can be tick volume. If you do not have volume, set it to `0`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config/example.yaml config/local.yaml
```

Put historical candles at `data/xauusd_m15.csv`, then:

```bash
xauusd-agent train --config config/local.yaml
xauusd-agent backtest --config config/local.yaml
xauusd-agent predict --config config/local.yaml --input data/latest_xauusd_m15.csv
```

## Model Philosophy

For spot gold, raw next-price prediction is usually too noisy. This project predicts a tradable event instead:

- `long` when forward return is greater than estimated trading cost plus threshold
- `short` when forward return is less than negative threshold
- `flat` otherwise

That makes the model care about whether there may be enough move to pay spread and slippage.

## Production Checklist

- Use broker-native XAUUSD candles and current spreads
- Backtest with realistic execution costs
- Walk-forward validate by market regime
- Paper trade before risking money
- Add daily loss limits and max position limits at broker/order layer
- Log every prediction, order, fill, rejection, and model version
- Retrain on a schedule and compare challenger models before promotion

## Research Commands

```bash
.venv/bin/python scripts/prepare_broker_data.py --input data/gold_data/XAUUSD.csv --output data/xauusd_broker_1h.csv --timeframe 1h
.venv/bin/python scripts/walk_forward_report.py --config config/report_hourly_barrier.yaml --output reports/xauusd_hourly_regime_walk_forward.html
.venv/bin/python scripts/train_rl_policy.py --config config/report_hourly_barrier.yaml --output reports/xauusd_rl_policy_penalized.html
.venv/bin/python scripts/walk_forward_rl.py --config config/report_hourly_barrier.yaml --output reports/xauusd_rl_walk_forward.html
.venv/bin/python scripts/meta_label_report.py --config config/report_hourly_barrier.yaml --output reports/xauusd_meta_label_walk_forward.html
.venv/bin/python scripts/meta_label_report.py --config config/report_hourly_barrier.yaml --output reports/xauusd_meta_label_no_trend_follow.html --exclude-setup trend_follow
.venv/bin/python scripts/meta_label_report.py --config config/report_hourly_barrier.yaml --output reports/xauusd_meta_label_regime_no_trend_follow.html --exclude-setup trend_follow --use-regime-gate
.venv/bin/python scripts/fetch_macro_data.py --input data/xauusd_1h.csv --output data/xauusd_1h_macro.csv --macro-output data/macro_yahoo_daily.csv
.venv/bin/python scripts/meta_label_report.py --config config/report_hourly_macro_barrier.yaml --output reports/xauusd_meta_label_macro_no_trend_follow.html --exclude-setup trend_follow
```

## References

- MetaTrader 5 Python `copy_rates_from` stores bar times in UTC and returns OHLC/tick volume fields: https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
- OANDA v20 pricing supports candlestick data specifications for instruments and granularities: https://developer.oanda.com/rest-live-v20/pricing-df/
