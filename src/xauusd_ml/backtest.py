from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ml.features import build_features
from xauusd_ml.model import signal_from_probabilities
from xauusd_ml.regime import signal_allowed


def run_backtest(
    df: pd.DataFrame,
    model,
    feature_names: list[str],
    horizon_bars: int,
    min_confidence: float,
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    start_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    features = build_features(df).reindex(columns=feature_names).dropna()
    close = df["close"].reindex(features.index)
    probabilities = model.predict_proba(features)

    rows = []
    equity = initial_equity
    cost = (spread_bps + slippage_bps) / 10_000

    for idx, probs in zip(features.index[:-horizon_bars], probabilities[:-horizon_bars], strict=True):
        if start_at is not None and idx < start_at:
            continue
        signal = signal_from_probabilities(model.classes_, probs, min_confidence)
        direction = {"long": 1, "short": -1, "flat": 0}[str(signal["signal"])]
        entry = float(close.loc[idx])
        exit_price = float(close.shift(-horizon_bars).loc[idx])
        gross_return = direction * (exit_price / entry - 1)
        net_return = gross_return - abs(direction) * cost
        notional = min(equity * risk_per_trade / max(cost, 0.0001), max_position_notional)
        pnl = notional * net_return
        equity += pnl
        rows.append(
            {
                "timestamp": idx,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "entry": entry,
                "exit": exit_price,
                "net_return": net_return,
                "notional": notional,
                "pnl": pnl,
                "equity": equity,
            }
        )

    result = pd.DataFrame(rows).set_index("timestamp")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result["trade"] = result["signal"] != "flat"
    result["win"] = np.where(result["trade"], result["pnl"] > 0, np.nan)
    return result


def summarize_backtest(result: pd.DataFrame, initial_equity: float | None = None) -> dict[str, float]:
    trades = result[result["trade"]]
    starting_equity = initial_equity if initial_equity is not None else float(result["equity"].iloc[0])
    gross_profit = float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if len(trades) else 0.0
    gross_loss = float(trades.loc[trades["pnl"] < 0, "pnl"].sum()) if len(trades) else 0.0
    return {
        "ending_equity": float(result["equity"].iloc[-1]),
        "total_return": float(result["equity"].iloc[-1] / starting_equity - 1),
        "max_drawdown": float(result["drawdown"].min()),
        "trades": float(len(trades)),
        "win_rate": float(trades["win"].mean()) if len(trades) else 0.0,
        "avg_trade_pnl": float(trades["pnl"].mean()) if len(trades) else 0.0,
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def run_managed_backtest(
    df: pd.DataFrame,
    model,
    feature_names: list[str],
    min_confidence: float,
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
    start_at: pd.Timestamp | None = None,
    allow_overlapping_trades: bool = False,
    regime_mask: pd.Series | None = None,
    confluence_direction: pd.Series | None = None,
) -> pd.DataFrame:
    all_features = build_features(df)
    features = all_features.reindex(columns=feature_names).dropna()
    probabilities = model.predict_proba(features)
    probability_by_time = dict(zip(features.index, probabilities, strict=True))

    equity = initial_equity
    cost = (spread_bps + slippage_bps) / 10_000
    next_available_idx = 0
    rows = []
    index = df.index

    for feature_time in features.index:
        bar_idx = index.get_indexer([feature_time])[0]
        if bar_idx < 0 or bar_idx + 1 >= len(index):
            continue
        if start_at is not None and feature_time < start_at:
            continue
        if regime_mask is not None and not bool(regime_mask.reindex([feature_time]).fillna(False).iloc[0]):
            continue
        if not allow_overlapping_trades and bar_idx < next_available_idx:
            continue

        signal = signal_from_probabilities(
            model.classes_,
            probability_by_time[feature_time],
            min_confidence,
        )
        if confluence_direction is not None:
            confluence_value = int(
                confluence_direction.reindex([feature_time]).fillna(0).iloc[0]
            )
            if not signal_allowed(str(signal["signal"]), confluence_value):
                continue
        direction = {"long": 1, "short": -1, "flat": 0}[str(signal["signal"])]
        if direction == 0:
            rows.append(
                {
                    "timestamp": feature_time,
                    "signal": "flat",
                    "confidence": signal["confidence"],
                    "entry": np.nan,
                    "exit": np.nan,
                    "exit_reason": "no_trade",
                    "net_return": 0.0,
                    "notional": 0.0,
                    "pnl": 0.0,
                    "equity": equity,
                }
            )
            continue

        entry_idx = bar_idx + 1
        entry_time = index[entry_idx]
        entry = float(df["open"].iloc[entry_idx])
        atr_pct = float(all_features["atr_pct_14"].loc[feature_time])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue

        stop_pct = max(stop_atr_multiple * atr_pct, cost * 2)
        target_pct = max(take_profit_atr_multiple * atr_pct, cost * 2)
        stop_price = entry * (1 - direction * stop_pct)
        target_price = entry * (1 + direction * target_pct)
        exit_idx = min(entry_idx + max_hold_bars, len(index) - 1)
        exit_price = float(df["close"].iloc[exit_idx])
        exit_reason = "time"

        for candidate_idx in range(entry_idx, exit_idx + 1):
            high = float(df["high"].iloc[candidate_idx])
            low = float(df["low"].iloc[candidate_idx])
            if direction == 1:
                hit_stop = low <= stop_price
                hit_target = high >= target_price
            else:
                hit_stop = high >= stop_price
                hit_target = low <= target_price

            if hit_stop and hit_target:
                exit_idx = candidate_idx
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar"
                break
            if hit_stop:
                exit_idx = candidate_idx
                exit_price = stop_price
                exit_reason = "stop"
                break
            if hit_target:
                exit_idx = candidate_idx
                exit_price = target_price
                exit_reason = "target"
                break

        gross_return = direction * (exit_price / entry - 1)
        net_return = gross_return - cost
        notional = min(equity * risk_per_trade / stop_pct, max_position_notional)
        pnl = notional * net_return
        equity += pnl
        if not allow_overlapping_trades:
            next_available_idx = exit_idx + 1
        rows.append(
            {
                "timestamp": entry_time,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "entry": entry,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "net_return": net_return,
                "notional": notional,
                "pnl": pnl,
                "equity": equity,
            }
        )

    result = pd.DataFrame(rows).set_index("timestamp")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result["trade"] = result["signal"] != "flat"
    result["win"] = np.where(result["trade"], result["pnl"] > 0, np.nan)
    return result
