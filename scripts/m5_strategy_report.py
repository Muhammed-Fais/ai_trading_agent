from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.backtest import summarize_backtest  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
from xauusd_ml.features import build_features  # noqa: E402


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _line_svg(series: pd.Series, width: int = 1000, height: int = 260, color: str = "#2563eb") -> str:
    values = series.dropna().astype(float)
    if values.empty:
        return "<svg></svg>"
    y_min, y_max = float(values.min()), float(values.max())
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    xs = np.linspace(18, width - 18, len(values))
    ys = height - 18 - ((values.to_numpy() - y_min) / (y_max - y_min)) * (height - 36)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    return f'<svg viewBox="0 0 {width} {height}"><polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" /></svg>'


def _underwater_svg(series: pd.Series, width: int = 1000, height: int = 220) -> str:
    values = series.dropna().astype(float)
    if values.empty:
        return "<svg></svg>"
    y_min = min(float(values.min()), 0.0)
    scale = (height - 36) / abs(y_min) if y_min else 1
    xs = np.linspace(18, width - 18, len(values))
    step = max((width - 36) / max(len(values), 1), 1)
    bars = [
        f'<rect x="{x:.1f}" y="18" width="{step:.1f}" height="{abs(value) * scale:.1f}" />'
        for x, value in zip(xs, values.to_numpy(), strict=True)
    ]
    return f'<svg viewBox="0 0 {width} {height}" class="underwater">{"".join(bars)}</svg>'


def _session_breakout_signal(
    df: pd.DataFrame,
    range_start_hour: int,
    trade_start_hour: int,
    trade_end_hour: int,
    trend_filter: bool,
) -> pd.Series:
    features = build_features(df)
    signal = pd.Series("flat", index=df.index, dtype="object")
    local = df.copy()
    local["day"] = local.index.date
    local["hour"] = local.index.hour
    local["minute"] = local.index.minute
    ema_21 = local["close"].ewm(span=21, adjust=False).mean()
    ema_144 = local["close"].ewm(span=144, adjust=False).mean()

    for _, day_df in local.groupby("day", sort=True):
        opening_range = day_df[
            (day_df["hour"] == range_start_hour)
            & (day_df["minute"] >= 0)
            & (day_df["minute"] <= 55)
        ]
        if len(opening_range) < 8:
            continue
        high = float(opening_range["high"].max())
        low = float(opening_range["low"].min())
        trade_window = day_df[(day_df["hour"] >= trade_start_hour) & (day_df["hour"] < trade_end_hour)]
        if trade_window.empty:
            continue
        for timestamp in trade_window.index:
            long_ok = df.at[timestamp, "close"] > high
            short_ok = df.at[timestamp, "close"] < low
            if trend_filter:
                long_ok = long_ok and ema_21.loc[timestamp] > ema_144.loc[timestamp]
                short_ok = short_ok and ema_21.loc[timestamp] < ema_144.loc[timestamp]
            long_ok = long_ok and features.at[timestamp, "atr_regime_96"] > -0.2
            short_ok = short_ok and features.at[timestamp, "atr_regime_96"] > -0.2
            if long_ok:
                signal.loc[timestamp] = "long"
                break
            elif short_ok:
                signal.loc[timestamp] = "short"
                break
    return signal


def _mean_reversion_signal(df: pd.DataFrame) -> pd.Series:
    features = build_features(df)
    signal = pd.Series("flat", index=df.index, dtype="object")
    london_ny = df.index.hour.isin([7, 8, 9, 10, 13, 14, 15, 16])
    calm = features["atr_regime_96"] < 0.35
    long_mask = (features["donchian_pos_96"] < 0.08) & (features["rsi_14"] < 0.28) & calm & london_ny
    short_mask = (features["donchian_pos_96"] > 0.92) & (features["rsi_14"] > 0.72) & calm & london_ny
    signal[long_mask & ~long_mask.shift(fill_value=False)] = "long"
    signal[short_mask & ~short_mask.shift(fill_value=False)] = "short"
    return signal


def _momentum_signal(df: pd.DataFrame) -> pd.Series:
    features = build_features(df)
    signal = pd.Series("flat", index=df.index, dtype="object")
    active_session = df.index.hour.isin([7, 8, 9, 10, 13, 14, 15, 16])
    vol_ok = features["atr_regime_96"] > 0
    long_mask = (features["ret_4"] > features["realized_vol_24"] * 1.8) & vol_ok & active_session
    short_mask = (features["ret_4"] < -features["realized_vol_24"] * 1.8) & vol_ok & active_session
    signal[long_mask & ~long_mask.shift(fill_value=False)] = "long"
    signal[short_mask & ~short_mask.shift(fill_value=False)] = "short"
    return signal


def _ist_daily_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].replace(0, np.nan)
    ist_day = df.index.tz_convert("Asia/Kolkata").date
    weighted_price = typical_price * volume
    cumulative_weighted = weighted_price.groupby(ist_day).cumsum()
    cumulative_volume = volume.groupby(ist_day).cumsum()
    vwap = cumulative_weighted / cumulative_volume
    fallback = typical_price.groupby(ist_day).expanding().mean().reset_index(level=0, drop=True)
    return vwap.fillna(fallback)


def _vwap_6am_ist_signal(df: pd.DataFrame, require_candle_color: bool) -> pd.Series:
    vwap = _ist_daily_vwap(df)
    signal = pd.Series("flat", index=df.index, dtype="object")
    ist_index = df.index.tz_convert("Asia/Kolkata")
    signal_bar = (ist_index.hour == 6) & (ist_index.minute == 0)
    above_vwap = df["close"] > vwap
    below_vwap = df["close"] < vwap
    bullish = df["close"] > df["open"]
    bearish = df["close"] < df["open"]
    if require_candle_color:
        above_vwap = above_vwap & bullish
        below_vwap = below_vwap & bearish
    signal[signal_bar & above_vwap] = "long"
    signal[signal_bar & below_vwap] = "short"
    return signal


def build_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "vwap_6am_ist": _vwap_6am_ist_signal(df, require_candle_color=False),
        "vwap_6am_ist_color": _vwap_6am_ist_signal(df, require_candle_color=True),
        "london_or_breakout": _session_breakout_signal(df, 7, 8, 12, trend_filter=True),
        "ny_or_breakout": _session_breakout_signal(df, 13, 14, 18, trend_filter=True),
        "m5_mean_reversion": _mean_reversion_signal(df),
        "m5_session_momentum": _momentum_signal(df),
    }


def run_rule_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> pd.DataFrame:
    features = build_features(df)
    equity = initial_equity
    cost = (spread_bps + slippage_bps) / 10_000
    next_available_idx = 0
    rows = []

    for feature_time, label in signals.items():
        bar_idx = df.index.get_indexer([feature_time])[0]
        if bar_idx < 0 or bar_idx + 1 >= len(df) or bar_idx < next_available_idx:
            continue
        direction = {"long": 1, "short": -1, "flat": 0}[label]
        if direction == 0:
            continue

        atr_pct = float(features["atr_pct_14"].loc[feature_time])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue

        entry_idx = bar_idx + 1
        entry_time = df.index[entry_idx]
        entry = float(df["open"].iloc[entry_idx])
        stop_pct = max(stop_atr_multiple * atr_pct, cost * 2)
        target_pct = max(take_profit_atr_multiple * atr_pct, cost * 2)
        stop = entry * (1 - direction * stop_pct)
        target = entry * (1 + direction * target_pct)
        exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
        exit_price = float(df["close"].iloc[exit_idx])
        exit_reason = "time"

        for candidate_idx in range(entry_idx, exit_idx + 1):
            high = float(df["high"].iloc[candidate_idx])
            low = float(df["low"].iloc[candidate_idx])
            if direction == 1:
                hit_stop = low <= stop
                hit_target = high >= target
            else:
                hit_stop = high >= stop
                hit_target = low <= target
            if hit_stop and hit_target:
                exit_idx = candidate_idx
                exit_price = stop
                exit_reason = "stop_and_target_same_bar"
                break
            if hit_stop:
                exit_idx = candidate_idx
                exit_price = stop
                exit_reason = "stop"
                break
            if hit_target:
                exit_idx = candidate_idx
                exit_price = target
                exit_reason = "target"
                break

        net_return = direction * (exit_price / entry - 1) - cost
        notional = min(equity * risk_per_trade / stop_pct, max_position_notional)
        pnl = notional * net_return
        equity += pnl
        next_available_idx = exit_idx + 1
        rows.append(
            {
                "timestamp": entry_time,
                "signal": label,
                "entry": entry,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "net_return": net_return,
                "notional": notional,
                "pnl": pnl,
                "equity": equity,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["signal", "entry", "exit", "exit_reason", "net_return", "notional", "pnl", "equity", "drawdown", "trade", "win"])
    result = result.set_index("timestamp")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result["trade"] = True
    result["win"] = result["pnl"] > 0
    return result


def run_vwap_6am_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    max_hold_bars: int,
    reward_risk: float = 2.0,
) -> pd.DataFrame:
    equity = initial_equity
    cost = (spread_bps + slippage_bps) / 10_000
    rows = []

    for feature_time, label in signals.items():
        bar_idx = df.index.get_indexer([feature_time])[0]
        if bar_idx < 0 or bar_idx + 1 >= len(df):
            continue
        direction = {"long": 1, "short": -1, "flat": 0}[label]
        if direction == 0:
            continue

        entry_idx = bar_idx + 1
        entry_time = df.index[entry_idx]
        entry = float(df["open"].iloc[entry_idx])
        signal_high = float(df["high"].iloc[bar_idx])
        signal_low = float(df["low"].iloc[bar_idx])
        if direction == 1:
            stop = signal_low
            stop_pct = (entry - stop) / entry
            target = entry * (1 + reward_risk * stop_pct)
        else:
            stop = signal_high
            stop_pct = (stop - entry) / entry
            target = entry * (1 - reward_risk * stop_pct)
        stop_pct = max(stop_pct, cost * 2)
        if direction == 1:
            stop = entry * (1 - stop_pct)
            target = entry * (1 + reward_risk * stop_pct)
        else:
            stop = entry * (1 + stop_pct)
            target = entry * (1 - reward_risk * stop_pct)

        exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
        exit_price = float(df["close"].iloc[exit_idx])
        exit_reason = "time"
        for candidate_idx in range(entry_idx, exit_idx + 1):
            high = float(df["high"].iloc[candidate_idx])
            low = float(df["low"].iloc[candidate_idx])
            if direction == 1:
                hit_stop = low <= stop
                hit_target = high >= target
            else:
                hit_stop = high >= stop
                hit_target = low <= target
            if hit_stop and hit_target:
                exit_idx = candidate_idx
                exit_price = stop
                exit_reason = "stop_and_target_same_bar"
                break
            if hit_stop:
                exit_idx = candidate_idx
                exit_price = stop
                exit_reason = "stop"
                break
            if hit_target:
                exit_idx = candidate_idx
                exit_price = target
                exit_reason = "target"
                break

        net_return = direction * (exit_price / entry - 1) - cost
        notional = min(equity * risk_per_trade / stop_pct, max_position_notional)
        pnl = notional * net_return
        equity += pnl
        rows.append(
            {
                "timestamp": entry_time,
                "signal": label,
                "entry": entry,
                "exit": exit_price,
                "exit_reason": exit_reason,
                "net_return": net_return,
                "notional": notional,
                "pnl": pnl,
                "equity": equity,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "entry",
                "exit",
                "exit_reason",
                "net_return",
                "notional",
                "pnl",
                "equity",
                "drawdown",
                "trade",
                "win",
            ]
        )
    result = result.set_index("timestamp")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result["trade"] = True
    result["win"] = result["pnl"] > 0
    return result


def generate(config_path: Path, output_path: Path) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    reports = {}
    curves = {}
    for name, signals in build_signals(df).items():
        if name.startswith("vwap_6am_ist"):
            bt = run_vwap_6am_backtest(
                df,
                signals,
                cfg.backtest.initial_equity,
                cfg.backtest.risk_per_trade,
                cfg.backtest.spread_bps,
                cfg.backtest.slippage_bps,
                cfg.backtest.max_position_notional,
                cfg.strategy.max_hold_bars,
            )
        else:
            bt = run_rule_backtest(
                df,
                signals,
                cfg.backtest.initial_equity,
                cfg.backtest.risk_per_trade,
                cfg.backtest.spread_bps,
                cfg.backtest.slippage_bps,
                cfg.backtest.max_position_notional,
                cfg.strategy.max_hold_bars,
                cfg.strategy.stop_atr_multiple,
                cfg.strategy.take_profit_atr_multiple,
            )
        summary = summarize_backtest(bt, cfg.backtest.initial_equity) if not bt.empty else {
            "ending_equity": cfg.backtest.initial_equity,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }
        reports[name] = summary
        curves[name] = bt

    best_name = max(reports, key=lambda key: reports[key]["profit_factor"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not curves[best_name].empty:
        curves[best_name].to_csv(output_path.with_suffix(".trades.csv"))

    rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{summary['trades']:.0f}</td>"
        f"<td>{_fmt_pct(summary['total_return'])}</td>"
        f"<td>{_fmt_pct(summary['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(summary['win_rate'])}</td>"
        f"<td>{summary['profit_factor']:.2f}</td>"
        f"<td>{_fmt_money(summary['avg_trade_pnl'])}</td>"
        "</tr>"
        for name, summary in sorted(reports.items(), key=lambda item: item[1]["profit_factor"], reverse=True)
    )
    best_curve = curves[best_name]
    best_equity = _line_svg(best_curve["equity"]) if not best_curve.empty else "<svg></svg>"
    best_underwater = _underwater_svg(best_curve["drawdown"]) if not best_curve.empty else "<svg></svg>"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD M5 Strategy Report</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f7f8fb; color:#111827; }}
    main {{ max-width:1120px; margin:0 auto; padding:32px 20px 52px; }}
    section {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:18px; margin-top:16px; }}
    h1 {{ margin:0; font-size:32px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:20px; }}
    p {{ color:#4b5563; line-height:1.5; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; white-space:nowrap; }}
    th {{ background:#f9fafb; }}
    .grid {{ display:grid; grid-template-columns:1fr; gap:12px; }}
    svg {{ display:block; width:100%; height:auto; border:1px solid #eef2f7; border-radius:6px; background:#fff; }}
    .underwater rect {{ fill:#dc2626; opacity:.72; }}
    @media (max-width: 760px) {{ section {{ overflow-x:auto; }} }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD M5 Strategy Report</h1>
  <p>Data window: {df.index.min()} to {df.index.max()}. VWAP is anchored to each IST day; because this broker export has no usable volume, VWAP falls back to an intraday typical-price average. This is a short broker-export sample, so treat the result as a first filter, not production evidence.</p>
  <section>
    <h2>Strategy Comparison</h2>
    <table>
      <thead><tr><th>Strategy</th><th>Trades</th><th>Return</th><th>Max DD</th><th>Win</th><th>PF</th><th>Avg PnL</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Best Equity Curve: {html.escape(best_name)}</h2>
    <div class="grid">{best_equity}</div>
  </section>
  <section>
    <h2>Best Underwater Drawdown</h2>
    <div class="grid">{best_underwater}</div>
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path("config/report_m5.yaml"), type=Path)
    parser.add_argument("--output", default=Path("reports/xauusd_m5_strategy_report.html"), type=Path)
    args = parser.parse_args()
    generate(args.config, args.output)


if __name__ == "__main__":
    main()
