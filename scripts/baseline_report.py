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


def _line_svg(series: pd.Series, width: int = 960, height: int = 260, color: str = "#2563eb") -> str:
    values = series.dropna().astype(float)
    y_min, y_max = float(values.min()), float(values.max())
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    xs = np.linspace(20, width - 20, len(values))
    ys = height - 20 - ((values.to_numpy() - y_min) / (y_max - y_min)) * (height - 40)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />'
        "</svg>"
    )


def _signal(df: pd.DataFrame, mode: str) -> pd.Series:
    close = df["close"]
    ema_fast = close.ewm(span=55, adjust=False).mean()
    ema_slow = close.ewm(span=288, adjust=False).mean()
    donchian_high = df["high"].rolling(96).max().shift(1)
    donchian_low = df["low"].rolling(96).min().shift(1)
    vol = close.pct_change().rolling(96).std()
    vol_ok = vol > vol.rolling(500).median()

    signal = pd.Series("flat", index=df.index, dtype="object")
    if mode == "ema_trend":
        signal[(ema_fast > ema_slow) & (ema_fast.pct_change(8) > 0)] = "long"
        signal[(ema_fast < ema_slow) & (ema_fast.pct_change(8) < 0)] = "short"
    elif mode == "donchian_breakout":
        signal[(close > donchian_high) & vol_ok] = "long"
        signal[(close < donchian_low) & vol_ok] = "short"
    elif mode == "hybrid":
        signal[(close > donchian_high) & (ema_fast > ema_slow) & vol_ok] = "long"
        signal[(close < donchian_low) & (ema_fast < ema_slow) & vol_ok] = "short"
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")
    return signal


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
    start_at: pd.Timestamp,
) -> pd.DataFrame:
    features = build_features(df)
    equity = initial_equity
    cost = (spread_bps + slippage_bps) / 10_000
    next_available_idx = 0
    rows = []

    for feature_time, label in signals.items():
        bar_idx = df.index.get_indexer([feature_time])[0]
        if feature_time < start_at or bar_idx < 0 or bar_idx + 1 >= len(df):
            continue
        if bar_idx < next_available_idx:
            continue
        direction = {"long": 1, "short": -1, "flat": 0}[label]
        if direction == 0:
            continue

        entry_idx = bar_idx + 1
        entry_time = df.index[entry_idx]
        entry = float(df["open"].iloc[entry_idx])
        atr_pct = float(features["atr_pct_14"].loc[feature_time])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue

        stop_pct = max(stop_atr_multiple * atr_pct, cost * 2)
        target_pct = max(take_profit_atr_multiple * atr_pct, cost * 2)
        stop_price = entry * (1 - direction * stop_pct)
        target_price = entry * (1 + direction * target_pct)
        exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
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

    result = pd.DataFrame(rows).set_index("timestamp")
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1
    result["trade"] = True
    result["win"] = result["pnl"] > 0
    return result


def generate(config_path: Path, output_path: Path, start_year: int) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    start_at = pd.Timestamp(f"{start_year}-01-01", tz="UTC")
    rows = []
    curves = {}
    for mode in ("ema_trend", "donchian_breakout", "hybrid"):
        bt = run_rule_backtest(
            df,
            _signal(df, mode),
            cfg.backtest.initial_equity,
            cfg.backtest.risk_per_trade,
            cfg.backtest.spread_bps,
            cfg.backtest.slippage_bps,
            cfg.backtest.max_position_notional,
            cfg.strategy.max_hold_bars,
            cfg.strategy.stop_atr_multiple,
            cfg.strategy.take_profit_atr_multiple,
            start_at,
        )
        summary = summarize_backtest(bt, cfg.backtest.initial_equity)
        curves[mode] = bt
        rows.append((mode, summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps({mode: summary for mode, summary in rows}, indent=2),
        encoding="utf-8",
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(mode)}</td>"
        f"<td>{summary['trades']:.0f}</td>"
        f"<td>{_fmt_pct(summary['total_return'])}</td>"
        f"<td>{_fmt_pct(summary['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(summary['win_rate'])}</td>"
        f"<td>{summary['profit_factor']:.2f}</td>"
        f"<td>{_fmt_money(summary['avg_trade_pnl'])}</td>"
        "</tr>"
        for mode, summary in rows
    )
    best_mode, _ = max(rows, key=lambda item: item[1]["profit_factor"])
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD Baseline Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 48px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
    svg {{ width: 100%; height: auto; background: white; border: 1px solid #eef2f7; }}
    p {{ color: #4b5563; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD Transparent Baselines</h1>
  <p>Start year: {start_year}. Same ATR exit and risk model as the ML strategy.</p>
  <section>
    <table>
      <thead><tr><th>Rule</th><th>Trades</th><th>Return</th><th>DD</th><th>Win</th><th>PF</th><th>Avg PnL</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </section>
  <section>
    <h2>Best Baseline Equity: {html.escape(best_mode)}</h2>
    {_line_svg(curves[best_mode]["equity"])}
  </section>
</main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=2014)
    args = parser.parse_args()
    generate(args.config, args.output, args.start_year)


if __name__ == "__main__":
    main()

