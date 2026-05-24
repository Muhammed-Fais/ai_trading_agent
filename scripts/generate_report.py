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

from xauusd_ml.backtest import run_managed_backtest, summarize_backtest
from xauusd_ml.config import load_config
from xauusd_ml.data import load_ohlcv_csv
from xauusd_ml.features import build_dataset
from xauusd_ml.model import fit_model, save_artifact, validate_model


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _line_svg(series: pd.Series, width: int = 960, height: int = 260, color: str = "#2563eb") -> str:
    values = series.dropna().astype(float)
    if values.empty:
        return "<svg></svg>"
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


def _bar_svg(series: pd.Series, width: int = 960, height: int = 220, color: str = "#dc2626") -> str:
    values = series.dropna().astype(float)
    if values.empty:
        return "<svg></svg>"
    y_min = min(float(values.min()), 0.0)
    xs = np.linspace(20, width - 20, len(values))
    baseline = 20
    scale = (height - 40) / abs(y_min) if y_min else 1
    bars = []
    step = max((width - 40) / max(len(values), 1), 1)
    for x, value in zip(xs, values.to_numpy(), strict=True):
        bar_height = abs(value) * scale
        bars.append(
            f'<rect x="{x:.1f}" y="{baseline:.1f}" width="{max(step, 1):.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" opacity="0.75" />'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg>'


def _classification_table(report: str) -> str:
    rows = []
    for line in report.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] in {"short", "flat", "long"}:
            label, precision, recall, f1, support = parts
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td><td>{precision}</td><td>{recall}</td>"
                f"<td>{f1}</td><td>{support}</td>"
                "</tr>"
            )
    return "".join(rows)


def generate_report(
    config_path: Path,
    output_path: Path,
    train_fraction: float,
    periods_per_year: float,
) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)

    split_at = int(len(df) * train_fraction)
    train_df = df.iloc[:split_at]
    test_df = df.iloc[max(0, split_at - 200) :]

    x_train, y_train = build_dataset(
        train_df,
        cfg.features.horizon_bars,
        cfg.features.threshold_bps,
        cfg.features.min_rows,
        label_mode=cfg.features.label_mode,
        max_hold_bars=cfg.strategy.max_hold_bars,
        stop_atr_multiple=cfg.strategy.stop_atr_multiple,
        take_profit_atr_multiple=cfg.strategy.take_profit_atr_multiple,
    )
    validation = validate_model(
        x_train,
        y_train,
        cfg.validation.splits,
        cfg.validation.embargo_bars,
        cfg.model.random_state,
    )

    model = fit_model(x_train, y_train, cfg.model.random_state)
    save_artifact(model, list(x_train.columns), cfg.model.artifact_path)

    backtest = run_managed_backtest(
        df=test_df,
        model=model,
        feature_names=list(x_train.columns),
        min_confidence=cfg.model.min_confidence,
        initial_equity=cfg.backtest.initial_equity,
        risk_per_trade=cfg.backtest.risk_per_trade,
        spread_bps=cfg.backtest.spread_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        max_position_notional=cfg.backtest.max_position_notional,
        max_hold_bars=cfg.strategy.max_hold_bars,
        stop_atr_multiple=cfg.strategy.stop_atr_multiple,
        take_profit_atr_multiple=cfg.strategy.take_profit_atr_multiple,
        start_at=df.index[split_at],
        allow_overlapping_trades=cfg.strategy.allow_overlapping_trades,
    )
    summary = summarize_backtest(backtest, cfg.backtest.initial_equity)

    trade_pnl = backtest.loc[backtest["trade"], "pnl"]
    trade_sharpe = float(trade_pnl.mean() / trade_pnl.std()) if trade_pnl.std() else 0.0
    exposure = float(backtest["trade"].mean()) if len(backtest) else 0.0

    validation_rows = "".join(
        f"<tr><td>{result.fold}</td><td>{result.log_loss:.4f}</td>"
        f"<td><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>"
        f"<th>F1</th><th>Support</th></tr></thead><tbody>{_classification_table(result.report)}</tbody>"
        f"</table></td></tr>"
        for result in validation
    )

    metrics = {
        "Ending Equity": _fmt_money(summary["ending_equity"]),
        "Total Return": _fmt_pct(summary["total_return"]),
        "Max Drawdown": _fmt_pct(summary["max_drawdown"]),
        "Trades": f"{summary['trades']:,.0f}",
        "Win Rate": _fmt_pct(summary["win_rate"]),
        "Avg Trade PnL": _fmt_money(summary["avg_trade_pnl"]),
        "Profit Factor": f"{summary['profit_factor']:.2f}",
        "Trade Sharpe": f"{trade_sharpe:.2f}",
        "Signal Exposure": _fmt_pct(exposure),
    }
    metric_cards = "".join(f"<div><span>{k}</span><strong>{v}</strong></div>" for k, v in metrics.items())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(output_path.with_suffix(".trades.csv"))
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation_payload = [
        {"fold": result.fold, "log_loss": result.log_loss, "classification_report": result.report}
        for result in validation
    ]
    output_path.with_suffix(".validation.json").write_text(
        json.dumps(validation_payload, indent=2),
        encoding="utf-8",
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD ML Backtest Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    p {{ color: #4b5563; line-height: 1.55; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metrics div {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metrics span {{ display: block; color: #6b7280; font-size: 13px; }}
    .metrics strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 16px; }}
    svg {{ width: 100%; height: auto; background: #ffffff; border: 1px solid #eef2f7; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ color: #374151; background: #f9fafb; }}
    code {{ background: #eef2ff; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD ML Backtest Report</h1>
  <p>
    Dataset: <code>{html.escape(str(cfg.data.train_csv))}</code>,
    rows {len(df):,}, coverage {df.index.min().date()} to {df.index.max().date()}.
    Train window ends {df.index[split_at].date()}; out-of-sample backtest starts there.
    Horizon: {cfg.features.horizon_bars} bars, threshold: {cfg.features.threshold_bps} bps,
    confidence gate: {cfg.model.min_confidence:.2f}.
  </p>

  <div class="metrics">{metric_cards}</div>

  <section>
    <h2>Equity Curve</h2>
    {_line_svg(backtest["equity"], color="#2563eb")}
  </section>

  <section>
    <h2>Underwater Drawdown</h2>
    {_bar_svg(backtest["drawdown"], color="#dc2626")}
  </section>

  <section>
    <h2>Validation Results</h2>
    <table>
      <thead><tr><th>Fold</th><th>Log Loss</th><th>Classification Report</th></tr></thead>
      <tbody>{validation_rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Assumptions</h2>
    <table>
      <tbody>
        <tr><th>Model</th><td>Soft-voting ensemble: logistic regression, random forest, histogram gradient boosting.</td></tr>
        <tr><th>Execution Cost</th><td>Spread {cfg.backtest.spread_bps} bps plus slippage {cfg.backtest.slippage_bps} bps per trade.</td></tr>
        <tr><th>Sizing</th><td>Risk fraction {cfg.backtest.risk_per_trade:.3f}; max notional {_fmt_money(cfg.backtest.max_position_notional)}.</td></tr>
        <tr><th>Trade Management</th><td>Next-bar entry, ATR stop {cfg.strategy.stop_atr_multiple:.2f}x, ATR target {cfg.strategy.take_profit_atr_multiple:.2f}x, max hold {cfg.strategy.max_hold_bars} bars.</td></tr>
        <tr><th>Data Caveat</th><td>Configured research data from <code>{html.escape(str(cfg.data.train_csv))}</code>. Add broker-native spread history and paper-trading logs before trusting production results.</td></tr>
      </tbody>
    </table>
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
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--periods-per-year", type=float, default=252)
    args = parser.parse_args()
    generate_report(args.config, args.output, args.train_fraction, args.periods_per_year)


if __name__ == "__main__":
    main()
