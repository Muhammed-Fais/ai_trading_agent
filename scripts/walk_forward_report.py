from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.backtest import run_managed_backtest, summarize_backtest  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
from xauusd_ml.features import build_dataset  # noqa: E402
from xauusd_ml.model import fit_model  # noqa: E402
from xauusd_ml.regime import build_confluence_direction, build_regime_mask  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    min_confidence: float
    stop_atr: float
    take_profit_atr: float
    max_hold_bars: int
    regime_mode: str
    confluence_mode: str


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    selected: Candidate
    validation_summary: dict[str, float]
    test_summary: dict[str, float]


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
    scale = (height - 40) / abs(y_min) if y_min else 1
    step = max((width - 40) / max(len(values), 1), 1)
    bars = [
        f'<rect x="{x:.1f}" y="20" width="{max(step, 1):.1f}" '
        f'height="{abs(value) * scale:.1f}" fill="{color}" opacity="0.75" />'
        for x, value in zip(xs, values.to_numpy(), strict=True)
    ]
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg>'


def _score(summary: dict[str, float]) -> float:
    if summary["trades"] < 30:
        return -999.0
    return summary["total_return"] - 1.5 * abs(summary["max_drawdown"])


def _candidates() -> list[Candidate]:
    return [
        Candidate(confidence, stop, target, hold, regime, confluence)
        for regime in ("none", "trend_vol", "breakout_vol", "calm_trend")
        for confluence in ("none", "h4_trend", "h4_trend_or_breakout")
        for confidence, stop, target, hold in (
            (0.46, 1.6, 2.5, 16),
            (0.42, 2.0, 3.0, 16),
            (0.46, 2.0, 3.0, 16),
            (0.50, 2.0, 3.0, 16),
            (0.50, 2.0, 3.0, 24),
        )
    ]


def _date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01", tz="UTC")


def run_walk_forward(
    config_path: Path,
    output_path: Path,
    first_test_year: int,
    last_test_year: int,
    train_years: int,
    validation_years: int,
    test_years: int,
) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    candidates = _candidates()
    fold_results: list[FoldResult] = []
    test_curves = []
    equity_offset = cfg.backtest.initial_equity

    for fold, test_year in enumerate(range(first_test_year, last_test_year + 1, test_years), start=1):
        train_start = _date(test_year - validation_years - train_years)
        validation_start = _date(test_year - validation_years)
        test_start = _date(test_year)
        test_end = _date(test_year + test_years)

        train_df = df[(df.index >= train_start) & (df.index < validation_start)]
        validation_df = df[(df.index >= validation_start - pd.Timedelta(days=21)) & (df.index < test_start)]
        test_df = df[(df.index >= test_start - pd.Timedelta(days=21)) & (df.index < test_end)]
        if train_df.empty or validation_df.empty or test_df.empty:
            continue

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
        model = fit_model(x_train, y_train, cfg.model.random_state + fold)
        feature_names = list(x_train.columns)

        scored: list[tuple[float, Candidate, dict[str, float]]] = []
        validation_regimes = {
            candidate.regime_mode: build_regime_mask(validation_df, candidate.regime_mode)
            for candidate in candidates
        }
        validation_confluence = {
            candidate.confluence_mode: build_confluence_direction(
                validation_df,
                candidate.confluence_mode,
            )
            for candidate in candidates
        }
        for candidate in candidates:
            validation_bt = run_managed_backtest(
                df=validation_df,
                model=model,
                feature_names=feature_names,
                min_confidence=candidate.min_confidence,
                initial_equity=cfg.backtest.initial_equity,
                risk_per_trade=cfg.backtest.risk_per_trade,
                spread_bps=cfg.backtest.spread_bps,
                slippage_bps=cfg.backtest.slippage_bps,
                max_position_notional=cfg.backtest.max_position_notional,
                max_hold_bars=candidate.max_hold_bars,
                stop_atr_multiple=candidate.stop_atr,
                take_profit_atr_multiple=candidate.take_profit_atr,
                start_at=validation_start,
                allow_overlapping_trades=cfg.strategy.allow_overlapping_trades,
                regime_mask=validation_regimes[candidate.regime_mode],
                confluence_direction=validation_confluence[candidate.confluence_mode],
            )
            summary = summarize_backtest(validation_bt, cfg.backtest.initial_equity)
            scored.append((_score(summary), candidate, summary))

        _, selected, validation_summary = max(scored, key=lambda item: item[0])
        test_regime = build_regime_mask(test_df, selected.regime_mode)
        test_confluence = build_confluence_direction(test_df, selected.confluence_mode)
        test_bt = run_managed_backtest(
            df=test_df,
            model=model,
            feature_names=feature_names,
            min_confidence=selected.min_confidence,
            initial_equity=cfg.backtest.initial_equity,
            risk_per_trade=cfg.backtest.risk_per_trade,
            spread_bps=cfg.backtest.spread_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            max_position_notional=cfg.backtest.max_position_notional,
            max_hold_bars=selected.max_hold_bars,
            stop_atr_multiple=selected.stop_atr,
            take_profit_atr_multiple=selected.take_profit_atr,
            start_at=test_start,
            allow_overlapping_trades=cfg.strategy.allow_overlapping_trades,
            regime_mask=test_regime,
            confluence_direction=test_confluence,
        )
        test_summary = summarize_backtest(test_bt, cfg.backtest.initial_equity)

        curve = test_bt.copy()
        curve["fold"] = fold
        curve["fold_equity"] = curve["equity"]
        curve["equity"] = equity_offset + (curve["equity"] - cfg.backtest.initial_equity)
        equity_offset = float(curve["equity"].iloc[-1])
        test_curves.append(curve)

        fold_results.append(
            FoldResult(
                fold=fold,
                train_start=str(train_df.index.min()),
                train_end=str(train_df.index.max()),
                validation_start=str(validation_start),
                validation_end=str(test_start - pd.Timedelta(hours=1)),
                test_start=str(test_start),
                test_end=str(test_end - pd.Timedelta(hours=1)),
                selected=selected,
                validation_summary=validation_summary,
                test_summary=test_summary,
            )
        )
        print(
            f"fold={fold} test={test_year}-{test_year + test_years} "
            f"trades={test_summary['trades']:.0f} return={test_summary['total_return']:.2%} "
            f"dd={test_summary['max_drawdown']:.2%}"
        )

    if not fold_results or not test_curves:
        raise ValueError("No walk-forward folds were produced.")

    combined = pd.concat(test_curves).sort_index()
    combined["drawdown"] = combined["equity"] / combined["equity"].cummax() - 1
    combined_summary = summarize_backtest(combined, cfg.backtest.initial_equity)
    fold_payload = [
        {
            **asdict(result),
            "selected": asdict(result.selected),
        }
        for result in fold_results
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path.with_suffix(".trades.csv"))
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )
    output_path.with_suffix(".folds.json").write_text(json.dumps(fold_payload, indent=2), encoding="utf-8")

    rows = "".join(
        "<tr>"
        f"<td>{result.fold}</td>"
        f"<td>{html.escape(result.test_start[:10])} to {html.escape(result.test_end[:10])}</td>"
        f"<td>{result.selected.min_confidence:.2f}</td>"
        f"<td>{result.selected.stop_atr:.1f}</td>"
        f"<td>{result.selected.take_profit_atr:.1f}</td>"
        f"<td>{result.selected.max_hold_bars}</td>"
        f"<td>{html.escape(result.selected.regime_mode)}</td>"
        f"<td>{html.escape(result.selected.confluence_mode)}</td>"
        f"<td>{result.test_summary['trades']:.0f}</td>"
        f"<td>{_fmt_pct(result.test_summary['total_return'])}</td>"
        f"<td>{_fmt_pct(result.test_summary['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(result.test_summary['win_rate'])}</td>"
        f"<td>{result.test_summary['profit_factor']:.2f}</td>"
        "</tr>"
        for result in fold_results
    )
    metrics = {
        "Ending Equity": _fmt_money(combined_summary["ending_equity"]),
        "Total Return": _fmt_pct(combined_summary["total_return"]),
        "Max Drawdown": _fmt_pct(combined_summary["max_drawdown"]),
        "Trades": f"{combined_summary['trades']:,.0f}",
        "Win Rate": _fmt_pct(combined_summary["win_rate"]),
        "Profit Factor": f"{combined_summary['profit_factor']:.2f}",
        "Avg Trade PnL": _fmt_money(combined_summary["avg_trade_pnl"]),
    }
    metric_cards = "".join(f"<div><span>{key}</span><strong>{value}</strong></div>" for key, value in metrics.items())

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD Walk-Forward Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; }}
    p {{ color: #4b5563; line-height: 1.55; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 24px 0; }}
    .metrics div {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metrics span {{ display: block; color: #6b7280; font-size: 13px; }}
    .metrics strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-top: 16px; }}
    svg {{ width: 100%; height: auto; background: #ffffff; border: 1px solid #eef2f7; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ color: #374151; background: #f9fafb; }}
    code {{ background: #eef2ff; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD Walk-Forward Report</h1>
  <p>
    Dataset: <code>{html.escape(str(cfg.data.train_csv))}</code>, rows {len(df):,},
    coverage {df.index.min().date()} to {df.index.max().date()}.
    Each fold trains on {train_years} years, chooses execution parameters on the next
    {validation_years} year(s), then tests the next unseen {test_years} year(s).
  </p>
  <div class="metrics">{metric_cards}</div>
  <section>
    <h2>Combined Equity Curve</h2>
    {_line_svg(combined["equity"])}
  </section>
  <section>
    <h2>Combined Underwater Drawdown</h2>
    {_bar_svg(combined["drawdown"])}
  </section>
  <section>
    <h2>Fold Results</h2>
    <table>
      <thead>
        <tr><th>Fold</th><th>Test Window</th><th>Conf</th><th>Stop ATR</th><th>Target ATR</th><th>Hold</th><th>Regime</th><th>Confluence</th><th>Trades</th><th>Return</th><th>DD</th><th>Win</th><th>PF</th></tr>
      </thead>
      <tbody>{rows}</tbody>
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
    parser.add_argument("--first-test-year", type=int, default=2014)
    parser.add_argument("--last-test-year", type=int, default=2024)
    parser.add_argument("--train-years", type=int, default=7)
    parser.add_argument("--validation-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=2)
    args = parser.parse_args()
    run_walk_forward(
        args.config,
        args.output,
        args.first_test_year,
        args.last_test_year,
        args.train_years,
        args.validation_years,
        args.test_years,
    )


if __name__ == "__main__":
    main()
