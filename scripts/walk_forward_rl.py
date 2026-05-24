from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.backtest import summarize_backtest  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
from xauusd_ml.features import build_features  # noqa: E402
from xauusd_ml.rl_env import RLEnvConfig, XAUUSDTradingEnv  # noqa: E402
from xauusd_ml.rl_training import evaluate_policy, train_cem  # noqa: E402


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _line_svg(series: pd.Series, width: int = 960, height: int = 260) -> str:
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
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}" />'
        "</svg>"
    )


def _feature_names(df: pd.DataFrame) -> tuple[str, ...]:
    features = build_features(df)
    return tuple(features.dropna(axis=1, how="all").columns)


def _date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01", tz="UTC")


def run_walk_forward(
    config_path: Path,
    output_path: Path,
    first_test_year: int,
    last_test_year: int,
    train_years: int,
    test_years: int,
    iterations: int,
    population: int,
) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    fold_payload = []
    curves = []
    equity_offset = cfg.backtest.initial_equity

    for fold, test_year in enumerate(range(first_test_year, last_test_year + 1, test_years), start=1):
        train_start = _date(test_year - train_years)
        train_end = _date(test_year)
        test_start = _date(test_year)
        test_end = _date(test_year + test_years)

        train = df[(df.index >= train_start) & (df.index < train_end)]
        test = df[(df.index >= test_start) & (df.index < test_end)]
        if train.empty or test.empty:
            continue

        feature_names = _feature_names(train)
        env_config = RLEnvConfig(
            feature_names=feature_names,
            initial_equity=cfg.backtest.initial_equity,
            position_notional=cfg.backtest.max_position_notional * 0.25,
            spread_bps=cfg.backtest.spread_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            drawdown_penalty=0.2,
            turnover_penalty=0.03,
            max_episode_steps=2_000,
        )
        env = XAUUSDTradingEnv(train, env_config)
        policy = train_cem(
            env,
            iterations=iterations,
            population=population,
            elite_fraction=0.2,
            seed=cfg.model.random_state + fold,
        )
        result = evaluate_policy(
            test,
            policy,
            feature_names,
            env_config,
            start_at=test_start,
        )
        summary = summarize_backtest(result, cfg.backtest.initial_equity)
        curve = result.copy()
        curve["fold"] = fold
        curve["fold_equity"] = curve["equity"]
        curve["equity"] = equity_offset + (curve["equity"] - cfg.backtest.initial_equity)
        equity_offset = float(curve["equity"].iloc[-1])
        curves.append(curve)
        fold_payload.append(
            {
                "fold": fold,
                "train_start": str(train_start),
                "train_end": str(train_end - pd.Timedelta(hours=1)),
                "test_start": str(test_start),
                "test_end": str(test_end - pd.Timedelta(hours=1)),
                "summary": summary,
                "env_config": asdict(env_config),
            }
        )
        print(
            f"fold={fold} test={test_year}-{test_year + test_years} "
            f"return={summary['total_return']:.2%} dd={summary['max_drawdown']:.2%} "
            f"trades={summary['trades']:.0f} pf={summary['profit_factor']:.2f}"
        )

    if not curves:
        raise ValueError("No RL walk-forward folds were produced.")

    combined = pd.concat(curves).sort_index()
    combined["drawdown"] = combined["equity"] / combined["equity"].cummax() - 1
    combined_summary = summarize_backtest(combined, cfg.backtest.initial_equity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path.with_suffix(".trades.csv"))
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(combined_summary, indent=2),
        encoding="utf-8",
    )
    output_path.with_suffix(".folds.json").write_text(
        json.dumps(fold_payload, indent=2),
        encoding="utf-8",
    )

    rows = "".join(
        "<tr>"
        f"<td>{fold['fold']}</td>"
        f"<td>{html.escape(fold['test_start'][:10])} to {html.escape(fold['test_end'][:10])}</td>"
        f"<td>{fold['summary']['trades']:.0f}</td>"
        f"<td>{_fmt_pct(fold['summary']['total_return'])}</td>"
        f"<td>{_fmt_pct(fold['summary']['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(fold['summary']['win_rate'])}</td>"
        f"<td>{fold['summary']['profit_factor']:.2f}</td>"
        "</tr>"
        for fold in fold_payload
    )
    cards = "".join(
        f"<div><span>{label}</span><strong>{value}</strong></div>"
        for label, value in {
            "Ending Equity": _fmt_money(combined_summary["ending_equity"]),
            "Total Return": _fmt_pct(combined_summary["total_return"]),
            "Max Drawdown": _fmt_pct(combined_summary["max_drawdown"]),
            "Trades": f"{combined_summary['trades']:,.0f}",
            "Win Rate": _fmt_pct(combined_summary["win_rate"]),
            "Profit Factor": f"{combined_summary['profit_factor']:.2f}",
        }.items()
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD RL Walk-Forward</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metrics div, section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    span {{ display: block; color: #6b7280; font-size: 13px; }}
    strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    section {{ margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
    svg {{ width: 100%; height: auto; border: 1px solid #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD RL Walk-Forward</h1>
  <p>Offline CEM-trained linear policies. Each fold trains only on prior data and tests the next unseen window.</p>
  <div class="metrics">{cards}</div>
  <section>
    <h2>Combined Equity</h2>
    {_line_svg(combined["equity"])}
  </section>
  <section>
    <h2>Folds</h2>
    <table>
      <thead><tr><th>Fold</th><th>Test Window</th><th>Trades</th><th>Return</th><th>DD</th><th>Win</th><th>PF</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--first-test-year", type=int, default=2014)
    parser.add_argument("--last-test-year", type=int, default=2024)
    parser.add_argument("--train-years", type=int, default=7)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--population", type=int, default=16)
    args = parser.parse_args()
    run_walk_forward(
        args.config,
        args.output,
        args.first_test_year,
        args.last_test_year,
        args.train_years,
        args.test_years,
        args.iterations,
        args.population,
    )


if __name__ == "__main__":
    main()
