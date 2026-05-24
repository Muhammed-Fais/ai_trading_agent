from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
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


def _feature_names(df: pd.DataFrame) -> tuple[str, ...]:
    features = build_features(df)
    return tuple(features.dropna(axis=1, how="all").columns)


def generate_html(output_path: Path, summary: dict[str, float], train_years: str, test_years: str) -> None:
    metrics = {
        "Ending Equity": _fmt_money(summary["ending_equity"]),
        "Total Return": _fmt_pct(summary["total_return"]),
        "Max Drawdown": _fmt_pct(summary["max_drawdown"]),
        "Trades": f"{summary['trades']:,.0f}",
        "Win Rate": _fmt_pct(summary["win_rate"]),
        "Profit Factor": f"{summary['profit_factor']:.2f}",
        "Avg PnL": _fmt_money(summary["avg_trade_pnl"]),
    }
    cards = "".join(f"<div><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong></div>" for k, v in metrics.items())
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD RL Policy Report</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .metrics div {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    span {{ display: block; color: #6b7280; font-size: 13px; }}
    strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    p {{ color: #4b5563; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD Offline RL Policy</h1>
  <p>Train: {html.escape(train_years)}. Test: {html.escape(test_years)}. Offline CEM-trained linear policy; not live-learning.</p>
  <div class="metrics">{cards}</div>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-start", default="2007-01-01")
    parser.add_argument("--train-end", default="2016-01-01")
    parser.add_argument("--test-start", default="2016-01-01")
    parser.add_argument("--test-end", default="2025-01-01")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    train = df[(df.index >= args.train_start) & (df.index < args.train_end)]
    test = df[(df.index >= args.test_start) & (df.index < args.test_end)]
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
        iterations=args.iterations,
        population=args.population,
        elite_fraction=0.2,
        seed=cfg.model.random_state,
    )
    result = evaluate_policy(
        test,
        policy,
        feature_names,
        env_config,
        start_at=pd.Timestamp(args.test_start, tz="UTC"),
    )
    summary = summarize_backtest(result, cfg.backtest.initial_equity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output.with_suffix(".trades.csv"))
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(
        {"weights": policy.weights, "feature_names": feature_names, "env_config": asdict(env_config)},
        args.output.with_suffix(".joblib"),
    )
    generate_html(args.output, summary, f"{args.train_start} to {args.train_end}", f"{args.test_start} to {args.test_end}")
    print(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
