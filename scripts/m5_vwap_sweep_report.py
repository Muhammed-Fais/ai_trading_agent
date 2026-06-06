from __future__ import annotations

import argparse
import html
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from m5_strategy_report import (  # noqa: E402
    _fmt_money,
    _fmt_pct,
    _line_svg,
    _underwater_svg,
    _vwap_cross_6am_entries,
    run_vwap_cross_backtest,
)
from xauusd_ml.backtest import summarize_backtest  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402


GRID = {
    "include_shorts": [False],
    "trade_end_hour": [10, 12, 16],
    "min_distance_bps": [0.0, 4.0, 8.0],
    "trend_filter": [False, True],
    "max_confirm_bars": [3, 6, 12],
    "reward_risk": [1.5, 2.0, 3.0],
    "max_hold_bars": [12, 24, 36],
    "stop_buffer_bps": [0.0],
}


def _empty_summary(initial_equity: float) -> dict[str, float]:
    return {
        "ending_equity": initial_equity,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "trades": 0.0,
        "win_rate": 0.0,
        "avg_trade_pnl": 0.0,
        "profit_factor": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }


def _score(summary: dict[str, float]) -> float:
    if summary["trades"] < 30 or summary["profit_factor"] <= 1.0:
        return -999.0
    return (
        summary["profit_factor"]
        + summary["total_return"] * 3
        + min(summary["trades"], 160) / 500
        + summary["max_drawdown"]
    )


def _run_variant(df: pd.DataFrame, cfg, params: dict, spread_mult: float = 1.0) -> tuple[pd.DataFrame, dict[str, float]]:
    entries = _vwap_cross_6am_entries(
        df,
        include_shorts=params["include_shorts"],
        trade_end_hour=params["trade_end_hour"],
        min_distance_bps=params["min_distance_bps"],
        trend_filter=params["trend_filter"],
        max_confirm_bars=params["max_confirm_bars"],
    )
    bt = run_vwap_cross_backtest(
        df,
        entries,
        cfg.backtest.initial_equity,
        cfg.backtest.risk_per_trade,
        cfg.backtest.spread_bps * spread_mult,
        cfg.backtest.slippage_bps * spread_mult,
        cfg.backtest.max_position_notional,
        params["max_hold_bars"],
        reward_risk=params["reward_risk"],
        stop_buffer_bps=params["stop_buffer_bps"],
    )
    summary = summarize_backtest(bt, cfg.backtest.initial_equity) if not bt.empty else _empty_summary(
        cfg.backtest.initial_equity
    )
    return bt, summary


def _param_grid() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(GRID[key] for key in keys))]


def build_report(config_path: Path, output_path: Path, top_n: int) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    split_idx = int(len(df) * 0.7)
    split_at = df.index[split_idx]
    train_df = df[df.index < split_at]
    test_df = df[df.index >= split_at]

    rows = []
    for params in _param_grid():
        _, train_summary = _run_variant(train_df, cfg, params)
        rows.append({"params": params, "train": train_summary, "score": _score(train_summary)})

    selected = sorted(rows, key=lambda item: item["score"], reverse=True)[:top_n]
    evaluated = []
    for item in selected:
        test_bt, test_summary = _run_variant(test_df, cfg, item["params"])
        _, stress_summary = _run_variant(test_df, cfg, item["params"], spread_mult=2.0)
        evaluated.append(
            {
                "params": item["params"],
                "train": item["train"],
                "test": test_summary,
                "stress_2x": stress_summary,
                "test_trades": test_bt,
                "score": item["score"],
            }
        )

    best = evaluated[0]
    best_trades = best["test_trades"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "data_start": str(df.index.min()),
                "data_end": str(df.index.max()),
                "split_at": str(split_at),
                "best_params": best["params"],
                "best_train": best["train"],
                "best_test": best["test"],
                "best_stress_2x": best["stress_2x"],
                "top": [
                    {
                        "params": item["params"],
                        "train": item["train"],
                        "test": item["test"],
                        "stress_2x": item["stress_2x"],
                        "score": item["score"],
                    }
                    for item in evaluated
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not best_trades.empty:
        best_trades.to_csv(output_path.with_suffix(".trades.csv"))

    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(json.dumps(item['params'], sort_keys=True))}</td>"
        f"<td>{_fmt_pct(item['train']['total_return'])}</td>"
        f"<td>{item['train']['profit_factor']:.2f}</td>"
        f"<td>{item['train']['trades']:.0f}</td>"
        f"<td>{_fmt_pct(item['test']['total_return'])}</td>"
        f"<td>{_fmt_pct(item['test']['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(item['test']['win_rate'])}</td>"
        f"<td>{item['test']['profit_factor']:.2f}</td>"
        f"<td>{item['test']['trades']:.0f}</td>"
        f"<td>{item['stress_2x']['profit_factor']:.2f}</td>"
        "</tr>"
        for item in sorted(evaluated, key=lambda candidate: candidate["test"]["profit_factor"], reverse=True)
    )
    best_equity = _line_svg(best_trades["equity"]) if not best_trades.empty else "<svg></svg>"
    best_underwater = _underwater_svg(best_trades["drawdown"]) if not best_trades.empty else "<svg></svg>"
    best_metrics = "".join(
        [
            f"<div><span>Test Return</span><strong>{_fmt_pct(best['test']['total_return'])}</strong></div>",
            f"<div><span>Test Max DD</span><strong>{_fmt_pct(best['test']['max_drawdown'])}</strong></div>",
            f"<div><span>Test Trades</span><strong>{best['test']['trades']:.0f}</strong></div>",
            f"<div><span>Test Win</span><strong>{_fmt_pct(best['test']['win_rate'])}</strong></div>",
            f"<div><span>Test PF</span><strong>{best['test']['profit_factor']:.2f}</strong></div>",
            f"<div><span>Avg PnL</span><strong>{_fmt_money(best['test']['avg_trade_pnl'])}</strong></div>",
            f"<div><span>2x Cost PF</span><strong>{best['stress_2x']['profit_factor']:.2f}</strong></div>",
        ]
    )
    verdict = "WATCH" if best["test"]["profit_factor"] >= 1.1 else "FAIL"
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD M5 VWAP Sweep</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f7f8fb; color:#111827; }}
    main {{ max-width:1280px; margin:0 auto; padding:32px 20px 52px; }}
    section {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:18px; margin-top:16px; overflow-x:auto; }}
    h1 {{ margin:0; font-size:32px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:20px; }}
    p {{ color:#4b5563; line-height:1.5; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; vertical-align:top; }}
    th {{ background:#f9fafb; white-space:nowrap; }}
    td:first-child {{ min-width:420px; white-space:normal; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-top:14px; }}
    .metrics div {{ border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#fbfcfe; }}
    .metrics span {{ display:block; color:#667085; font-size:12px; }}
    .metrics strong {{ display:block; margin-top:5px; font-size:20px; }}
    .badge {{ display:inline-block; border-radius:6px; padding:5px 9px; background:#eef2ff; color:#1d4ed8; font-weight:700; }}
    svg {{ display:block; width:100%; height:auto; border:1px solid #eef2f7; border-radius:6px; background:#fff; }}
    .underwater rect {{ fill:#dc2626; opacity:.72; }}
  </style>
</head>
<body>
<main>
  <h1>XAUUSD M5 VWAP Sweep</h1>
  <p>Train window: {train_df.index.min()} to {train_df.index.max()}. Test window: {test_df.index.min()} to {test_df.index.max()}. Variants are selected on train only, then evaluated on unseen later data and doubled-cost stress.</p>
  <section>
    <h2>Selected Variant <span class="badge">{verdict}</span></h2>
    <p>{html.escape(json.dumps(best['params'], sort_keys=True))}</p>
    <div class="metrics">{best_metrics}</div>
  </section>
  <section>
    <h2>Selected Variant Equity</h2>
    {best_equity}
  </section>
  <section>
    <h2>Selected Variant Underwater Drawdown</h2>
    {best_underwater}
  </section>
  <section>
    <h2>Top Train-Selected Variants</h2>
    <table>
      <thead><tr><th>Params</th><th>Train Ret</th><th>Train PF</th><th>Train Trades</th><th>Test Ret</th><th>Test DD</th><th>Test Win</th><th>Test PF</th><th>Test Trades</th><th>2x PF</th></tr></thead>
      <tbody>{table_rows}</tbody>
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
    parser.add_argument("--config", default=Path("config/report_m5.yaml"), type=Path)
    parser.add_argument("--output", default=Path("reports/xauusd_m5_vwap_sweep.html"), type=Path)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    build_report(args.config, args.output, args.top_n)


if __name__ == "__main__":
    main()
