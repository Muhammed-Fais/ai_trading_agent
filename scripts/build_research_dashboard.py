from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPORTS = [
    {
        "name": "Macro Meta-Label Quality",
        "slug": "macro_quality",
        "summary": "reports/xauusd_meta_label_macro_quality_no_trend_follow.summary.json",
        "folds": "reports/xauusd_meta_label_macro_quality_no_trend_follow.folds.json",
        "trades": "reports/xauusd_meta_label_macro_quality_no_trend_follow.trades.csv",
        "note": "Current best paper-trading candidate; macro features, no trend_follow, quality objective.",
    },
    {
        "name": "Macro Meta-Label",
        "slug": "macro_return",
        "summary": "reports/xauusd_meta_label_macro_no_trend_follow.summary.json",
        "folds": "reports/xauusd_meta_label_macro_no_trend_follow.folds.json",
        "trades": "reports/xauusd_meta_label_macro_no_trend_follow.trades.csv",
        "note": "Higher return, slightly lower win rate and profit factor.",
    },
    {
        "name": "Meta-Label No Trend Follow",
        "slug": "meta_no_trend",
        "summary": "reports/xauusd_meta_label_no_trend_follow.summary.json",
        "folds": "reports/xauusd_meta_label_no_trend_follow.folds.json",
        "trades": "reports/xauusd_meta_label_no_trend_follow.trades.csv",
        "note": "Best price-only meta-label variant.",
    },
    {
        "name": "Regime-Gated Barrier ML",
        "slug": "regime_barrier",
        "summary": "reports/xauusd_hourly_regime_walk_forward.summary.json",
        "folds": "reports/xauusd_hourly_regime_walk_forward.folds.json",
        "trades": "reports/xauusd_hourly_regime_walk_forward.trades.csv",
        "note": "Reduced damage versus ungated barrier ML, but too close to breakeven.",
    },
    {
        "name": "RL Walk-Forward",
        "slug": "rl_wf",
        "summary": "reports/xauusd_rl_walk_forward.summary.json",
        "folds": "reports/xauusd_rl_walk_forward.folds.json",
        "trades": "reports/xauusd_rl_walk_forward.trades.csv",
        "note": "Offline RL did not generalize in walk-forward.",
    },
]


def _load_json(path: str) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _line_svg(series: pd.Series, width: int = 1000, height: int = 280, color: str = "#2563eb") -> str:
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
    return f'<svg viewBox="0 0 {width} {height}"><polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}" /></svg>'


def _underwater_svg(series: pd.Series, width: int = 1000, height: int = 220) -> str:
    values = series.dropna().astype(float)
    if values.empty:
        return "<svg></svg>"
    y_min = min(float(values.min()), 0.0)
    scale = (height - 36) / abs(y_min) if y_min else 1
    xs = np.linspace(18, width - 18, len(values))
    step = max((width - 36) / max(len(values), 1), 1)
    bars = [
        f'<rect x="{x:.1f}" y="18" width="{step:.1f}" height="{abs(value) * scale:.1f}" rx="0" />'
        for x, value in zip(xs, values.to_numpy(), strict=True)
    ]
    return f'<svg viewBox="0 0 {width} {height}" class="underwater">{"".join(bars)}</svg>'


def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _fold_summary(fold: dict) -> dict:
    summary = fold.get("test_summary") or fold.get("summary") or {}
    return {
        "fold": fold.get("fold", ""),
        "test": f"{str(fold.get('test_start', ''))[:10]} to {str(fold.get('test_end', ''))[:10]}",
        "trades": summary.get("trades", 0),
        "return": summary.get("total_return", 0),
        "dd": summary.get("max_drawdown", 0),
        "win": summary.get("win_rate", 0),
        "pf": summary.get("profit_factor", 0),
    }


def _report_section(report: dict) -> str:
    summary = _load_json(report["summary"])
    folds = _load_json(report["folds"])
    trades = pd.read_csv(report["trades"])
    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    if "drawdown" not in trades.columns and "equity" in trades.columns:
        trades["drawdown"] = trades["equity"] / trades["equity"].cummax() - 1

    fold_rows = "".join(
        "<tr>"
        f"<td>{item['fold']}</td>"
        f"<td>{html.escape(item['test'])}</td>"
        f"<td>{item['trades']:.0f}</td>"
        f"<td>{_fmt_pct(item['return'])}</td>"
        f"<td>{_fmt_pct(item['dd'])}</td>"
        f"<td>{_fmt_pct(item['win'])}</td>"
        f"<td>{item['pf']:.2f}</td>"
        "</tr>"
        for item in (_fold_summary(fold) for fold in folds)
    )

    setup_rows = ""
    if "setup" in trades.columns:
        grouped = []
        for setup, group in trades.groupby("setup"):
            gross_profit = group.loc[group["pnl"] > 0, "pnl"].sum()
            gross_loss = group.loc[group["pnl"] < 0, "pnl"].sum()
            grouped.append(
                {
                    "setup": setup,
                    "trades": len(group),
                    "win": (group["pnl"] > 0).mean(),
                    "pnl": group["pnl"].sum(),
                    "pf": gross_profit / abs(gross_loss) if gross_loss else 0,
                }
            )
        setup_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row['setup']))}</td>"
            f"<td>{row['trades']:.0f}</td>"
            f"<td>{_fmt_pct(row['win'])}</td>"
            f"<td>{_fmt_money(row['pnl'])}</td>"
            f"<td>{row['pf']:.2f}</td>"
            "</tr>"
            for row in sorted(grouped, key=lambda x: x["pnl"], reverse=True)
        )

    metrics = "".join(
        [
            _metric_card("Return", _fmt_pct(summary["total_return"])),
            _metric_card("Max DD", _fmt_pct(summary["max_drawdown"])),
            _metric_card("Trades", f"{summary['trades']:,.0f}"),
            _metric_card("Win Rate", _fmt_pct(summary["win_rate"])),
            _metric_card("Profit Factor", f"{summary['profit_factor']:.2f}"),
            _metric_card("Avg PnL", _fmt_money(summary["avg_trade_pnl"])),
        ]
    )

    setup_table = (
        f"""
        <div class="table-wrap">
          <h3>Setup Attribution</h3>
          <table><thead><tr><th>Setup</th><th>Trades</th><th>Win</th><th>PnL</th><th>PF</th></tr></thead><tbody>{setup_rows}</tbody></table>
        </div>
        """
        if setup_rows
        else ""
    )

    return f"""
    <section id="{html.escape(report['slug'])}" class="report">
      <div class="section-head">
        <div>
          <h2>{html.escape(report['name'])}</h2>
          <p>{html.escape(report['note'])}</p>
        </div>
      </div>
      <div class="metrics">{metrics}</div>
      <div class="chart-grid">
        <div class="chart"><h3>Equity Curve</h3>{_line_svg(trades['equity'])}</div>
        <div class="chart"><h3>Underwater Drawdown</h3>{_underwater_svg(trades['drawdown'])}</div>
      </div>
      <div class="table-grid">
        <div class="table-wrap">
          <h3>Walk-Forward Folds</h3>
          <table><thead><tr><th>Fold</th><th>Test Window</th><th>Trades</th><th>Return</th><th>DD</th><th>Win</th><th>PF</th></tr></thead><tbody>{fold_rows}</tbody></table>
        </div>
        {setup_table}
      </div>
    </section>
    """


def build_dashboard(output: Path) -> None:
    summaries = [(report, _load_json(report["summary"])) for report in REPORTS if Path(report["summary"]).exists()]
    leaderboard_rows = "".join(
        "<tr>"
        f"<td><a href='#{html.escape(report['slug'])}'>{html.escape(report['name'])}</a></td>"
        f"<td>{_fmt_pct(summary['total_return'])}</td>"
        f"<td>{_fmt_pct(summary['max_drawdown'])}</td>"
        f"<td>{summary['trades']:.0f}</td>"
        f"<td>{_fmt_pct(summary['win_rate'])}</td>"
        f"<td>{summary['profit_factor']:.2f}</td>"
        "</tr>"
        for report, summary in sorted(summaries, key=lambda item: item[1]["profit_factor"], reverse=True)
    )
    best_report, best_summary = max(summaries, key=lambda item: item[1]["profit_factor"])
    sections = "\n".join(_report_section(report) for report, _ in summaries)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD Research Dashboard</title>
  <style>
    :root {{ color-scheme: light; --bg: #f6f7f9; --panel: #ffffff; --ink: #172033; --muted: #667085; --line: #e4e7ec; --blue: #2563eb; --red: #dc2626; }}
    body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 24px; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; color: #344054; }}
    p {{ color: var(--muted); line-height: 1.55; margin: 8px 0 0; }}
    a {{ color: var(--blue); text-decoration: none; }}
    .hero {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 18px; align-items: stretch; margin-bottom: 18px; }}
    .hero-panel, .report, .chart, .table-wrap {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }}
    .best {{ display: grid; gap: 10px; }}
    .best strong {{ font-size: 28px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric {{ border: 1px solid var(--line); background: #fbfcfe; border-radius: 8px; padding: 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 21px; }}
    .report {{ margin-top: 18px; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 16px; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 12px; }}
    .table-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; margin-top: 12px; }}
    svg {{ width: 100%; height: auto; display: block; background: #fff; border-radius: 6px; }}
    .underwater rect {{ fill: var(--red); opacity: .72; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f9fafb; color: #344054; font-weight: 600; }}
    @media (max-width: 860px) {{ .hero {{ grid-template-columns: 1fr; }} .table-grid {{ grid-template-columns: 1fr; overflow-x: auto; }} }}
  </style>
</head>
<body>
<main>
  <div class="hero">
    <div class="hero-panel">
      <h1>XAUUSD Research Dashboard</h1>
      <p>Consolidated walk-forward reports for the current ML, meta-label, macro, and RL experiments. The current candidate is research-grade and ready for paper logging, not live execution.</p>
      <div class="metrics">
        {_metric_card("Best Candidate", best_report["name"])}
        {_metric_card("Best PF", f"{best_summary['profit_factor']:.2f}")}
        {_metric_card("Best Win Rate", _fmt_pct(best_summary["win_rate"]))}
        {_metric_card("Best Max DD", _fmt_pct(best_summary["max_drawdown"]))}
      </div>
    </div>
    <div class="hero-panel best">
      <span>Paper Candidate</span>
      <strong>{html.escape(best_report["name"])}</strong>
      <p>{html.escape(best_report["note"])}</p>
    </div>
  </div>

  <section class="report">
    <h2>Leaderboard</h2>
    <div class="table-wrap" style="padding:0; border:0; margin-top:12px;">
      <table><thead><tr><th>System</th><th>Return</th><th>Max DD</th><th>Trades</th><th>Win</th><th>PF</th></tr></thead><tbody>{leaderboard_rows}</tbody></table>
    </div>
  </section>

  {sections}
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("reports/research_dashboard.html"), type=Path)
    args = parser.parse_args()
    build_dashboard(args.output)


if __name__ == "__main__":
    main()

