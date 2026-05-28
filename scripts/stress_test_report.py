from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import yaml


VARIANTS = [
    {"name": "base", "spread_mult": 1.0, "slippage_mult": 1.0, "risk_mult": 1.0},
    {"name": "cost_2x", "spread_mult": 2.0, "slippage_mult": 2.0, "risk_mult": 1.0},
    {"name": "cost_3x", "spread_mult": 3.0, "slippage_mult": 3.0, "risk_mult": 1.0},
    {"name": "half_risk", "spread_mult": 1.0, "slippage_mult": 1.0, "risk_mult": 0.5},
]


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _write_variant_config(base_config: Path, output: Path, variant: dict) -> None:
    raw = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    raw["backtest"]["spread_bps"] *= variant["spread_mult"]
    raw["backtest"]["slippage_bps"] *= variant["slippage_mult"]
    raw["backtest"]["risk_per_trade"] *= variant["risk_mult"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _run_variant(base_config: Path, output_dir: Path, variant: dict) -> dict:
    config_path = output_dir / f"{variant['name']}.yaml"
    report_path = output_dir / f"{variant['name']}.html"
    _write_variant_config(base_config, config_path, variant)
    subprocess.run(
        [
            sys.executable,
            "scripts/meta_label_report.py",
            "--config",
            str(config_path),
            "--output",
            str(report_path),
            "--first-test-year",
            "2014",
            "--last-test-year",
            "2024",
            "--train-years",
            "7",
            "--validation-years",
            "1",
            "--test-years",
            "2",
            "--exclude-setup",
            "trend_follow",
            "--objective",
            "quality",
        ],
        check=True,
    )
    summary = json.loads(report_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    return {"variant": variant, "report": str(report_path), "summary": summary}


def build_stress_report(base_config: Path, output: Path, output_dir: Path) -> None:
    results = [_run_variant(base_config, output_dir, variant) for variant in VARIANTS]
    rows = "".join(
        "<tr>"
        f"<td><a href='{html.escape(str(Path(result['report']).relative_to(output.parent)))}'>"
        f"{html.escape(result['variant']['name'])}</a></td>"
        f"<td>{result['variant']['spread_mult']:.1f}x</td>"
        f"<td>{result['variant']['slippage_mult']:.1f}x</td>"
        f"<td>{result['variant']['risk_mult']:.1f}x</td>"
        f"<td>{_fmt_pct(result['summary']['total_return'])}</td>"
        f"<td>{_fmt_pct(result['summary']['max_drawdown'])}</td>"
        f"<td>{result['summary']['trades']:.0f}</td>"
        f"<td>{_fmt_pct(result['summary']['win_rate'])}</td>"
        f"<td>{result['summary']['profit_factor']:.2f}</td>"
        "</tr>"
        for result in results
    )
    base_summary = results[0]["summary"]
    cost_2x_summary = results[1]["summary"]
    verdict = (
        "PASS"
        if base_summary["profit_factor"] >= 1.2
        and cost_2x_summary["profit_factor"] >= 1.2
        and cost_2x_summary["max_drawdown"] > -0.15
        else "WATCH"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD Stress Test</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f8fafc; color:#111827; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px; }}
    section {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:18px; }}
    .verdict {{ display:inline-block; padding:6px 10px; border-radius:6px; background:#eef2ff; color:#1d4ed8; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; }}
    th {{ background:#f9fafb; }}
    p {{ color:#475467; }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>XAUUSD Stress Test</h1>
    <p>Current candidate: macro-enriched meta-label, no trend_follow, quality objective, H4/D1 context features.</p>
    <p>Verdict: <span class="verdict">{verdict}</span></p>
    <table>
      <thead><tr><th>Variant</th><th>Spread</th><th>Slippage</th><th>Risk</th><th>Return</th><th>Max DD</th><th>Trades</th><th>Win</th><th>PF</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path("config/report_hourly_macro_barrier.yaml"), type=Path)
    parser.add_argument("--output", default=Path("reports/stress_test.html"), type=Path)
    parser.add_argument("--output-dir", default=Path("reports/stress"), type=Path)
    args = parser.parse_args()
    build_stress_report(args.config, args.output, args.output_dir)


if __name__ == "__main__":
    main()
