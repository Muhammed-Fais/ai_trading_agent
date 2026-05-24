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
from xauusd_ml.meta_label import build_meta_dataset, generate_candidate_trades, make_meta_model  # noqa: E402


def _fmt_pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-01-01", tz="UTC")


def _simulate_candidates(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    min_probability: float,
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> pd.DataFrame:
    from xauusd_ml.features import build_features

    features = build_features(df)
    cost = (spread_bps + slippage_bps) / 10_000
    equity = initial_equity
    next_available_idx = 0
    rows = []

    for row, probability in zip(candidates.itertuples(index=False), probabilities, strict=True):
        if probability < min_probability:
            continue
        idx = df.index.get_indexer([row.timestamp])[0]
        if idx < 0 or idx + 1 >= len(df) or idx < next_available_idx:
            continue
        atr_pct = float(features["atr_pct_14"].iloc[idx])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            continue

        direction = 1 if row.side == "long" else -1
        entry_idx = idx + 1
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
                "signal": row.side,
                "setup": row.setup,
                "probability": probability,
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
                "setup",
                "probability",
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


def _simulate_with_setup_thresholds(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    initial_equity: float,
    risk_per_trade: float,
    spread_bps: float,
    slippage_bps: float,
    max_position_notional: float,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> pd.DataFrame:
    keep = np.array(
        [
            probability >= thresholds.get(setup, 1.01)
            for setup, probability in zip(candidates["setup"], probabilities, strict=True)
        ]
    )
    if not keep.any():
        return _simulate_candidates(
            df,
            candidates.iloc[0:0],
            np.array([]),
            0.0,
            initial_equity,
            risk_per_trade,
            spread_bps,
            slippage_bps,
            max_position_notional,
            max_hold_bars,
            stop_atr_multiple,
            take_profit_atr_multiple,
        )
    return _simulate_candidates(
        df,
        candidates[keep].reset_index(drop=True),
        probabilities[keep],
        0.0,
        initial_equity,
        risk_per_trade,
        spread_bps,
        slippage_bps,
        max_position_notional,
        max_hold_bars,
        stop_atr_multiple,
        take_profit_atr_multiple,
    )


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
    return f'<svg viewBox="0 0 {width} {height}"><polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}" /></svg>'


def run_walk_forward(
    config_path: Path,
    output_path: Path,
    first_test_year: int,
    last_test_year: int,
    train_years: int,
    validation_years: int,
    test_years: int,
    exclude_setups: set[str],
) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    curves = []
    fold_rows = []
    equity_offset = cfg.backtest.initial_equity
    blocked_setups: set[str] = set()

    for fold, test_year in enumerate(range(first_test_year, last_test_year + 1, test_years), start=1):
        train_start = _date(test_year - validation_years - train_years)
        validation_start = _date(test_year - validation_years)
        test_start = _date(test_year)
        test_end = _date(test_year + test_years)

        train_df = df[(df.index >= train_start) & (df.index < validation_start)]
        validation_df = df[(df.index >= validation_start) & (df.index < test_start)]
        test_df = df[(df.index >= test_start) & (df.index < test_end)]

        train_candidates = generate_candidate_trades(train_df)
        validation_candidates = generate_candidate_trades(validation_df)
        test_candidates = generate_candidate_trades(test_df)
        if exclude_setups:
            train_candidates = train_candidates[~train_candidates["setup"].isin(exclude_setups)]
            validation_candidates = validation_candidates[~validation_candidates["setup"].isin(exclude_setups)]
            test_candidates = test_candidates[~test_candidates["setup"].isin(exclude_setups)]
        if train_candidates.empty or validation_candidates.empty or test_candidates.empty:
            continue

        x_train, y_train, _ = build_meta_dataset(
            train_df,
            train_candidates,
            cfg.strategy.max_hold_bars,
            cfg.strategy.stop_atr_multiple,
            cfg.strategy.take_profit_atr_multiple,
        )
        x_validation, _, validation_meta = build_meta_dataset(
            validation_df,
            validation_candidates,
            cfg.strategy.max_hold_bars,
            cfg.strategy.stop_atr_multiple,
            cfg.strategy.take_profit_atr_multiple,
        )
        x_test, _, test_meta = build_meta_dataset(
            test_df,
            test_candidates,
            cfg.strategy.max_hold_bars,
            cfg.strategy.stop_atr_multiple,
            cfg.strategy.take_profit_atr_multiple,
        )
        x_validation = x_validation.reindex(columns=x_train.columns, fill_value=0)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        model = make_meta_model(cfg.model.random_state + fold)
        model.fit(x_train, y_train)
        validation_prob = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]

        setup_thresholds: dict[str, float] = {}
        setup_validation: dict[str, dict[str, float]] = {}
        for setup in sorted(validation_meta["setup"].unique()):
            setup_mask = validation_meta["setup"] == setup
            best_score = -999.0
            best_threshold = 1.01
            best_summary: dict[str, float] = {
                "trades": 0.0,
                "total_return": -1.0,
                "max_drawdown": -1.0,
                "profit_factor": 0.0,
            }
            for threshold in (0.45, 0.50, 0.55, 0.60, 0.65):
                validation_bt = _simulate_candidates(
                    validation_df,
                    validation_meta[setup_mask].reset_index(drop=True),
                    validation_prob[setup_mask],
                    threshold,
                    cfg.backtest.initial_equity,
                    cfg.backtest.risk_per_trade,
                    cfg.backtest.spread_bps,
                    cfg.backtest.slippage_bps,
                    cfg.backtest.max_position_notional,
                    cfg.strategy.max_hold_bars,
                    cfg.strategy.stop_atr_multiple,
                    cfg.strategy.take_profit_atr_multiple,
                )
                if validation_bt.empty:
                    continue
                summary = summarize_backtest(validation_bt, cfg.backtest.initial_equity)
                if summary["trades"] < 8:
                    score = -999.0 + summary["trades"]
                else:
                    score = summary["total_return"] - abs(summary["max_drawdown"])
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_summary = summary

            if (
                setup not in blocked_setups
                and best_score > 0
                and best_summary["profit_factor"] >= 1.05
                and best_summary["max_drawdown"] > -0.03
            ):
                setup_thresholds[setup] = best_threshold
            setup_validation[setup] = best_summary

        if not setup_thresholds:
            threshold_scores = []
            for threshold in (0.45, 0.50, 0.55, 0.60):
                validation_bt = _simulate_candidates(
                    validation_df,
                    validation_meta,
                    validation_prob,
                    threshold,
                    cfg.backtest.initial_equity,
                    cfg.backtest.risk_per_trade,
                    cfg.backtest.spread_bps,
                    cfg.backtest.slippage_bps,
                    cfg.backtest.max_position_notional,
                    cfg.strategy.max_hold_bars,
                    cfg.strategy.stop_atr_multiple,
                    cfg.strategy.take_profit_atr_multiple,
                )
                if validation_bt.empty:
                    score = -999.0
                    summary = {
                        "trades": 0.0,
                        "total_return": -1.0,
                        "max_drawdown": -1.0,
                        "profit_factor": 0.0,
                    }
                else:
                    summary = summarize_backtest(validation_bt, cfg.backtest.initial_equity)
                    score = (
                        -999.0 + summary["trades"]
                        if summary["trades"] < 20
                        else summary["total_return"] - abs(summary["max_drawdown"])
                    )
                threshold_scores.append((score, threshold, summary))
            _, selected_threshold, validation_summary = max(threshold_scores, key=lambda item: item[0])
            setup_thresholds = {setup: selected_threshold for setup in validation_meta["setup"].unique()}
        else:
            validation_bt = _simulate_with_setup_thresholds(
                validation_df,
                validation_meta,
                validation_prob,
                setup_thresholds,
                cfg.backtest.initial_equity,
                cfg.backtest.risk_per_trade,
                cfg.backtest.spread_bps,
                cfg.backtest.slippage_bps,
                cfg.backtest.max_position_notional,
                cfg.strategy.max_hold_bars,
                cfg.strategy.stop_atr_multiple,
                cfg.strategy.take_profit_atr_multiple,
            )
            validation_summary = summarize_backtest(validation_bt, cfg.backtest.initial_equity)

        test_prob = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        test_bt = _simulate_with_setup_thresholds(
            test_df,
            test_meta,
            test_prob,
            setup_thresholds,
            cfg.backtest.initial_equity,
            cfg.backtest.risk_per_trade,
            cfg.backtest.spread_bps,
            cfg.backtest.slippage_bps,
            cfg.backtest.max_position_notional,
            cfg.strategy.max_hold_bars,
            cfg.strategy.stop_atr_multiple,
            cfg.strategy.take_profit_atr_multiple,
        )
        if test_bt.empty:
            continue
        test_summary = summarize_backtest(test_bt, cfg.backtest.initial_equity)
        curve = test_bt.copy()
        curve["fold"] = fold
        curve["fold_equity"] = curve["equity"]
        curve["equity"] = equity_offset + (curve["equity"] - cfg.backtest.initial_equity)
        equity_offset = float(curve["equity"].iloc[-1])
        curves.append(curve)
        fold_rows.append(
            {
                "fold": fold,
                "test_start": str(test_start),
                "test_end": str(test_end - pd.Timedelta(hours=1)),
                "thresholds": setup_thresholds,
                "setup_validation": setup_validation,
                "validation_summary": validation_summary,
                "test_summary": test_summary,
            }
        )
        failed_setups = set()
        for setup, group in test_bt.groupby("setup"):
            setup_summary = summarize_backtest(group, cfg.backtest.initial_equity)
            if setup_summary["profit_factor"] < 0.9 or setup_summary["total_return"] < -0.03:
                failed_setups.add(str(setup))
        blocked_setups = failed_setups
        print(
            f"fold={fold} setups={','.join(sorted(setup_thresholds))} trades={test_summary['trades']:.0f} "
            f"return={test_summary['total_return']:.2%} dd={test_summary['max_drawdown']:.2%} "
            f"pf={test_summary['profit_factor']:.2f}"
        )

    if not curves:
        raise ValueError("No meta-label folds produced trades.")

    combined = pd.concat(curves).sort_index()
    combined["drawdown"] = combined["equity"] / combined["equity"].cummax() - 1
    summary = summarize_backtest(combined, cfg.backtest.initial_equity)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path.with_suffix(".trades.csv"))
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output_path.with_suffix(".folds.json").write_text(json.dumps(fold_rows, indent=2), encoding="utf-8")

    rows = "".join(
        "<tr>"
        f"<td>{row['fold']}</td><td>{html.escape(row['test_start'][:10])} to {html.escape(row['test_end'][:10])}</td>"
        f"<td>{html.escape(', '.join(sorted(row['thresholds'])))}</td><td>{row['test_summary']['trades']:.0f}</td>"
        f"<td>{_fmt_pct(row['test_summary']['total_return'])}</td>"
        f"<td>{_fmt_pct(row['test_summary']['max_drawdown'])}</td>"
        f"<td>{_fmt_pct(row['test_summary']['win_rate'])}</td>"
        f"<td>{row['test_summary']['profit_factor']:.2f}</td></tr>"
        for row in fold_rows
    )
    cards = "".join(
        f"<div><span>{label}</span><strong>{value}</strong></div>"
        for label, value in {
            "Ending Equity": _fmt_money(summary["ending_equity"]),
            "Total Return": _fmt_pct(summary["total_return"]),
            "Max Drawdown": _fmt_pct(summary["max_drawdown"]),
            "Trades": f"{summary['trades']:,.0f}",
            "Win Rate": _fmt_pct(summary["win_rate"]),
            "Profit Factor": f"{summary['profit_factor']:.2f}",
        }.items()
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XAUUSD Meta-Label Walk-Forward</title>
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
  <h1>XAUUSD Meta-Label Walk-Forward</h1>
  <p>Rule-generated candidate trades filtered by an ML accept/reject model.</p>
  <div class="metrics">{cards}</div>
  <section><h2>Combined Equity</h2>{_line_svg(combined["equity"])}</section>
  <section>
    <h2>Folds</h2>
    <table>
      <thead><tr><th>Fold</th><th>Test Window</th><th>Setups</th><th>Trades</th><th>Return</th><th>DD</th><th>Win</th><th>PF</th></tr></thead>
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
    parser.add_argument("--validation-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--exclude-setup", action="append", default=[])
    args = parser.parse_args()
    run_walk_forward(
        args.config,
        args.output,
        args.first_test_year,
        args.last_test_year,
        args.train_years,
        args.validation_years,
        args.test_years,
        set(args.exclude_setup),
    )


if __name__ == "__main__":
    main()
