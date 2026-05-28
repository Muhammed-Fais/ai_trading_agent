from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.backtest import summarize_backtest  # noqa: E402
from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
from xauusd_ml.meta_label import (  # noqa: E402
    build_meta_dataset,
    generate_candidate_trades,
    make_meta_model,
    simulate_candidates,
    simulate_with_setup_thresholds,
)


def train_artifact(
    config_path: Path,
    output_path: Path,
    exclude_setups: set[str],
    validation_years: int,
    objective: str,
) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    validation_start = df.index.max() - pd.DateOffset(years=validation_years)
    train_df = df[df.index < validation_start]
    validation_df = df[df.index >= validation_start]

    train_candidates = generate_candidate_trades(train_df)
    validation_candidates = generate_candidate_trades(validation_df)
    if exclude_setups:
        train_candidates = train_candidates[~train_candidates["setup"].isin(exclude_setups)]
        validation_candidates = validation_candidates[~validation_candidates["setup"].isin(exclude_setups)]

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
    x_validation = x_validation.reindex(columns=x_train.columns, fill_value=0)
    model = make_meta_model(cfg.model.random_state)
    model.fit(x_train, y_train)
    validation_prob = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]

    thresholds: dict[str, float] = {}
    setup_summaries = {}
    for setup in sorted(validation_meta["setup"].unique()):
        setup_mask = validation_meta["setup"] == setup
        best_score = -999.0
        best_threshold = 1.01
        best_summary = None
        for threshold in (0.45, 0.50, 0.55, 0.60, 0.65):
            bt = simulate_candidates(
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
            if bt.empty:
                continue
            summary = summarize_backtest(bt, cfg.backtest.initial_equity)
            if summary["trades"] < 8:
                score = -999.0 + summary["trades"]
            elif objective == "quality":
                score = (
                    summary["profit_factor"]
                    + summary["win_rate"]
                    + summary["total_return"]
                    - abs(summary["max_drawdown"]) * 2
                )
            else:
                score = summary["total_return"] - abs(summary["max_drawdown"])
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_summary = summary
        if best_summary and best_score > 0 and best_summary["profit_factor"] >= 1.05:
            thresholds[setup] = best_threshold
            setup_summaries[setup] = best_summary

    validation_bt = simulate_with_setup_thresholds(
        validation_df,
        validation_meta,
        validation_prob,
        thresholds,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(x_train.columns),
            "thresholds": thresholds,
            "exclude_setups": sorted(exclude_setups),
            "objective": objective,
            "validation_summary": validation_summary,
            "setup_summaries": setup_summaries,
            "strategy_id": "xauusd_meta_label_macro_quality_no_trend_follow",
        },
        output_path,
    )
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "validation_summary": validation_summary,
                "setup_summaries": setup_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(validation_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-setup", action="append", default=[])
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--objective", choices=["return", "quality"], default="quality")
    args = parser.parse_args()
    train_artifact(
        args.config,
        args.output,
        set(args.exclude_setup),
        args.validation_years,
        args.objective,
    )


if __name__ == "__main__":
    main()
