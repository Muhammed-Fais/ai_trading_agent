from __future__ import annotations

import argparse
import json

from xauusd_ml.backtest import run_backtest, summarize_backtest
from xauusd_ml.config import load_config
from xauusd_ml.data import load_ohlcv_csv
from xauusd_ml.features import build_dataset, build_features
from xauusd_ml.model import (
    fit_model,
    load_artifact,
    save_artifact,
    signal_from_probabilities,
    validate_model,
)


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    x, y = build_dataset(
        df,
        cfg.features.horizon_bars,
        cfg.features.threshold_bps,
        cfg.features.min_rows,
    )
    results = validate_model(
        x,
        y,
        cfg.validation.splits,
        cfg.validation.embargo_bars,
        cfg.model.random_state,
    )
    for result in results:
        print(f"\nFold {result.fold} log_loss={result.log_loss:.4f}")
        print(result.report)

    model = fit_model(x, y, cfg.model.random_state)
    save_artifact(model, list(x.columns), cfg.model.artifact_path)
    print(f"Saved model artifact to {cfg.model.artifact_path}")


def backtest(config_path: str) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(cfg.data.train_csv, cfg.data.timestamp_column)
    model, feature_names = load_artifact(cfg.model.artifact_path)
    result = run_backtest(
        df=df,
        model=model,
        feature_names=feature_names,
        horizon_bars=cfg.features.horizon_bars,
        min_confidence=cfg.model.min_confidence,
        initial_equity=cfg.backtest.initial_equity,
        risk_per_trade=cfg.backtest.risk_per_trade,
        spread_bps=cfg.backtest.spread_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        max_position_notional=cfg.backtest.max_position_notional,
    )
    print(json.dumps(summarize_backtest(result, cfg.backtest.initial_equity), indent=2))


def predict(config_path: str, input_path: str) -> None:
    cfg = load_config(config_path)
    df = load_ohlcv_csv(input_path, cfg.data.timestamp_column)
    model, feature_names = load_artifact(cfg.model.artifact_path)
    features = build_features(df).reindex(columns=feature_names).dropna()
    if features.empty:
        raise ValueError("Not enough rows to build a prediction feature vector.")
    latest = features.tail(1)
    probabilities = model.predict_proba(latest)[0]
    signal = signal_from_probabilities(model.classes_, probabilities, cfg.model.min_confidence)
    signal["timestamp"] = latest.index[-1].isoformat()
    print(json.dumps(signal, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="xauusd-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--config", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--input", required=True)

    args = parser.parse_args()
    if args.command == "train":
        train(args.config)
    elif args.command == "backtest":
        backtest(args.config)
    elif args.command == "predict":
        predict(args.config, args.input)


if __name__ == "__main__":
    main()
