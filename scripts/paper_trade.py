from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
from xauusd_ml.meta_label import build_meta_features, generate_candidate_trades  # noqa: E402
from xauusd_ml.paper import (  # noqa: E402
    KillSwitchConfig,
    PaperLedger,
    check_kill_switches,
    load_state,
    record_decision,
    save_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candles", required=True, type=Path)
    parser.add_argument("--state", default=Path("paper/state.json"), type=Path)
    parser.add_argument("--ledger", default=Path("paper/ledger.jsonl"), type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--strategy-id", default="xauusd_meta_label_macro_quality_no_trend_follow")
    parser.add_argument("--spread-bps", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    candles = load_ohlcv_csv(args.candles, cfg.data.timestamp_column)
    latest_timestamp = pd.Timestamp(candles.index[-1])
    state = load_state(args.state, cfg.backtest.initial_equity)
    state = check_kill_switches(state, KillSwitchConfig(), args.spread_bps)
    ledger = PaperLedger(args.ledger)
    artifact = joblib.load(args.artifact)
    candidates = generate_candidate_trades(candles)
    excluded = set(artifact.get("exclude_setups", []))
    if excluded and not candidates.empty:
        candidates = candidates[~candidates["setup"].isin(excluded)]
    latest_candidates = candidates[candidates["timestamp"] == latest_timestamp].reset_index(drop=True)

    signal = "flat"
    probability = 0.0
    setup = "none"
    reason = "no_candidate"
    if state.halted:
        reason = f"halted_{state.halt_reason}"
    elif not latest_candidates.empty:
        x, meta = build_meta_features(candles, latest_candidates)
        x = x.reindex(columns=artifact["feature_columns"], fill_value=0)
        model = artifact["model"]
        probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)]
        thresholds = artifact["thresholds"]
        scored = []
        for row, prob in zip(meta.itertuples(index=False), probabilities, strict=True):
            threshold = thresholds.get(row.setup, 1.01)
            scored.append((float(prob), threshold, row.side, row.setup))
        accepted = [item for item in scored if item[0] >= item[1]]
        if accepted:
            probability, _, signal, setup = max(accepted, key=lambda item: item[0])
            reason = "accepted"
        else:
            probability, _, signal, setup = max(scored, key=lambda item: item[0])
            signal = "flat"
            reason = "below_threshold"

    record_decision(
        ledger=ledger,
        timestamp=latest_timestamp,
        strategy_id=args.strategy_id,
        signal=signal,
        probability=probability,
        setup=setup,
        state=state,
        reason=reason,
    )
    save_state(args.state, state)
    print(f"recorded paper decision at {latest_timestamp}")
    print(f"signal={signal} probability={probability:.4f} setup={setup} reason={reason}")
    print(f"halted={state.halted} reason={state.halt_reason}")


if __name__ == "__main__":
    main()
