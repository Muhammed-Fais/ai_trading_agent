from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.config import load_config  # noqa: E402
from xauusd_ml.data import load_ohlcv_csv  # noqa: E402
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
    parser.add_argument("--strategy-id", default="xauusd_meta_label_macro_quality_no_trend_follow")
    parser.add_argument("--spread-bps", type=float, default=3.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    candles = load_ohlcv_csv(args.candles, cfg.data.timestamp_column)
    latest_timestamp = pd.Timestamp(candles.index[-1])
    state = load_state(args.state, cfg.backtest.initial_equity)
    state = check_kill_switches(state, KillSwitchConfig(), args.spread_bps)
    ledger = PaperLedger(args.ledger)

    # This is intentionally conservative scaffolding. The live model adapter will fill
    # signal/probability/setup once the paper pipeline is connected to trained artifacts.
    record_decision(
        ledger=ledger,
        timestamp=latest_timestamp,
        strategy_id=args.strategy_id,
        signal="flat",
        probability=0.0,
        setup="bootstrap",
        state=state,
        reason="paper_scaffold_no_live_model_loaded",
    )
    save_state(args.state, state)
    print(f"recorded paper decision at {latest_timestamp}")
    print(f"halted={state.halted} reason={state.halt_reason}")


if __name__ == "__main__":
    main()

