from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class KillSwitchConfig:
    max_daily_loss_pct: float = 0.02
    max_total_drawdown_pct: float = 0.08
    max_consecutive_losses: int = 5
    max_spread_bps: float = 8.0


@dataclass
class PaperState:
    equity: float
    peak_equity: float
    daily_start_equity: float
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""


class PaperLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")


def check_kill_switches(
    state: PaperState,
    config: KillSwitchConfig,
    spread_bps: float,
) -> PaperState:
    if state.halted:
        return state
    daily_loss = state.equity / state.daily_start_equity - 1
    drawdown = state.equity / state.peak_equity - 1
    if spread_bps > config.max_spread_bps:
        state.halted = True
        state.halt_reason = "max_spread"
    elif daily_loss <= -config.max_daily_loss_pct:
        state.halted = True
        state.halt_reason = "max_daily_loss"
    elif drawdown <= -config.max_total_drawdown_pct:
        state.halted = True
        state.halt_reason = "max_total_drawdown"
    elif state.consecutive_losses >= config.max_consecutive_losses:
        state.halted = True
        state.halt_reason = "max_consecutive_losses"
    return state


def record_decision(
    ledger: PaperLedger,
    timestamp: pd.Timestamp,
    strategy_id: str,
    signal: str,
    probability: float,
    setup: str,
    state: PaperState,
    reason: str,
) -> None:
    ledger.append(
        {
            "event": "decision",
            "timestamp": timestamp,
            "strategy_id": strategy_id,
            "signal": signal,
            "probability": probability,
            "setup": setup,
            "equity": state.equity,
            "peak_equity": state.peak_equity,
            "halted": state.halted,
            "halt_reason": state.halt_reason,
            "reason": reason,
        }
    )


def save_state(path: Path, state: PaperState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def load_state(path: Path, initial_equity: float) -> PaperState:
    if not path.exists():
        return PaperState(
            equity=initial_equity,
            peak_equity=initial_equity,
            daily_start_equity=initial_equity,
        )
    return PaperState(**json.loads(path.read_text(encoding="utf-8")))

