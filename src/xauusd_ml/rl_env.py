from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xauusd_ml.features import build_features


@dataclass(frozen=True)
class RLEnvConfig:
    feature_names: tuple[str, ...]
    initial_equity: float = 10_000.0
    position_notional: float = 10_000.0
    spread_bps: float = 3.0
    slippage_bps: float = 1.0
    drawdown_penalty: float = 0.1
    turnover_penalty: float = 0.02
    flat_penalty: float = 0.0
    max_episode_steps: int = 2_000


class XAUUSDTradingEnv:
    """Small offline trading environment for policy research.

    Actions are target positions:
    0 = flat, 1 = long, 2 = short.
    """

    def __init__(self, df: pd.DataFrame, config: RLEnvConfig) -> None:
        self.df = df.copy()
        self.config = config
        self.cost = (config.spread_bps + config.slippage_bps) / 10_000
        self.features = build_features(self.df).reindex(columns=config.feature_names)
        dataset = self.features.join(self.df[["open", "close"]]).dropna()
        self.features = dataset[list(config.feature_names)]
        self.prices = dataset[["open", "close"]]
        self.feature_mean = self.features.mean()
        self.feature_std = self.features.std().replace(0, 1)
        self.normalized_features = (self.features - self.feature_mean) / self.feature_std
        self.reset()

    @property
    def observation_size(self) -> int:
        return len(self.config.feature_names) + 3

    def reset(self, start: int | None = None) -> np.ndarray:
        max_start = max(0, len(self.normalized_features) - self.config.max_episode_steps - 2)
        self.step_idx = int(start if start is not None else np.random.randint(0, max_start + 1))
        self.end_idx = min(self.step_idx + self.config.max_episode_steps, len(self.normalized_features) - 2)
        self.equity = self.config.initial_equity
        self.peak_equity = self.equity
        self.position = 0
        self.entry_price = float(self.prices["close"].iloc[self.step_idx])
        self.time_in_position = 0
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | int]]:
        target_position = {0: 0, 1: 1, 2: -1}[int(action)]
        current_close = float(self.prices["close"].iloc[self.step_idx])
        next_close = float(self.prices["close"].iloc[self.step_idx + 1])

        trade_cost = 0.0
        if target_position != self.position:
            trade_cost = self.config.position_notional * self.cost
            self.entry_price = current_close
            self.time_in_position = 0

        gross_pnl = self.position * self.config.position_notional * (next_close / current_close - 1)
        pnl = gross_pnl - trade_cost
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = self.equity / self.peak_equity - 1
        turnover = 1.0 if target_position != self.position else 0.0
        flat_penalty = self.config.flat_penalty if target_position == 0 else 0.0
        reward = (
            pnl / self.config.initial_equity
            + self.config.drawdown_penalty * drawdown
            - self.config.turnover_penalty * turnover
            - flat_penalty
        )

        self.position = target_position
        self.time_in_position = self.time_in_position + 1 if self.position else 0
        self.step_idx += 1
        done = self.step_idx >= self.end_idx or self.equity <= self.config.initial_equity * 0.5
        info = {
            "equity": self.equity,
            "pnl": pnl,
            "position": self.position,
            "drawdown": drawdown,
        }
        return self._observation(), float(reward), done, info

    def _observation(self) -> np.ndarray:
        features = self.normalized_features.iloc[self.step_idx].to_numpy(dtype=float)
        current_close = float(self.prices["close"].iloc[self.step_idx])
        unrealized = self.position * (current_close / self.entry_price - 1) if self.position else 0.0
        state = np.array(
            [
                float(self.position),
                float(self.time_in_position / max(self.config.max_episode_steps, 1)),
                float(unrealized),
            ]
        )
        return np.nan_to_num(np.concatenate([features, state]), copy=False)


class LinearSoftmaxPolicy:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights

    @classmethod
    def random(cls, observation_size: int, action_size: int, scale: float, rng: np.random.Generator):
        return cls(rng.normal(0, scale, size=(observation_size + 1, action_size)))

    def act(self, observation: np.ndarray) -> int:
        x = np.append(observation, 1.0)
        logits = x @ self.weights
        return int(np.argmax(logits))
