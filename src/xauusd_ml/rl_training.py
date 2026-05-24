from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ml.rl_env import LinearSoftmaxPolicy, RLEnvConfig, XAUUSDTradingEnv


def run_episode(env: XAUUSDTradingEnv, policy: LinearSoftmaxPolicy, start: int | None = None) -> float:
    observation = env.reset(start=start)
    total_reward = 0.0
    done = False
    while not done:
        observation, reward, done, _ = env.step(policy.act(observation))
        total_reward += reward
    return total_reward


def train_cem(
    env: XAUUSDTradingEnv,
    iterations: int,
    population: int,
    elite_fraction: float,
    seed: int,
) -> LinearSoftmaxPolicy:
    rng = np.random.default_rng(seed)
    observation_size = env.observation_size
    action_size = 3
    mean = np.zeros((observation_size + 1, action_size))
    std = np.ones_like(mean) * 0.4
    elite_count = max(2, int(population * elite_fraction))

    for iteration in range(iterations):
        candidates = []
        for _ in range(population):
            weights = rng.normal(mean, std)
            policy = LinearSoftmaxPolicy(weights)
            score = np.mean([run_episode(env, policy) for _ in range(3)])
            candidates.append((score, weights))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = np.array([weights for _, weights in candidates[:elite_count]])
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.03)
        print(f"iteration={iteration + 1} best_reward={candidates[0][0]:.4f}")

    return LinearSoftmaxPolicy(mean)


def evaluate_policy(
    df: pd.DataFrame,
    policy: LinearSoftmaxPolicy,
    feature_names: tuple[str, ...],
    config: RLEnvConfig,
    start_at: pd.Timestamp,
) -> pd.DataFrame:
    env = XAUUSDTradingEnv(df, config)
    observation = env.reset(start=0)
    rows = []
    done = False
    while not done:
        timestamp = env.normalized_features.index[env.step_idx]
        action = policy.act(observation)
        signal = {0: "flat", 1: "long", 2: "short"}[action]
        observation, _, done, info = env.step(action)
        if timestamp < start_at:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "signal": signal,
                "pnl": float(info["pnl"]),
                "equity": float(info["equity"]),
                "position": int(info["position"]),
                "drawdown": float(info["drawdown"]),
                "notional": config.position_notional if signal != "flat" else 0.0,
            }
        )
    result = pd.DataFrame(rows).set_index("timestamp")
    result["trade"] = result["signal"] != "flat"
    result["win"] = np.where(result["trade"], result["pnl"] > 0, np.nan)
    result["entry"] = np.nan
    result["exit"] = np.nan
    result["net_return"] = result["pnl"] / config.position_notional
    return result

