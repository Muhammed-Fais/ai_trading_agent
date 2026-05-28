from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xauusd_ml.features import build_features


@dataclass(frozen=True)
class CandidateTrade:
    timestamp: pd.Timestamp
    side: str
    setup: str


def _signal_side(long_mask: pd.Series, short_mask: pd.Series) -> pd.Series:
    side = pd.Series("flat", index=long_mask.index, dtype="object")
    side[long_mask & ~short_mask] = "long"
    side[short_mask & ~long_mask] = "short"
    return side


def _regime_masks(df: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"]
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_288 = close.ewm(span=288, adjust=False).mean()
    trend_strength = (ema_55 / ema_288 - 1).abs()
    trend_ok = trend_strength > trend_strength.rolling(500).median()
    atr = features["atr_pct_14"]
    atr_median = atr.rolling(500).median()
    atr_q75 = atr.rolling(500).quantile(0.75)
    atr_q90 = atr.rolling(500).quantile(0.90)
    normal_vol = (atr > atr_median) & (atr < atr_q90)
    expansion_vol = (atr > atr_q75) & (atr < atr_q90)
    calm_vol = atr < atr_q75
    not_asia = ~features["is_asia_session"].astype(bool)
    london_ny = features["is_london_ny_overlap"].astype(bool)

    return {
        "trend": trend_ok & normal_vol & not_asia,
        "breakout": expansion_vol & not_asia,
        "reversal": calm_vol & not_asia,
        "session": london_ny & normal_vol,
        "volatility": expansion_vol & not_asia,
        "any": pd.Series(True, index=df.index),
    }


def generate_candidate_trades(df: pd.DataFrame, use_regime_gate: bool = False) -> pd.DataFrame:
    close = df["close"]
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_144 = close.ewm(span=144, adjust=False).mean()
    donchian_high = df["high"].rolling(96).max().shift(1)
    donchian_low = df["low"].rolling(96).min().shift(1)
    features = build_features(df)
    regimes = _regime_masks(df, features)
    atr = features["atr_pct_14"]
    atr_ok = atr > atr.rolling(500).median()
    london_ny = features["is_london_ny_overlap"].astype(bool)

    setups = {
        "trend_pullback": _signal_side(
            (ema_21 > ema_55) & (ema_55 > ema_144) & (close > ema_21) & (close.shift(1) <= ema_21.shift(1)),
            (ema_21 < ema_55) & (ema_55 < ema_144) & (close < ema_21) & (close.shift(1) >= ema_21.shift(1)),
        ),
        "trend_follow": _signal_side(
            (ema_21 > ema_55) & (ema_55 > ema_144) & (features["ret_4"] > 0),
            (ema_21 < ema_55) & (ema_55 < ema_144) & (features["ret_4"] < 0),
        ),
        "donchian_breakout": _signal_side((close > donchian_high) & atr_ok, (close < donchian_low) & atr_ok),
        "donchian_reversal": _signal_side(
            (features["donchian_pos_96"] < 0.12) & (features["rsi_14"] < 0.35),
            (features["donchian_pos_96"] > 0.88) & (features["rsi_14"] > 0.65),
        ),
        "session_momentum": _signal_side(
            london_ny & (features["ret_4"] > features["realized_vol_24"] * 1.5),
            london_ny & (features["ret_4"] < -features["realized_vol_24"] * 1.5),
        ),
        "volatility_expansion": _signal_side(
            atr_ok & (features["ret_8"] > features["realized_vol_96"] * 1.2),
            atr_ok & (features["ret_8"] < -features["realized_vol_96"] * 1.2),
        ),
    }
    setup_regime = {
        "trend_pullback": "trend",
        "trend_follow": "trend",
        "donchian_breakout": "breakout",
        "donchian_reversal": "reversal",
        "session_momentum": "session",
        "volatility_expansion": "volatility",
    }

    rows = []
    for setup, sides in setups.items():
        if use_regime_gate:
            sides = sides.mask(~regimes[setup_regime[setup]], "flat")
        active = sides[sides != "flat"]
        for timestamp, side in active.items():
            rows.append({"timestamp": timestamp, "side": side, "setup": setup})
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return pd.DataFrame(columns=["timestamp", "side", "setup"])
    return candidates.drop_duplicates(["timestamp", "side", "setup"]).sort_values("timestamp")


def label_candidates(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> pd.Series:
    features = build_features(df)
    labels = []
    index = df.index

    for row in candidates.itertuples(index=False):
        idx = index.get_indexer([row.timestamp])[0]
        if idx < 0 or idx + 1 >= len(df):
            labels.append(np.nan)
            continue
        atr_pct = float(features["atr_pct_14"].iloc[idx])
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            labels.append(np.nan)
            continue

        direction = 1 if row.side == "long" else -1
        entry_idx = idx + 1
        entry = float(df["open"].iloc[entry_idx])
        stop = entry * (1 - direction * stop_atr_multiple * atr_pct)
        target = entry * (1 + direction * take_profit_atr_multiple * atr_pct)
        exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
        outcome = 0

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
                outcome = 0
                break
            if hit_stop:
                outcome = 0
                break
            if hit_target:
                outcome = 1
                break
        labels.append(outcome)

    return pd.Series(labels, index=candidates.index, dtype="float")


def build_meta_dataset(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features = build_features(df)
    labels = label_candidates(
        df,
        candidates,
        max_hold_bars=max_hold_bars,
        stop_atr_multiple=stop_atr_multiple,
        take_profit_atr_multiple=take_profit_atr_multiple,
    )
    candidates = candidates.copy()
    candidates["target"] = labels
    candidates = candidates.dropna(subset=["target"])
    candidate_features = features.reindex(candidates["timestamp"]).reset_index(drop=True)
    side_feature = candidates["side"].map({"long": 1.0, "short": -1.0}).reset_index(drop=True)
    setup_features = pd.get_dummies(candidates["setup"], prefix="setup", dtype=float).reset_index(drop=True)
    x = pd.concat([candidate_features.reset_index(drop=True), side_feature.rename("side"), setup_features], axis=1)
    y = candidates["target"].astype(int).reset_index(drop=True)
    return x.replace([np.inf, -np.inf], np.nan), y, candidates.reset_index(drop=True)


def build_meta_features(df: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_features(df)
    candidates = candidates.copy().reset_index(drop=True)
    candidate_features = features.reindex(candidates["timestamp"]).reset_index(drop=True)
    side_feature = candidates["side"].map({"long": 1.0, "short": -1.0}).reset_index(drop=True)
    setup_features = pd.get_dummies(candidates["setup"], prefix="setup", dtype=float).reset_index(drop=True)
    x = pd.concat([candidate_features.reset_index(drop=True), side_feature.rename("side"), setup_features], axis=1)
    return x.replace([np.inf, -np.inf], np.nan), candidates


def make_meta_model(random_state: int) -> VotingClassifier:
    linear = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=random_state)),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=25,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    boosting = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, random_state=random_state)),
        ]
    )
    return VotingClassifier(
        estimators=[("linear", linear), ("forest", forest), ("boosting", boosting)],
        voting="soft",
        weights=[1, 2, 2],
    )


def simulate_candidates(
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


def simulate_with_setup_thresholds(
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
    return simulate_candidates(
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
