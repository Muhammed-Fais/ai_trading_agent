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


def generate_candidate_trades(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_144 = close.ewm(span=144, adjust=False).mean()
    donchian_high = df["high"].rolling(96).max().shift(1)
    donchian_low = df["low"].rolling(96).min().shift(1)
    features = build_features(df)
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

    rows = []
    for setup, sides in setups.items():
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
