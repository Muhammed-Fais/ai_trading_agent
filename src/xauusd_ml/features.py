from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)
    return rsi.fillna(50)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    features = pd.DataFrame(index=df.index)

    for window in (1, 2, 4, 8, 16, 32, 96):
        features[f"ret_{window}"] = close.pct_change(window)

    for window in (8, 21, 55, 144, 288):
        ema = close.ewm(span=window, adjust=False).mean()
        features[f"close_to_ema_{window}"] = close / ema - 1
        features[f"ema_slope_{window}"] = ema.pct_change(8)

    atr_14 = _atr(df, 14)
    features["atr_pct_14"] = atr_14 / close
    features["atr_regime_96"] = features["atr_pct_14"] / features["atr_pct_14"].rolling(96).median() - 1
    features["range_pct"] = (df["high"] - df["low"]) / close
    features["body_pct"] = (df["close"] - df["open"]) / close
    features["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / close
    features["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / close
    features["rsi_14"] = _rsi(close, 14) / 100
    features["rsi_56"] = _rsi(close, 56) / 100

    rolling_mean = close.rolling(32).mean()
    rolling_std = close.rolling(32).std()
    features["zscore_32"] = (close - rolling_mean) / rolling_std
    features["donchian_pos_96"] = (
        (close - df["low"].rolling(96).min())
        / (df["high"].rolling(96).max() - df["low"].rolling(96).min())
    )
    features["donchian_break_96"] = close / df["high"].rolling(96).max().shift(1) - 1
    features["realized_vol_24"] = close.pct_change().rolling(24).std()
    features["realized_vol_96"] = close.pct_change().rolling(96).std()

    volume = df["volume"].replace(0, np.nan)
    features["volume_z_96"] = (
        (volume - volume.rolling(96).mean()) / volume.rolling(96).std()
    ).fillna(0)
    hour_angle = 2 * np.pi * df.index.hour / 24
    dow_angle = 2 * np.pi * df.index.dayofweek / 7
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["dow_sin"] = np.sin(dow_angle)
    features["dow_cos"] = np.cos(dow_angle)
    features["is_london_ny_overlap"] = df.index.hour.isin([13, 14, 15, 16]).astype(float)
    features["is_asia_session"] = df.index.hour.isin([0, 1, 2, 3, 4, 5, 6]).astype(float)

    return features.replace([np.inf, -np.inf], np.nan)


def build_labels(close: pd.Series, horizon_bars: int, threshold_bps: float) -> pd.Series:
    forward_return = close.shift(-horizon_bars) / close - 1
    threshold = threshold_bps / 10_000
    labels = pd.Series("flat", index=close.index, dtype="object")
    labels[forward_return > threshold] = "long"
    labels[forward_return < -threshold] = "short"
    return labels


def _barrier_outcome(
    df: pd.DataFrame,
    start_idx: int,
    direction: int,
    stop_pct: float,
    target_pct: float,
    max_hold_bars: int,
) -> str:
    entry_idx = start_idx + 1
    if entry_idx >= len(df):
        return "none"

    entry = float(df["open"].iloc[entry_idx])
    stop_price = entry * (1 - direction * stop_pct)
    target_price = entry * (1 + direction * target_pct)
    exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)

    for candidate_idx in range(entry_idx, exit_idx + 1):
        high = float(df["high"].iloc[candidate_idx])
        low = float(df["low"].iloc[candidate_idx])
        if direction == 1:
            hit_stop = low <= stop_price
            hit_target = high >= target_price
        else:
            hit_stop = high >= stop_price
            hit_target = low <= target_price

        if hit_stop and hit_target:
            return "loss"
        if hit_stop:
            return "loss"
        if hit_target:
            return "win"

    return "none"


def build_barrier_labels(
    df: pd.DataFrame,
    max_hold_bars: int,
    stop_atr_multiple: float,
    take_profit_atr_multiple: float,
) -> pd.Series:
    features = build_features(df)
    atr_pct = features["atr_pct_14"]
    labels = pd.Series("flat", index=df.index, dtype="object")

    for idx in range(len(df) - max_hold_bars - 1):
        current_atr_pct = float(atr_pct.iloc[idx])
        if not np.isfinite(current_atr_pct) or current_atr_pct <= 0:
            continue

        stop_pct = stop_atr_multiple * current_atr_pct
        target_pct = take_profit_atr_multiple * current_atr_pct
        long_outcome = _barrier_outcome(df, idx, 1, stop_pct, target_pct, max_hold_bars)
        short_outcome = _barrier_outcome(df, idx, -1, stop_pct, target_pct, max_hold_bars)

        if long_outcome == "win" and short_outcome != "win":
            labels.iloc[idx] = "long"
        elif short_outcome == "win" and long_outcome != "win":
            labels.iloc[idx] = "short"

    return labels


def build_dataset(
    df: pd.DataFrame,
    horizon_bars: int,
    threshold_bps: float,
    min_rows: int,
    label_mode: str = "return",
    max_hold_bars: int | None = None,
    stop_atr_multiple: float | None = None,
    take_profit_atr_multiple: float | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    x = build_features(df)
    if label_mode == "barrier":
        if max_hold_bars is None or stop_atr_multiple is None or take_profit_atr_multiple is None:
            raise ValueError("Barrier labels require max hold, stop ATR, and take-profit ATR settings.")
        y = build_barrier_labels(
            df,
            max_hold_bars=max_hold_bars,
            stop_atr_multiple=stop_atr_multiple,
            take_profit_atr_multiple=take_profit_atr_multiple,
        )
        trim_bars = max_hold_bars + 1
    else:
        y = build_labels(df["close"], horizon_bars, threshold_bps)
        trim_bars = horizon_bars
    dataset = x.join(y.rename("target")).dropna()
    dataset = dataset.iloc[:-trim_bars]

    if len(dataset) < min_rows:
        raise ValueError(f"Need at least {min_rows} usable rows, found {len(dataset)}.")

    return dataset.drop(columns=["target"]), dataset["target"]
