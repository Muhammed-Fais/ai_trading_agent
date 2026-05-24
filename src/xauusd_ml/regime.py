from __future__ import annotations

import pandas as pd

from xauusd_ml.features import build_features


REGIME_MODES = (
    "none",
    "trend_vol",
    "breakout_vol",
    "calm_trend",
)

CONFLUENCE_MODES = (
    "none",
    "h4_trend",
    "h4_breakout",
    "h4_trend_or_breakout",
)


def build_regime_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode not in REGIME_MODES:
        raise ValueError(f"Unknown regime mode: {mode}")
    if mode == "none":
        return pd.Series(True, index=df.index)

    features = build_features(df)
    close = df["close"]
    ema_55 = close.ewm(span=55, adjust=False).mean()
    ema_288 = close.ewm(span=288, adjust=False).mean()
    trend_strength = (ema_55 / ema_288 - 1).abs()
    trend_ok = trend_strength > trend_strength.rolling(500).median()

    atr = features["atr_pct_14"]
    atr_median = atr.rolling(500).median()
    atr_q75 = atr.rolling(500).quantile(0.75)
    atr_q90 = atr.rolling(500).quantile(0.90)
    vol_normal = (atr > atr_median) & (atr < atr_q90)
    vol_breakout = (atr > atr_q75) & (atr < atr_q90)
    vol_calm = atr < atr_q75

    donchian_position = features["donchian_pos_96"]
    near_extreme = (donchian_position > 0.75) | (donchian_position < 0.25)

    london_ny = features["is_london_ny_overlap"].astype(bool)
    asia = features["is_asia_session"].astype(bool)

    if mode == "trend_vol":
        mask = trend_ok & vol_normal & ~asia
    elif mode == "breakout_vol":
        mask = near_extreme & vol_breakout & (london_ny | ~asia)
    else:
        mask = trend_ok & vol_calm & ~asia

    return mask.reindex(df.index).fillna(False)


def build_confluence_direction(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode not in CONFLUENCE_MODES:
        raise ValueError(f"Unknown confluence mode: {mode}")
    if mode == "none":
        return pd.Series(0, index=df.index, dtype="int64")

    h4 = (
        df.resample("4h")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    close = h4["close"]
    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=89, adjust=False).mean()
    trend_long = (ema_fast > ema_slow) & (ema_fast.pct_change(3) > 0)
    trend_short = (ema_fast < ema_slow) & (ema_fast.pct_change(3) < 0)

    donchian_high = h4["high"].rolling(55).max().shift(1)
    donchian_low = h4["low"].rolling(55).min().shift(1)
    breakout_long = close > donchian_high
    breakout_short = close < donchian_low

    direction = pd.Series(0, index=h4.index, dtype="int64")
    if mode == "h4_trend":
        direction[trend_long] = 1
        direction[trend_short] = -1
    elif mode == "h4_breakout":
        direction[breakout_long] = 1
        direction[breakout_short] = -1
    else:
        direction[trend_long | breakout_long] = 1
        direction[trend_short | breakout_short] = -1

    return direction.reindex(df.index, method="ffill").fillna(0).astype("int64")


def signal_allowed(signal: str, confluence_value: int) -> bool:
    if confluence_value == 0:
        return True
    return (signal == "long" and confluence_value == 1) or (
        signal == "short" and confluence_value == -1
    )
