import pandas as pd

from xauusd_ml.features import build_dataset


def test_build_dataset_drops_forward_leakage_rows():
    idx = pd.date_range("2024-01-01", periods=220, freq="15min", tz="UTC")
    close = pd.Series(range(2000, 2220), index=idx, dtype=float)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100,
        },
        index=idx,
    )

    x, y = build_dataset(df, horizon_bars=4, threshold_bps=1, min_rows=100)

    assert len(x) == len(y)
    assert x.index.max() <= idx[-5]
    assert set(y.unique()).issubset({"short", "flat", "long"})

