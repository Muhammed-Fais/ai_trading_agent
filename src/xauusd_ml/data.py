from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv_csv(path: str | Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
        rename_map = {
            "Date": timestamp_column,
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)
    else:
        df = df.rename(columns={c: c.lower() for c in df.columns})

    if timestamp_column not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": timestamp_column})
    if "volume" not in df.columns:
        df["volume"] = 0

    missing = {timestamp_column, *REQUIRED_COLUMNS}.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    df = df.sort_values(timestamp_column).drop_duplicates(timestamp_column)
    df = df.set_index(timestamp_column)

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    return df
