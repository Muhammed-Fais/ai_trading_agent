from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.data import load_ohlcv_csv  # noqa: E402


def prepare(input_path: Path, output_path: Path, timeframe: str) -> None:
    df = load_ohlcv_csv(input_path, "timestamp")
    resampled = (
        df.resample(timeframe)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.reset_index().to_csv(output_path, index=False)
    print(output_path, len(resampled), resampled.index.min(), resampled.index.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    prepare(args.input, args.output, args.timeframe)


if __name__ == "__main__":
    main()
