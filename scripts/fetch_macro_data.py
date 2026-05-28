from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xauusd_ml.data import load_ohlcv_csv  # noqa: E402


YAHOO_SERIES = {
    "macro_vix": "^VIX",
    "macro_us_10y_proxy": "^TNX",
    "macro_dollar_index": "DX-Y.NYB",
    "macro_spx": "^GSPC",
}


def _fetch_yahoo(symbol: str) -> pd.Series:
    period2 = int(time.time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1=0&period2={period2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    close = result["indicators"]["quote"][0]["close"]
    index = pd.to_datetime(timestamps, unit="s", utc=True).normalize()
    return pd.Series(close, index=index, dtype="float64").dropna()


def build_macro_frame() -> pd.DataFrame:
    series = {}
    for column, symbol in YAHOO_SERIES.items():
        try:
            series[column] = _fetch_yahoo(symbol)
        except Exception as exc:
            print(f"Failed to fetch Yahoo series {symbol}: {exc}")
    macro = pd.DataFrame(series).sort_index().ffill()
    if "macro_vix" in macro.columns:
        macro["macro_vix_risk_on"] = -macro["macro_vix"]
    if "macro_spx" in macro.columns:
        macro["macro_spx_return_5d"] = macro["macro_spx"].pct_change(5)
    if "macro_dollar_index" in macro.columns:
        macro["macro_dollar_return_5d"] = macro["macro_dollar_index"].pct_change(5)
    return macro


def merge_macro(input_path: Path, output_path: Path, macro_output: Path) -> None:
    prices = load_ohlcv_csv(input_path, "timestamp")
    macro = build_macro_frame()
    macro_output.parent.mkdir(parents=True, exist_ok=True)
    macro.reset_index(names="timestamp").to_csv(macro_output, index=False)

    price_frame = prices.reset_index().sort_values("timestamp")
    macro_frame = macro.reset_index(names="timestamp").sort_values("timestamp")
    price_frame["timestamp"] = pd.to_datetime(price_frame["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    macro_frame["timestamp"] = pd.to_datetime(macro_frame["timestamp"], utc=True).astype("datetime64[ns, UTC]")

    merged = pd.merge_asof(
        price_frame,
        macro_frame,
        on="timestamp",
        direction="backward",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(output_path, len(merged), merged["timestamp"].min(), merged["timestamp"].max())
    print(macro_output, len(macro), macro.index.min(), macro.index.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--macro-output", default=Path("data/macro_yahoo_daily.csv"), type=Path)
    args = parser.parse_args()
    merge_macro(args.input, args.output, args.macro_output)


if __name__ == "__main__":
    main()
