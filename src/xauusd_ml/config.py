from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train_csv: Path
    timestamp_column: str = "timestamp"
    timezone: str = "UTC"


class FeatureConfig(BaseModel):
    horizon_bars: int = Field(default=4, ge=1)
    threshold_bps: float = Field(default=8.0, ge=0.0)
    min_rows: int = Field(default=500, ge=100)
    label_mode: str = Field(default="return", pattern="^(return|barrier)$")


class ValidationConfig(BaseModel):
    splits: int = Field(default=5, ge=2)
    embargo_bars: int = Field(default=4, ge=0)


class ModelConfig(BaseModel):
    artifact_path: Path = Path("artifacts/xauusd_ensemble.joblib")
    random_state: int = 42
    min_confidence: float = Field(default=0.56, ge=0.0, le=1.0)


class BacktestConfig(BaseModel):
    initial_equity: float = Field(default=10_000.0, gt=0.0)
    risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.05)
    spread_bps: float = Field(default=3.0, ge=0.0)
    slippage_bps: float = Field(default=1.0, ge=0.0)
    max_position_notional: float = Field(default=50_000.0, gt=0.0)


class StrategyConfig(BaseModel):
    max_hold_bars: int = Field(default=8, ge=1)
    stop_atr_multiple: float = Field(default=1.4, gt=0.0)
    take_profit_atr_multiple: float = Field(default=2.2, gt=0.0)
    allow_overlapping_trades: bool = False


class AppConfig(BaseModel):
    data: DataConfig
    features: FeatureConfig = FeatureConfig()
    validation: ValidationConfig = ValidationConfig()
    model: ModelConfig = ModelConfig()
    backtest: BacktestConfig = BacktestConfig()
    strategy: StrategyConfig = StrategyConfig()


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig.model_validate(raw)
