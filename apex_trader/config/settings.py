import os
from pathlib import Path
from typing import List, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv()


class SystemConfig(BaseModel):
    mode: str = "paper"
    poll_interval_seconds: int = 60
    log_level: str = "INFO"
    db_path: str = "storage/apex_trader.db"


class AssetConfig(BaseModel):
    symbol: str
    asset_type: str = "crypto"  # "crypto" or "stock"
    exchange: str = "binance"


class TradingConfig(BaseModel):
    universe: List[AssetConfig] = Field(default_factory=list)
    timeframe: str = "5m"
    lookback_candles: int = 100


class RiskConfig(BaseModel):
    max_portfolio_risk_pct: float = 1.5
    max_open_positions: int = 3
    daily_max_drawdown_pct: float = 3.0
    risk_reward_ratio_min: float = 1.5
    max_slippage_tolerance_pct: float = 0.2
    atr_multiplier_stop: float = 1.5


class AgentsConfig(BaseModel):
    llm_provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.2
    debate_rounds: int = 1
    confidence_threshold: float = 0.65


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    # API Keys from environment
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    finnhub_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("FINNHUB_API_KEY"))
    alpha_vantage_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY"))
    binance_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("BINANCE_API_KEY"))
    binance_api_secret: Optional[str] = Field(default_factory=lambda: os.getenv("BINANCE_API_SECRET"))
    alpaca_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ALPACA_API_KEY"))
    alpaca_api_secret: Optional[str] = Field(default_factory=lambda: os.getenv("ALPACA_API_SECRET"))
    alpaca_base_url: str = Field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file and merge with environment variables."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return AppConfig(**data)
    return AppConfig()
