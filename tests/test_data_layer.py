import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from apex_trader.data.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_vwap,
    calculate_indicators,
    IndicatorSnapshot,
)
from apex_trader.data.price_feed import PriceFeedManager, OHLCVSnapshot
from apex_trader.data.order_flow import OrderFlowAnalyzer, OrderFlowSnapshot
from apex_trader.data.news_feed import NewsFeedCollector


@pytest.fixture
def sample_ohlcv_df():
    """Generates a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range(end=datetime.utcnow(), periods=n, freq="5min")
    prices = 100.0 + np.cumsum(np.random.normal(0.1, 1.0, size=n))
    highs = prices + np.abs(np.random.normal(0.5, 0.5, size=n))
    lows = prices - np.abs(np.random.normal(0.5, 0.5, size=n))
    opens = (prices + np.roll(prices, 1)) / 2
    opens[0] = 100.0
    volumes = np.random.uniform(100, 1000, size=n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=dates)


def test_rsi_bounds(sample_ohlcv_df):
    rsi = calculate_rsi(sample_ohlcv_df["close"], 14)
    assert len(rsi) == len(sample_ohlcv_df)
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()


def test_bollinger_bands(sample_ohlcv_df):
    upper, middle, lower, pb = calculate_bollinger_bands(sample_ohlcv_df["close"], 20, 2.0)
    # Check valid tail values
    valid_mask = ~upper.isna()
    assert (upper[valid_mask] >= middle[valid_mask]).all()
    assert (middle[valid_mask] >= lower[valid_mask]).all()


def test_atr_positive(sample_ohlcv_df):
    atr = calculate_atr(sample_ohlcv_df, 14)
    assert len(atr) == len(sample_ohlcv_df)
    assert (atr > 0).all()


def test_calculate_indicators(sample_ohlcv_df):
    snapshot = calculate_indicators(sample_ohlcv_df)
    assert isinstance(snapshot, IndicatorSnapshot)
    assert snapshot.current_price > 0
    assert 0 <= snapshot.rsi_14 <= 100
    assert snapshot.bb_upper >= snapshot.bb_lower
    assert snapshot.trend_regime in ["STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"]
    assert snapshot.volatility_regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]
    d = snapshot.to_dict()
    assert "current_price" in d
    assert "rsi_14" in d


def test_price_feed():
    feed = PriceFeedManager(exchange_id="binance")
    snapshot = feed.fetch_ohlcv("BTC/USDT", asset_type="crypto", timeframe="5m", limit=30)
    assert isinstance(snapshot, OHLCVSnapshot)
    assert snapshot.symbol == "BTC/USDT"
    assert len(snapshot.df) == 30
    assert snapshot.current_price > 0


def test_order_flow():
    analyzer = OrderFlowAnalyzer(exchange_id="binance")
    flow = analyzer.fetch_order_flow("BTC/USDT", limit=10)
    assert isinstance(flow, OrderFlowSnapshot)
    assert flow.best_ask >= flow.best_bid
    assert flow.spread >= 0
    assert -1.0 <= flow.imbalance_ratio <= 1.0


def test_news_feed():
    collector = NewsFeedCollector()
    news = collector.fetch_news(category="crypto", limit=3)
    assert len(news) > 0
    for item in news:
        assert item.title
        assert -1.0 <= item.sentiment_hint <= 1.0
