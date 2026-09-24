from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class IndicatorSnapshot:
    """Snapshot of technical and quantitative indicator values for the latest candle."""
    current_price: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_percent_b: float
    atr_14: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: float
    vwap: float
    support_level: float
    resistance_level: float
    trend_regime: str  # "STRONGLY_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONGLY_BEARISH"
    volatility_regime: str  # "LOW", "NORMAL", "HIGH", "EXTREME"
    volume_ratio: float  # Latest volume / 20-period avg volume

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_price": round(self.current_price, 4),
            "rsi_14": round(self.rsi_14, 2),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_hist": round(self.macd_hist, 4),
            "bb_upper": round(self.bb_upper, 4),
            "bb_middle": round(self.bb_middle, 4),
            "bb_lower": round(self.bb_lower, 4),
            "bb_percent_b": round(self.bb_percent_b, 3),
            "atr_14": round(self.atr_14, 4),
            "ema_9": round(self.ema_9, 4),
            "ema_21": round(self.ema_21, 4),
            "ema_50": round(self.ema_50, 4),
            "ema_200": round(self.ema_200, 4),
            "vwap": round(self.vwap, 4),
            "support_level": round(self.support_level, 4),
            "resistance_level": round(self.resistance_level, 4),
            "trend_regime": self.trend_regime,
            "volatility_regime": self.volatility_regime,
            "volume_ratio": round(self.volume_ratio, 2),
        }


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates MACD, Signal line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculates Bollinger Bands (Upper, Middle, Lower) and %B."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # %B = (Price - Lower) / (Upper - Lower)
    band_width = upper - lower
    percent_b = (series - lower) / (band_width.replace(0, np.nan))
    percent_b = percent_b.fillna(0.5)
    return upper, middle, lower, percent_b


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.fillna(tr.mean() if not tr.empty else 0.0)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculates Volume Weighted Average Price (VWAP)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol_price = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_vol_price / (cum_vol + 1e-10)
    return vwap.fillna(typical_price)


def find_support_resistance(df: pd.DataFrame, window: int = 20):
    """Finds recent local support (swing low) and resistance (swing high) levels."""
    if len(df) < window:
        return float(df["low"].min()), float(df["high"].max())
    
    recent_slice = df.tail(window)
    support = float(recent_slice["low"].min())
    resistance = float(recent_slice["high"].max())
    return support, resistance


def calculate_indicators(df: pd.DataFrame) -> IndicatorSnapshot:
    """Computes all technical indicators for the latest candle from OHLCV dataframe."""
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required OHLCV columns: {required_cols - set(df.columns)}")

    close = df["close"]
    
    # 1. RSI
    rsi_series = calculate_rsi(close, 14)
    latest_rsi = float(rsi_series.iloc[-1])
    
    # 2. MACD
    macd, macd_sig, macd_h = calculate_macd(close)
    latest_macd = float(macd.iloc[-1])
    latest_macd_sig = float(macd_sig.iloc[-1])
    latest_macd_h = float(macd_h.iloc[-1])
    
    # 3. Bollinger Bands
    bb_u, bb_m, bb_l, bb_pb = calculate_bollinger_bands(close, 20, 2.0)
    latest_bb_u = float(bb_u.iloc[-1]) if not pd.isna(bb_u.iloc[-1]) else float(close.iloc[-1]) * 1.02
    latest_bb_m = float(bb_m.iloc[-1]) if not pd.isna(bb_m.iloc[-1]) else float(close.iloc[-1])
    latest_bb_l = float(bb_l.iloc[-1]) if not pd.isna(bb_l.iloc[-1]) else float(close.iloc[-1]) * 0.98
    latest_bb_pb = float(bb_pb.iloc[-1])
    
    # 4. ATR
    atr_series = calculate_atr(df, 14)
    latest_atr = float(atr_series.iloc[-1])
    
    # 5. EMAs
    ema_9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema_21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema_200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(df) >= 50 else ema_50
    
    # 6. VWAP
    vwap_series = calculate_vwap(df)
    latest_vwap = float(vwap_series.iloc[-1])
    
    # 7. Support & Resistance
    support, resistance = find_support_resistance(df, 20)
    
    # 8. Volume Ratio
    vol_mean_20 = float(df["volume"].rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(df["volume"].mean())
    latest_vol = float(df["volume"].iloc[-1])
    vol_ratio = (latest_vol / (vol_mean_20 + 1e-10)) if vol_mean_20 > 0 else 1.0
    
    # 9. Regime Classification
    current_p = float(close.iloc[-1])
    
    # Trend Regime
    if current_p > ema_9 > ema_21 > ema_50:
        trend_regime = "STRONGLY_BULLISH"
    elif current_p > ema_21 and latest_macd > 0:
        trend_regime = "BULLISH"
    elif current_p < ema_9 < ema_21 < ema_50:
        trend_regime = "STRONGLY_BEARISH"
    elif current_p < ema_21 and latest_macd < 0:
        trend_regime = "BEARISH"
    else:
        trend_regime = "NEUTRAL"
        
    # Volatility Regime based on ATR relative to price
    atr_pct = (latest_atr / (current_p + 1e-10)) * 100.0
    if atr_pct < 0.5:
        vol_regime = "LOW"
    elif atr_pct < 1.5:
        vol_regime = "NORMAL"
    elif atr_pct < 3.0:
        vol_regime = "HIGH"
    else:
        vol_regime = "EXTREME"
        
    return IndicatorSnapshot(
        current_price=current_p,
        rsi_14=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_macd_sig,
        macd_hist=latest_macd_h,
        bb_upper=latest_bb_u,
        bb_middle=latest_bb_m,
        bb_lower=latest_bb_l,
        bb_percent_b=latest_bb_pb,
        atr_14=latest_atr,
        ema_9=ema_9,
        ema_21=ema_21,
        ema_50=ema_50,
        ema_200=ema_200,
        vwap=latest_vwap,
        support_level=support,
        resistance_level=resistance,
        trend_regime=trend_regime,
        volatility_regime=vol_regime,
        volume_ratio=vol_ratio,
    )
