"""
Market Regime Feature Generation.

Classifies market conditions into different regimes:
- Trending vs Ranging
- High vs Low Volatility
- Momentum vs Mean-Reversion
- Bull vs Bear phases
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


def classify_market_regime(
    df: pd.DataFrame,
    lookback: int = 100,
    trend_threshold: float = 25.0,
    volatility_percentile: float = 0.7
) -> pd.DataFrame:
    """
    Classify the market into different regimes.
    
    Uses multiple indicators to determine:
    1. Trend regime (trending up, trending down, or ranging)
    2. Volatility regime (high, normal, or low)
    3. Combined regime label
    
    Args:
        df: OHLCV DataFrame with features (requires ADX, volatility)
        lookback: Lookback period for regime calculation
        trend_threshold: ADX threshold for trending market
        volatility_percentile: Percentile for high volatility
        
    Returns:
        DataFrame with regime classifications
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. Calculate trend direction and strength
    # Using price position relative to SMA
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    
    price_above_sma50 = close > sma_50
    price_above_sma200 = close > sma_200
    sma50_above_sma200 = sma_50 > sma_200
    
    # Calculate ADX for trend strength
    adx = _calculate_adx(df, 14)
    is_trending = adx > trend_threshold
    
    # 2. Calculate volatility regime
    returns = close.pct_change()
    volatility = returns.rolling(lookback).std()
    vol_median = volatility.rolling(lookback * 2).median()
    
    high_vol_threshold = volatility.rolling(lookback * 2).quantile(volatility_percentile)
    low_vol_threshold = volatility.rolling(lookback * 2).quantile(1 - volatility_percentile)
    
    is_high_vol = volatility > high_vol_threshold
    is_low_vol = volatility < low_vol_threshold
    
    # 3. Calculate range/consolidation
    # Price range relative to average
    price_range = high.rolling(lookback).max() - low.rolling(lookback).min()
    avg_range = price_range.rolling(lookback).mean()
    is_consolidating = price_range < avg_range * 0.7
    
    # 4. Classify trend regime
    trend_regime = np.where(
        is_trending & price_above_sma50 & price_above_sma200,
        'trending_up',
        np.where(
            is_trending & ~price_above_sma50 & ~price_above_sma200,
            'trending_down',
            'ranging'
        )
    )
    
    # 5. Classify volatility regime
    vol_regime = np.where(
        is_high_vol, 'high_vol',
        np.where(is_low_vol, 'low_vol', 'normal_vol')
    )
    
    # 6. Combined regime
    combined_regime = np.char.add(
        np.char.add(trend_regime.astype(str), '_'),
        vol_regime.astype(str)
    )
    
    results = pd.DataFrame({
        # Trend regime
        'regime_trend': trend_regime,
        'regime_is_trending': is_trending.astype(int),
        'regime_trend_direction': np.where(
            price_above_sma50, 1, -1
        ),
        'regime_trend_strength': adx,
        
        # Volatility regime
        'regime_volatility': vol_regime,
        'regime_is_high_vol': is_high_vol.astype(int),
        'regime_is_low_vol': is_low_vol.astype(int),
        'regime_vol_percentile': volatility.rolling(lookback * 2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ),
        
        # Consolidation
        'regime_is_consolidating': is_consolidating.astype(int),
        
        # Combined
        'regime_combined': combined_regime,
        
        # Momentum regime
        'regime_momentum': np.where(
            returns.rolling(lookback).sum() > 0, 'positive', 'negative'
        ),
        
        # SMA regime indicators
        'regime_above_sma50': price_above_sma50.astype(int),
        'regime_above_sma200': price_above_sma200.astype(int),
        'regime_golden_cross': sma50_above_sma200.astype(int)
    }, index=df.index)
    
    return results


def detect_volatility_regime(
    df: pd.DataFrame,
    lookback: int = 20,
    long_lookback: int = 100
) -> pd.DataFrame:
    """
    Detect volatility regime changes.
    
    Identifies:
    - Volatility expansion
    - Volatility contraction
    - Regime transitions
    
    Args:
        df: OHLCV DataFrame
        lookback: Short-term lookback
        long_lookback: Long-term lookback
        
    Returns:
        DataFrame with volatility regime features
    """
    close = df['close']
    
    # Calculate volatilities
    returns = close.pct_change()
    short_vol = returns.rolling(lookback).std()
    long_vol = returns.rolling(long_lookback).std()
    
    # Volatility ratio
    vol_ratio = short_vol / (long_vol + 1e-10)
    
    # Volatility expansion/contraction
    vol_expanding = vol_ratio > 1.2
    vol_contracting = vol_ratio < 0.8
    
    # Volatility change rate
    vol_change = short_vol.pct_change(5)
    
    # ATR-based volatility
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - close.shift(1)),
        abs(df['low'] - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr_short = tr.rolling(lookback).mean()
    atr_long = tr.rolling(long_lookback).mean()
    
    # Regime change detection
    vol_percentile = short_vol.rolling(long_lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    regime_change = (
        (vol_percentile > 0.8) & (vol_percentile.shift(1) <= 0.8) |
        (vol_percentile < 0.2) & (vol_percentile.shift(1) >= 0.2)
    )
    
    results = pd.DataFrame({
        'vol_short': short_vol,
        'vol_long': long_vol,
        'vol_ratio': vol_ratio,
        'vol_expanding': vol_expanding.astype(int),
        'vol_contracting': vol_contracting.astype(int),
        'vol_change_rate': vol_change,
        'vol_regime_change': regime_change.astype(int),
        'vol_percentile': vol_percentile,
        'atr_ratio': atr_short / (atr_long + 1e-10)
    }, index=df.index)
    
    return results


def detect_trend_regime(
    df: pd.DataFrame,
    short_period: int = 20,
    long_period: int = 50
) -> pd.DataFrame:
    """
    Detect trend regime and strength.
    
    Uses multiple methods:
    - Moving average alignment
    - Linear regression slope
    - Higher highs / lower lows pattern
    
    Args:
        df: OHLCV DataFrame
        short_period: Short-term period
        long_period: Long-term period
        
    Returns:
        DataFrame with trend regime features
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Moving average alignment
    ema_short = close.ewm(span=short_period, adjust=False).mean()
    ema_long = close.ewm(span=long_period, adjust=False).mean()
    
    # Price relative to EMAs
    price_above_short = close > ema_short
    price_above_long = close > ema_long
    ema_aligned_up = ema_short > ema_long
    
    # Linear regression slope
    def rolling_slope(series, window):
        """Calculate rolling linear regression slope."""
        from scipy import stats
        
        slopes = np.full(len(series), np.nan)
        x = np.arange(window)
        
        for i in range(window, len(series)):
            y = series.iloc[i-window:i].values
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes[i] = slope
        
        return pd.Series(slopes, index=series.index)
    
    trend_slope = rolling_slope(close, long_period)
    slope_normalized = trend_slope / close * 100
    
    # Higher highs / lower lows pattern
    swing_high = high.rolling(5).max()
    swing_low = low.rolling(5).min()
    
    hh = swing_high > swing_high.shift(short_period)  # Higher high
    ll = swing_low < swing_low.shift(short_period)    # Lower low
    hl = swing_low > swing_low.shift(short_period)    # Higher low
    lh = swing_high < swing_high.shift(short_period)  # Lower high
    
    # Trend patterns
    uptrend_pattern = hh & hl  # Higher highs and higher lows
    downtrend_pattern = ll & lh  # Lower lows and lower highs
    
    results = pd.DataFrame({
        'trend_ema_short': ema_short,
        'trend_ema_long': ema_long,
        'trend_aligned_up': (ema_aligned_up & price_above_short).astype(int),
        'trend_aligned_down': (~ema_aligned_up & ~price_above_short).astype(int),
        'trend_slope': trend_slope,
        'trend_slope_normalized': slope_normalized,
        'trend_uptrend_pattern': uptrend_pattern.astype(int),
        'trend_downtrend_pattern': downtrend_pattern.astype(int),
        'trend_higher_high': hh.astype(int),
        'trend_lower_low': ll.astype(int),
        'trend_higher_low': hl.astype(int),
        'trend_lower_high': lh.astype(int)
    }, index=df.index)
    
    # Overall trend score (-1 to 1)
    trend_score = (
        results['trend_aligned_up'] * 0.3 +
        (slope_normalized > 0).astype(int) * 0.3 +
        results['trend_uptrend_pattern'] * 0.2 +
        results['trend_higher_high'] * 0.1 +
        results['trend_higher_low'] * 0.1 -
        results['trend_aligned_down'] * 0.3 -
        (slope_normalized < 0).astype(int) * 0.3 -
        results['trend_downtrend_pattern'] * 0.2 -
        results['trend_lower_low'] * 0.1 -
        results['trend_lower_high'] * 0.1
    )
    results['trend_score'] = trend_score
    
    return results


def detect_momentum_vs_mean_reversion(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Detect whether the market is in momentum or mean-reversion mode.
    
    Momentum: Trends continue
    Mean-reversion: Trends reverse
    
    Args:
        df: OHLCV DataFrame
        lookback: Analysis period
        
    Returns:
        DataFrame with momentum/mean-reversion indicators
    """
    close = df['close']
    returns = close.pct_change()
    
    # Autocorrelation of returns (positive = momentum, negative = mean-reversion)
    autocorr = returns.rolling(lookback).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    
    # Hurst exponent approximation (simplified)
    # H > 0.5: Trending (momentum)
    # H < 0.5: Mean-reverting
    # H = 0.5: Random walk
    def estimate_hurst(series):
        """Simplified Hurst exponent estimation."""
        n = len(series)
        if n < 20:
            return 0.5
        
        # Use R/S analysis
        mean = series.mean()
        deviations = (series - mean).cumsum()
        r = deviations.max() - deviations.min()
        s = series.std()
        
        if s == 0:
            return 0.5
        
        rs = r / s
        h = np.log(rs) / np.log(n) if rs > 0 and n > 1 else 0.5
        
        return np.clip(h, 0, 1)
    
    hurst = close.rolling(lookback).apply(estimate_hurst, raw=True)
    
    # Mean reversion indicators
    zscore = (close - close.rolling(lookback).mean()) / (close.rolling(lookback).std() + 1e-10)
    
    # Momentum continuation probability
    up_after_up = (returns > 0) & (returns.shift(1) > 0)
    down_after_down = (returns < 0) & (returns.shift(1) < 0)
    continuation_rate = (up_after_up | down_after_down).rolling(lookback).mean()
    
    # Reversal rate
    up_after_down = (returns > 0) & (returns.shift(1) < 0)
    down_after_up = (returns < 0) & (returns.shift(1) > 0)
    reversal_rate = (up_after_down | down_after_up).rolling(lookback).mean()
    
    results = pd.DataFrame({
        'mr_autocorr': autocorr,
        'mr_hurst': hurst,
        'mr_zscore': zscore,
        'mr_is_momentum': (autocorr > 0.1).astype(int),
        'mr_is_mean_revert': (autocorr < -0.1).astype(int),
        'mr_continuation_rate': continuation_rate,
        'mr_reversal_rate': reversal_rate,
        'mr_regime': np.where(
            hurst > 0.55, 'momentum',
            np.where(hurst < 0.45, 'mean_reversion', 'random')
        )
    }, index=df.index)
    
    return results


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX (helper function)."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=df.index
    )
    
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


class RegimeFeatureGenerator:
    """
    Generates all market regime features.
    """
    
    def __init__(
        self,
        regime_lookback: int = 100,
        volatility_lookback: int = 20,
        trend_short_period: int = 20,
        trend_long_period: int = 50,
        trend_threshold: float = 25.0,
        volatility_percentile: float = 0.7
    ):
        """
        Initialize the generator.
        """
        self.regime_lookback = regime_lookback
        self.volatility_lookback = volatility_lookback
        self.trend_short_period = trend_short_period
        self.trend_long_period = trend_long_period
        self.trend_threshold = trend_threshold
        self.volatility_percentile = volatility_percentile
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all regime features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with regime features added
        """
        features = df.copy()
        
        # Market regime classification
        regime = classify_market_regime(
            df,
            self.regime_lookback,
            self.trend_threshold,
            self.volatility_percentile
        )
        features = pd.concat([features, regime], axis=1)
        
        # Volatility regime
        vol_regime = detect_volatility_regime(
            df,
            self.volatility_lookback,
            self.regime_lookback
        )
        features = pd.concat([features, vol_regime], axis=1)
        
        # Trend regime
        trend_regime = detect_trend_regime(
            df,
            self.trend_short_period,
            self.trend_long_period
        )
        features = pd.concat([features, trend_regime], axis=1)
        
        # Momentum vs mean-reversion
        mr_regime = detect_momentum_vs_mean_reversion(
            df,
            self.volatility_lookback
        )
        features = pd.concat([features, mr_regime], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} regime features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        return [
            # Market regime
            'regime_trend', 'regime_is_trending', 'regime_trend_direction',
            'regime_trend_strength', 'regime_volatility', 'regime_is_high_vol',
            'regime_is_low_vol', 'regime_vol_percentile', 'regime_is_consolidating',
            'regime_combined', 'regime_momentum', 'regime_above_sma50',
            'regime_above_sma200', 'regime_golden_cross',
            
            # Volatility regime
            'vol_short', 'vol_long', 'vol_ratio', 'vol_expanding',
            'vol_contracting', 'vol_change_rate', 'vol_regime_change',
            'vol_percentile', 'atr_ratio',
            
            # Trend regime
            'trend_ema_short', 'trend_ema_long', 'trend_aligned_up',
            'trend_aligned_down', 'trend_slope', 'trend_slope_normalized',
            'trend_uptrend_pattern', 'trend_downtrend_pattern',
            'trend_higher_high', 'trend_lower_low', 'trend_higher_low',
            'trend_lower_high', 'trend_score',
            
            # Momentum/Mean-reversion
            'mr_autocorr', 'mr_hurst', 'mr_zscore', 'mr_is_momentum',
            'mr_is_mean_revert', 'mr_continuation_rate', 'mr_reversal_rate',
            'mr_regime'
        ]
