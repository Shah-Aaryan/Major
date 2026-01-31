"""
Trend Indicator Generation.

Calculates trend-following indicators including:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- EMA Slope
- Average Directional Index (ADX)
- Trend strength metrics
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sma(
    prices: pd.Series,
    windows: List[int] = [5, 10, 20, 50, 100, 200]
) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages for multiple windows.
    
    Args:
        prices: Price series (typically close)
        windows: List of window sizes
        
    Returns:
        DataFrame with SMAs and derived features
    """
    results = {}
    
    for window in windows:
        sma = prices.rolling(window=window).mean()
        results[f'sma_{window}'] = sma
        
        # Price distance from SMA (percentage)
        results[f'price_sma_{window}_pct'] = (prices - sma) / sma
        
        # SMA slope (rate of change)
        results[f'sma_{window}_slope'] = sma.diff(5) / sma.shift(5)
    
    # Add SMA crossover features for common pairs
    if 20 in windows and 50 in windows:
        results['sma_20_50_cross'] = (
            results['sma_20'] > results['sma_50']
        ).astype(int)
    
    if 50 in windows and 200 in windows:
        results['sma_50_200_cross'] = (
            results['sma_50'] > results['sma_200']
        ).astype(int)
    
    return pd.DataFrame(results)


def calculate_ema(
    prices: pd.Series,
    windows: List[int] = [5, 10, 20, 50, 100]
) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages for multiple windows.
    
    EMA gives more weight to recent prices, making it more
    responsive to new information than SMA.
    
    Args:
        prices: Price series
        windows: List of window sizes (span parameter)
        
    Returns:
        DataFrame with EMAs and derived features
    """
    results = {}
    
    for window in windows:
        ema = prices.ewm(span=window, adjust=False).mean()
        results[f'ema_{window}'] = ema
        
        # Price distance from EMA (percentage)
        results[f'price_ema_{window}_pct'] = (prices - ema) / ema
        
        # EMA acceleration (second derivative)
        ema_diff = ema.diff()
        results[f'ema_{window}_accel'] = ema_diff.diff()
    
    # Add EMA crossover features
    if 9 in windows and 21 in windows:
        results['ema_9_21_cross'] = (
            results['ema_9'] > results['ema_21']
        ).astype(int)
    
    if 12 in windows and 26 in windows:
        results['ema_12_26_cross'] = (
            results['ema_12'] > results['ema_26']
        ).astype(int)
    
    return pd.DataFrame(results)


def calculate_ema_slope(
    prices: pd.Series,
    ema_window: int = 20,
    slope_period: int = 5
) -> pd.DataFrame:
    """
    Calculate EMA and its slope for trend direction analysis.
    
    The slope of the EMA indicates:
    - Positive slope: Uptrend
    - Negative slope: Downtrend
    - Near-zero slope: Ranging/consolidation
    
    Args:
        prices: Price series
        ema_window: EMA window size
        slope_period: Period for slope calculation
        
    Returns:
        DataFrame with EMA slope features
    """
    results = {}
    
    ema = prices.ewm(span=ema_window, adjust=False).mean()
    results['ema'] = ema
    
    # Raw slope
    results['ema_slope'] = ema.diff(slope_period)
    
    # Normalized slope (percentage change)
    results['ema_slope_pct'] = ema.diff(slope_period) / ema.shift(slope_period)
    
    # Slope direction (-1, 0, 1)
    slope_threshold = 0.001  # 0.1% threshold for "flat"
    results['ema_slope_direction'] = np.where(
        results['ema_slope_pct'] > slope_threshold, 1,
        np.where(results['ema_slope_pct'] < -slope_threshold, -1, 0)
    )
    
    # Slope momentum (is slope accelerating or decelerating?)
    results['ema_slope_momentum'] = results['ema_slope'].diff()
    
    return pd.DataFrame(results)


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) for trend strength.
    
    ADX measures trend strength regardless of direction:
    - ADX < 20: Weak/no trend (ranging market)
    - ADX 20-40: Moderate trend
    - ADX 40-60: Strong trend
    - ADX > 60: Very strong trend
    
    Also calculates +DI and -DI for trend direction.
    
    Args:
        df: OHLCV DataFrame
        period: ADX period (typically 14)
        
    Returns:
        DataFrame with ADX, +DI, -DI
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (similar to EMA)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    results = pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'dx': dx
    }, index=df.index)
    
    # Trend strength classification
    results['trend_strength'] = pd.cut(
        results['adx'],
        bins=[0, 20, 40, 60, 100],
        labels=['weak', 'moderate', 'strong', 'very_strong']
    )
    
    # Directional indicator (bullish or bearish trend)
    results['trend_direction'] = np.where(
        results['plus_di'] > results['minus_di'], 1, -1
    )
    
    # DI crossover signal
    results['di_crossover'] = (
        (results['plus_di'] > results['minus_di']) &
        (results['plus_di'].shift(1) <= results['minus_di'].shift(1))
    ).astype(int) - (
        (results['plus_di'] < results['minus_di']) &
        (results['plus_di'].shift(1) >= results['minus_di'].shift(1))
    ).astype(int)
    
    return results


def calculate_trend_lines(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Calculate linear regression trend lines.
    
    Uses linear regression to fit a trend line and calculate:
    - Slope of trend
    - R-squared (goodness of fit)
    - Distance from trend line
    
    Args:
        df: OHLCV DataFrame
        lookback: Period for trend line calculation
        
    Returns:
        DataFrame with trend line features
    """
    from scipy import stats
    
    close = df['close'].values
    n = len(close)
    
    slopes = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    
    x = np.arange(lookback)
    
    for i in range(lookback, n):
        y = close[i-lookback:i]
        slope, intercept, r, _, _ = stats.linregress(x, y)
        slopes[i] = slope
        r_squared[i] = r ** 2
        intercepts[i] = intercept
    
    results = pd.DataFrame({
        'trend_slope': slopes,
        'trend_r_squared': r_squared,
        'trend_intercept': intercepts
    }, index=df.index)
    
    # Trend line value at current point
    results['trend_line_value'] = (
        results['trend_intercept'] + results['trend_slope'] * (lookback - 1)
    )
    
    # Distance from trend line
    results['dist_from_trend'] = (
        df['close'] - results['trend_line_value']
    ) / results['trend_line_value']
    
    # Normalized slope (per period percentage)
    results['trend_slope_normalized'] = (
        results['trend_slope'] / df['close']
    )
    
    return results


class TrendIndicatorGenerator:
    """
    Generates all trend-related features.
    """
    
    def __init__(
        self,
        sma_windows: List[int] = [5, 10, 20, 50, 100, 200],
        ema_windows: List[int] = [5, 10, 20, 50, 100],
        ema_slope_period: int = 5,
        adx_period: int = 14,
        trend_lookback: int = 20
    ):
        """
        Initialize the generator.
        
        Args:
            sma_windows: Windows for SMA calculation
            ema_windows: Windows for EMA calculation
            ema_slope_period: Period for EMA slope
            adx_period: Period for ADX
            trend_lookback: Lookback for trend line
        """
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.ema_slope_period = ema_slope_period
        self.adx_period = adx_period
        self.trend_lookback = trend_lookback
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all trend features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with trend features added
        """
        features = df.copy()
        
        # SMA features
        sma_features = calculate_sma(df['close'], self.sma_windows)
        features = pd.concat([features, sma_features], axis=1)
        
        # EMA features
        ema_features = calculate_ema(df['close'], self.ema_windows)
        features = pd.concat([features, ema_features], axis=1)
        
        # EMA slope
        ema_slope = calculate_ema_slope(
            df['close'],
            ema_window=20,
            slope_period=self.ema_slope_period
        )
        # Rename to avoid collision
        ema_slope.columns = ['ema_20_for_slope', 'ema_20_slope', 
                            'ema_20_slope_pct', 'ema_20_slope_direction',
                            'ema_20_slope_momentum']
        features = pd.concat([features, ema_slope], axis=1)
        
        # ADX
        adx_features = calculate_adx(df, self.adx_period)
        features = pd.concat([features, adx_features], axis=1)
        
        # Trend lines
        trend_features = calculate_trend_lines(df, self.trend_lookback)
        features = pd.concat([features, trend_features], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} trend features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        names = []
        
        # SMA features
        for w in self.sma_windows:
            names.extend([f'sma_{w}', f'price_sma_{w}_pct', f'sma_{w}_slope'])
        if 20 in self.sma_windows and 50 in self.sma_windows:
            names.append('sma_20_50_cross')
        if 50 in self.sma_windows and 200 in self.sma_windows:
            names.append('sma_50_200_cross')
        
        # EMA features
        for w in self.ema_windows:
            names.extend([f'ema_{w}', f'price_ema_{w}_pct', f'ema_{w}_accel'])
        
        # ADX features
        names.extend(['adx', 'plus_di', 'minus_di', 'dx', 
                     'trend_strength', 'trend_direction', 'di_crossover'])
        
        # Trend line features
        names.extend(['trend_slope', 'trend_r_squared', 'trend_intercept',
                     'trend_line_value', 'dist_from_trend', 'trend_slope_normalized'])
        
        return names
