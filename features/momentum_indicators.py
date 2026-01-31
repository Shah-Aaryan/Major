"""
Momentum Indicator Generation.

Calculates momentum-based indicators including:
- Relative Strength Index (RSI)
- Stochastic RSI
- MACD (Moving Average Convergence Divergence)
- Rate of Change (ROC)
- Commodity Channel Index (CCI)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(
    prices: pd.Series,
    periods: List[int] = [6, 14, 21]
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index for multiple periods.
    
    RSI measures the speed and magnitude of price movements:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    - RSI = 50: Neutral
    
    Args:
        prices: Price series (typically close)
        periods: List of lookback periods
        
    Returns:
        DataFrame with RSI values and derived features
    """
    results = {}
    
    for period in periods:
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        results[f'rsi_{period}'] = rsi
        
        # RSI-based features
        results[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
        results[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
        
        # RSI momentum (change in RSI)
        results[f'rsi_{period}_momentum'] = rsi.diff(5)
        
        # Distance from neutral (50)
        results[f'rsi_{period}_from_neutral'] = rsi - 50
    
    # RSI divergence check (price vs RSI)
    if len(periods) > 0:
        main_rsi = results[f'rsi_{periods[0]}']
        price_trend = prices.diff(10) > 0
        rsi_trend = pd.Series(main_rsi).diff(10) > 0
        
        # Bearish divergence: price up, RSI down
        results['rsi_bearish_divergence'] = (
            price_trend & ~rsi_trend
        ).astype(int)
        
        # Bullish divergence: price down, RSI up
        results['rsi_bullish_divergence'] = (
            ~price_trend & rsi_trend
        ).astype(int)
    
    return pd.DataFrame(results)


def calculate_stoch_rsi(
    prices: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> pd.DataFrame:
    """
    Calculate Stochastic RSI.
    
    Stoch RSI applies the Stochastic oscillator formula to RSI values
    instead of prices, making it more sensitive to short-term changes.
    
    Args:
        prices: Price series
        rsi_period: Period for RSI calculation
        stoch_period: Period for Stochastic calculation
        smooth_k: Smoothing for %K line
        smooth_d: Smoothing for %D line
        
    Returns:
        DataFrame with Stoch RSI values
    """
    # First calculate RSI
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    avg_gains = gains.ewm(span=rsi_period, adjust=False).mean()
    avg_losses = losses.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Apply Stochastic formula to RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100
    
    # Smooth lines
    stoch_rsi_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(smooth_d).mean()
    
    results = pd.DataFrame({
        'stoch_rsi': stoch_rsi,
        'stoch_rsi_k': stoch_rsi_k,
        'stoch_rsi_d': stoch_rsi_d,
        'stoch_rsi_signal': (stoch_rsi_k > stoch_rsi_d).astype(int),
        'stoch_rsi_overbought': (stoch_rsi_k > 80).astype(int),
        'stoch_rsi_oversold': (stoch_rsi_k < 20).astype(int)
    })
    
    # Crossover signals
    results['stoch_rsi_cross_up'] = (
        (stoch_rsi_k > stoch_rsi_d) & 
        (stoch_rsi_k.shift(1) <= stoch_rsi_d.shift(1))
    ).astype(int)
    
    results['stoch_rsi_cross_down'] = (
        (stoch_rsi_k < stoch_rsi_d) & 
        (stoch_rsi_k.shift(1) >= stoch_rsi_d.shift(1))
    ).astype(int)
    
    return results


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD shows the relationship between two EMAs:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        
    Returns:
        DataFrame with MACD values and signals
    """
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    results = pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
        
        # Normalized MACD (percentage of price)
        'macd_normalized': macd_line / prices * 100,
        
        # MACD direction
        'macd_positive': (macd_line > 0).astype(int),
        'macd_above_signal': (macd_line > signal_line).astype(int),
        
        # Histogram momentum
        'macd_hist_momentum': histogram.diff(),
        'macd_hist_growing': (histogram.diff() > 0).astype(int)
    })
    
    # Crossover signals
    results['macd_cross_up'] = (
        (macd_line > signal_line) & 
        (macd_line.shift(1) <= signal_line.shift(1))
    ).astype(int)
    
    results['macd_cross_down'] = (
        (macd_line < signal_line) & 
        (macd_line.shift(1) >= signal_line.shift(1))
    ).astype(int)
    
    # Zero line crossover
    results['macd_zero_cross_up'] = (
        (macd_line > 0) & (macd_line.shift(1) <= 0)
    ).astype(int)
    
    results['macd_zero_cross_down'] = (
        (macd_line < 0) & (macd_line.shift(1) >= 0)
    ).astype(int)
    
    return results


def calculate_roc(
    prices: pd.Series,
    periods: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate Rate of Change (ROC) for multiple periods.
    
    ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100
    
    Args:
        prices: Price series
        periods: List of lookback periods
        
    Returns:
        DataFrame with ROC values
    """
    results = {}
    
    for period in periods:
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        results[f'roc_{period}'] = roc
        
        # ROC momentum
        results[f'roc_{period}_momentum'] = roc.diff()
        
        # ROC extremes
        roc_mean = roc.rolling(50).mean()
        roc_std = roc.rolling(50).std()
        results[f'roc_{period}_zscore'] = (roc - roc_mean) / (roc_std + 1e-10)
    
    return pd.DataFrame(results)


def calculate_cci(
    df: pd.DataFrame,
    period: int = 20,
    constant: float = 0.015
) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI measures the variation of price from its statistical mean.
    - CCI > 100: Strong uptrend
    - CCI < -100: Strong downtrend
    - CCI near 0: No clear trend
    
    Args:
        df: OHLCV DataFrame
        period: CCI period
        constant: Lambert constant (typically 0.015)
        
    Returns:
        DataFrame with CCI values
    """
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # SMA of typical price
    tp_sma = tp.rolling(period).mean()
    
    # Mean deviation
    mean_dev = tp.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    # CCI
    cci = (tp - tp_sma) / (constant * mean_dev + 1e-10)
    
    results = pd.DataFrame({
        'cci': cci,
        'cci_overbought': (cci > 100).astype(int),
        'cci_oversold': (cci < -100).astype(int),
        'cci_extreme': (abs(cci) > 200).astype(int)
    })
    
    return results


def calculate_williams_r(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Williams %R.
    
    Similar to Stochastic but inverted scale:
    - %R > -20: Overbought
    - %R < -80: Oversold
    
    Args:
        df: OHLCV DataFrame
        period: Lookback period
        
    Returns:
        DataFrame with Williams %R
    """
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
    
    results = pd.DataFrame({
        'williams_r': williams_r,
        'williams_r_overbought': (williams_r > -20).astype(int),
        'williams_r_oversold': (williams_r < -80).astype(int)
    })
    
    return results


class MomentumIndicatorGenerator:
    """
    Generates all momentum-related features.
    """
    
    def __init__(
        self,
        rsi_periods: List[int] = [6, 14, 21],
        stoch_rsi_period: int = 14,
        stoch_rsi_smooth: int = 3,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        roc_periods: List[int] = [5, 10, 20],
        cci_period: int = 20,
        williams_period: int = 14
    ):
        """
        Initialize the generator.
        
        Args:
            rsi_periods: Periods for RSI
            stoch_rsi_period: Period for Stoch RSI
            stoch_rsi_smooth: Smoothing for Stoch RSI
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            roc_periods: Periods for ROC
            cci_period: Period for CCI
            williams_period: Period for Williams %R
        """
        self.rsi_periods = rsi_periods
        self.stoch_rsi_period = stoch_rsi_period
        self.stoch_rsi_smooth = stoch_rsi_smooth
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.roc_periods = roc_periods
        self.cci_period = cci_period
        self.williams_period = williams_period
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all momentum features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with momentum features added
        """
        features = df.copy()
        
        # RSI
        rsi_features = calculate_rsi(df['close'], self.rsi_periods)
        features = pd.concat([features, rsi_features], axis=1)
        
        # Stochastic RSI
        stoch_rsi = calculate_stoch_rsi(
            df['close'],
            rsi_period=self.stoch_rsi_period,
            smooth_k=self.stoch_rsi_smooth,
            smooth_d=self.stoch_rsi_smooth
        )
        features = pd.concat([features, stoch_rsi], axis=1)
        
        # MACD
        macd = calculate_macd(
            df['close'],
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        features = pd.concat([features, macd], axis=1)
        
        # ROC
        roc = calculate_roc(df['close'], self.roc_periods)
        features = pd.concat([features, roc], axis=1)
        
        # CCI
        cci = calculate_cci(df, self.cci_period)
        features = pd.concat([features, cci], axis=1)
        
        # Williams %R
        williams = calculate_williams_r(df, self.williams_period)
        features = pd.concat([features, williams], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} momentum features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        names = []
        
        # RSI features
        for p in self.rsi_periods:
            names.extend([
                f'rsi_{p}', f'rsi_{p}_overbought', f'rsi_{p}_oversold',
                f'rsi_{p}_momentum', f'rsi_{p}_from_neutral'
            ])
        names.extend(['rsi_bearish_divergence', 'rsi_bullish_divergence'])
        
        # Stoch RSI features
        names.extend([
            'stoch_rsi', 'stoch_rsi_k', 'stoch_rsi_d', 'stoch_rsi_signal',
            'stoch_rsi_overbought', 'stoch_rsi_oversold',
            'stoch_rsi_cross_up', 'stoch_rsi_cross_down'
        ])
        
        # MACD features
        names.extend([
            'macd', 'macd_signal', 'macd_histogram', 'macd_normalized',
            'macd_positive', 'macd_above_signal', 'macd_hist_momentum',
            'macd_hist_growing', 'macd_cross_up', 'macd_cross_down',
            'macd_zero_cross_up', 'macd_zero_cross_down'
        ])
        
        # ROC features
        for p in self.roc_periods:
            names.extend([f'roc_{p}', f'roc_{p}_momentum', f'roc_{p}_zscore'])
        
        # CCI features
        names.extend(['cci', 'cci_overbought', 'cci_oversold', 'cci_extreme'])
        
        # Williams %R features
        names.extend(['williams_r', 'williams_r_overbought', 'williams_r_oversold'])
        
        return names
