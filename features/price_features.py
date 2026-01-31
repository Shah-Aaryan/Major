"""
Price and Returns Feature Generation.

Calculates price-based features including:
- Log returns
- Rolling returns
- Price momentum
- Gap returns
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def calculate_log_returns(
    prices: pd.Series,
    periods: Union[int, List[int]] = 1
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate log returns over specified periods.
    
    Log returns are preferred for:
    - Time additivity (sum of log returns = total log return)
    - Approximate normality
    - Mathematical convenience in modeling
    
    Args:
        prices: Price series (typically close prices)
        periods: Single period or list of periods
        
    Returns:
        Series for single period, DataFrame for multiple periods
    """
    if isinstance(periods, int):
        log_ret = np.log(prices / prices.shift(periods))
        log_ret.name = f'log_return_{periods}'
        return log_ret
    
    results = {}
    for p in periods:
        results[f'log_return_{p}'] = np.log(prices / prices.shift(p))
    
    return pd.DataFrame(results)


def calculate_rolling_returns(
    prices: pd.Series,
    windows: List[int] = [5, 15, 60, 240]
) -> pd.DataFrame:
    """
    Calculate rolling cumulative returns over different windows.
    
    Args:
        prices: Price series
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling returns for each window
    """
    results = {}
    
    for window in windows:
        # Simple return over window
        results[f'rolling_return_{window}'] = (
            prices / prices.shift(window) - 1
        )
        
        # Annualized rolling return (assuming 1-min data, 525600 mins/year)
        # For crypto 24/7: 365 * 24 * 60 = 525600
        annualization_factor = np.sqrt(525600 / window)
        results[f'rolling_return_{window}_ann'] = (
            results[f'rolling_return_{window}'] * annualization_factor
        )
    
    return pd.DataFrame(results)


def calculate_price_momentum(
    prices: pd.Series,
    periods: List[int] = [10, 20, 50],
    normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate price momentum indicators.
    
    Momentum = Current Price - Price N periods ago
    Normalized Momentum = Momentum / Price N periods ago (as percentage)
    
    Args:
        prices: Price series
        periods: Lookback periods for momentum
        normalize: Whether to normalize by past price
        
    Returns:
        DataFrame with momentum values
    """
    results = {}
    
    for period in periods:
        if normalize:
            results[f'momentum_{period}'] = (
                (prices - prices.shift(period)) / prices.shift(period)
            )
        else:
            results[f'momentum_{period}'] = prices - prices.shift(period)
        
        # Momentum rate of change (second derivative)
        results[f'momentum_roc_{period}'] = (
            results[f'momentum_{period}'] - 
            results[f'momentum_{period}'].shift(1)
        )
    
    return pd.DataFrame(results)


def calculate_gap_returns(
    df: pd.DataFrame,
    session_break_hours: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculate gap returns (for traditional markets with sessions).
    
    For crypto (24/7 market), this calculates:
    - Hourly open vs previous hourly close
    - Daily open vs previous daily close
    
    Args:
        df: OHLCV DataFrame
        session_break_hours: Hours where sessions start (for traditional markets)
        
    Returns:
        DataFrame with gap features
    """
    results = {}
    
    # For crypto: look at larger timeframe gaps
    # Calculate open vs previous close gap
    results['gap_oc'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Rolling gap analysis
    results['gap_5'] = (df['open'] - df['close'].shift(5)) / df['close'].shift(5)
    results['gap_15'] = (df['open'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # High-Low range as indicator of gap fill potential
    results['range_pct'] = (df['high'] - df['low']) / df['low']
    
    # Gap fill indicator (did price return to previous close?)
    prev_close = df['close'].shift(1)
    gap_up = df['open'] > prev_close
    gap_down = df['open'] < prev_close
    
    # Gap filled if low touched previous close (for gap up)
    # or high touched previous close (for gap down)
    results['gap_filled'] = (
        (gap_up & (df['low'] <= prev_close)) |
        (gap_down & (df['high'] >= prev_close))
    ).astype(int)
    
    return pd.DataFrame(results)


def calculate_price_levels(
    df: pd.DataFrame,
    lookback: int = 100
) -> pd.DataFrame:
    """
    Calculate support/resistance levels and price position.
    
    Args:
        df: OHLCV DataFrame
        lookback: Period for calculating levels
        
    Returns:
        DataFrame with price level features
    """
    results = {}
    
    # Rolling high and low (resistance/support proxies)
    results['rolling_high'] = df['high'].rolling(lookback).max()
    results['rolling_low'] = df['low'].rolling(lookback).min()
    
    # Price position within range (0 = at support, 1 = at resistance)
    range_size = results['rolling_high'] - results['rolling_low']
    results['price_position'] = (
        (df['close'] - results['rolling_low']) / 
        range_size.replace(0, np.nan)
    )
    
    # Distance from high/low as percentage
    results['dist_from_high_pct'] = (
        (results['rolling_high'] - df['close']) / df['close']
    )
    results['dist_from_low_pct'] = (
        (df['close'] - results['rolling_low']) / df['close']
    )
    
    # New high/low indicators
    results['is_new_high'] = (
        df['high'] >= df['high'].rolling(lookback).max().shift(1)
    ).astype(int)
    results['is_new_low'] = (
        df['low'] <= df['low'].rolling(lookback).min().shift(1)
    ).astype(int)
    
    return pd.DataFrame(results)


class PriceFeatureGenerator:
    """
    Generates all price-based features.
    
    This class provides a unified interface for generating
    price and return features from OHLCV data.
    """
    
    def __init__(
        self,
        log_return_periods: List[int] = [1, 5, 15, 60],
        rolling_return_windows: List[int] = [5, 15, 60, 240],
        momentum_periods: List[int] = [10, 20, 50],
        price_level_lookback: int = 100
    ):
        """
        Initialize the generator with configuration.
        
        Args:
            log_return_periods: Periods for log return calculation
            rolling_return_windows: Windows for rolling returns
            momentum_periods: Periods for momentum indicators
            price_level_lookback: Lookback for support/resistance
        """
        self.log_return_periods = log_return_periods
        self.rolling_return_windows = rolling_return_windows
        self.momentum_periods = momentum_periods
        self.price_level_lookback = price_level_lookback
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all price features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with original data plus price features
        """
        features = df.copy()
        
        # Log returns
        log_returns = calculate_log_returns(
            df['close'], 
            self.log_return_periods
        )
        features = pd.concat([features, log_returns], axis=1)
        
        # Rolling returns
        rolling_returns = calculate_rolling_returns(
            df['close'],
            self.rolling_return_windows
        )
        features = pd.concat([features, rolling_returns], axis=1)
        
        # Price momentum
        momentum = calculate_price_momentum(
            df['close'],
            self.momentum_periods
        )
        features = pd.concat([features, momentum], axis=1)
        
        # Gap returns
        gaps = calculate_gap_returns(df)
        features = pd.concat([features, gaps], axis=1)
        
        # Price levels
        levels = calculate_price_levels(df, self.price_level_lookback)
        features = pd.concat([features, levels], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} price features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        names = []
        
        # Log returns
        for p in self.log_return_periods:
            names.append(f'log_return_{p}')
        
        # Rolling returns
        for w in self.rolling_return_windows:
            names.extend([f'rolling_return_{w}', f'rolling_return_{w}_ann'])
        
        # Momentum
        for p in self.momentum_periods:
            names.extend([f'momentum_{p}', f'momentum_roc_{p}'])
        
        # Gaps
        names.extend(['gap_oc', 'gap_5', 'gap_15', 'range_pct', 'gap_filled'])
        
        # Price levels
        names.extend([
            'rolling_high', 'rolling_low', 'price_position',
            'dist_from_high_pct', 'dist_from_low_pct',
            'is_new_high', 'is_new_low'
        ])
        
        return names
