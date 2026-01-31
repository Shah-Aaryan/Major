"""
Data Resampling Module.

Handles resampling of OHLCV data from 1-minute to higher timeframes.
Properly aggregates OHLCV data maintaining correct price relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimeFrame(Enum):
    """Supported timeframes."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"


# Mapping from user-friendly strings to pandas offset aliases
TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D"
}


class DataResampler:
    """
    Resamples OHLCV data to different timeframes.
    
    Ensures proper aggregation:
    - Open: First value in period
    - High: Maximum value in period
    - Low: Minimum value in period
    - Close: Last value in period
    - Volume: Sum of all values in period
    """
    
    def __init__(self, source_timeframe: str = "1m"):
        """
        Initialize the resampler.
        
        Args:
            source_timeframe: The timeframe of the source data
        """
        self.source_timeframe = TIMEFRAME_MAP.get(
            source_timeframe, source_timeframe
        )
    
    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
        include_extra_columns: bool = True
    ) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.
        
        Args:
            df: Source OHLCV DataFrame with DatetimeIndex
            target_timeframe: Target timeframe (e.g., '5m', '15m', '1h')
            include_extra_columns: Whether to include additional derived columns
            
        Returns:
            Resampled DataFrame
        """
        # Map timeframe to pandas offset
        target_offset = TIMEFRAME_MAP.get(target_timeframe, target_timeframe)
        
        logger.info(f"Resampling from {self.source_timeframe} to {target_offset}")
        
        # Define aggregation rules for OHLCV
        ohlcv_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Handle additional columns if present
        additional_agg = {}
        
        # Raw price columns (if normalized data)
        for col in ['open_raw', 'high_raw', 'low_raw', 'close_raw']:
            if col in df.columns:
                additional_agg[col] = ohlcv_agg[col.replace('_raw', '')]
        
        # Volume columns
        if 'volume_raw' in df.columns:
            additional_agg['volume_raw'] = 'sum'
        
        if 'quote_volume' in df.columns:
            additional_agg['quote_volume'] = 'sum'
        
        if 'trades_count' in df.columns:
            additional_agg['trades_count'] = 'sum'
        
        if 'taker_buy_volume' in df.columns:
            additional_agg['taker_buy_volume'] = 'sum'
        
        # Combine aggregation rules
        agg_dict = {**ohlcv_agg, **additional_agg}
        
        # Filter to only columns that exist in the DataFrame
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        # Perform resampling
        resampled = df.resample(target_offset).agg(agg_dict)
        
        # Drop rows where we don't have complete data
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
        
        if include_extra_columns:
            resampled = self._add_derived_columns(resampled)
        
        logger.info(
            f"Resampled: {len(df)} rows -> {len(resampled)} rows"
        )
        
        return resampled
    
    def resample_to_multiple(
        self,
        df: pd.DataFrame,
        target_timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample to multiple timeframes at once.
        
        Args:
            df: Source OHLCV DataFrame
            target_timeframes: List of target timeframes
            
        Returns:
            Dictionary mapping timeframe to resampled DataFrame
        """
        results = {}
        
        for tf in target_timeframes:
            try:
                results[tf] = self.resample(df, tf)
            except Exception as e:
                logger.error(f"Failed to resample to {tf}: {e}")
        
        return results
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns to resampled data."""
        df = df.copy()
        
        # Candle body and range
        df['body'] = df['close'] - df['open']
        df['body_pct'] = df['body'] / df['open']
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['low']
        
        # Upper and lower wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Candle direction
        df['is_bullish'] = df['close'] > df['open']
        
        # Typical price (HLC/3)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP for the candle (if we had tick data, this would be more accurate)
        df['vwap'] = df['typical_price']  # Approximation
        
        return df
    
    def get_candles_per_period(
        self,
        source_tf: str,
        target_tf: str
    ) -> int:
        """
        Calculate how many source candles fit in one target candle.
        
        Args:
            source_tf: Source timeframe
            target_tf: Target timeframe
            
        Returns:
            Number of source candles per target candle
        """
        # Convert to minutes
        tf_to_minutes = {
            "1m": 1, "1min": 1,
            "5m": 5, "5min": 5,
            "15m": 15, "15min": 15,
            "30m": 30, "30min": 30,
            "1h": 60, "1H": 60,
            "4h": 240, "4H": 240,
            "1d": 1440, "1D": 1440
        }
        
        source_minutes = tf_to_minutes.get(source_tf, 1)
        target_minutes = tf_to_minutes.get(target_tf, 1)
        
        return target_minutes // source_minutes


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    source_timeframe: str = "1m"
) -> pd.DataFrame:
    """
    Convenience function to resample OHLCV data.
    
    Args:
        df: Source OHLCV DataFrame
        target_timeframe: Target timeframe
        source_timeframe: Source timeframe
        
    Returns:
        Resampled DataFrame
    """
    resampler = DataResampler(source_timeframe)
    return resampler.resample(df, target_timeframe)


def align_multi_timeframe_data(
    data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Align data from multiple timeframes to have matching date ranges.
    
    Args:
        data_dict: Dictionary mapping timeframe to DataFrame
        
    Returns:
        Dictionary with aligned DataFrames
    """
    if not data_dict:
        return data_dict
    
    # Find common date range
    start_dates = [df.index.min() for df in data_dict.values()]
    end_dates = [df.index.max() for df in data_dict.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Filter each DataFrame
    aligned = {}
    for tf, df in data_dict.items():
        aligned[tf] = df[(df.index >= common_start) & (df.index <= common_end)]
    
    return aligned


def get_timeframe_hierarchy() -> List[str]:
    """Get timeframes in order from smallest to largest."""
    return ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
