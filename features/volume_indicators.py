"""
Volume Indicator Generation.

Calculates volume-based indicators including:
- Volume Moving Averages
- Volume Spikes
- VWAP (Volume Weighted Average Price)
- On-Balance Volume (OBV)
- Volume Profile features
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_volume_ma(
    volume: pd.Series,
    windows: List[int] = [10, 20, 50]
) -> pd.DataFrame:
    """
    Calculate Volume Moving Averages and related features.
    
    Args:
        volume: Volume series
        windows: List of window sizes
        
    Returns:
        DataFrame with volume MA features
    """
    results = {}
    
    for window in windows:
        # Simple moving average of volume
        vol_ma = volume.rolling(window).mean()
        results[f'volume_ma_{window}'] = vol_ma
        
        # Volume ratio (current / MA)
        results[f'volume_ratio_{window}'] = volume / (vol_ma + 1e-10)
        
        # Volume trend (is volume increasing?)
        vol_slope = vol_ma.diff(5)
        results[f'volume_trend_{window}'] = np.sign(vol_slope)
        
        # Volume percentile
        results[f'volume_percentile_{window}'] = volume.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    
    # Relative volume (current vs historical)
    if len(windows) >= 2:
        short_ma = results[f'volume_ma_{windows[0]}']
        long_ma = results[f'volume_ma_{windows[-1]}']
        results['volume_relative'] = short_ma / (long_ma + 1e-10)
    
    return pd.DataFrame(results)


def calculate_volume_spikes(
    volume: pd.Series,
    threshold: float = 2.0,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Detect volume spikes and anomalies.
    
    Volume spikes often precede significant price movements.
    
    Args:
        volume: Volume series
        threshold: Multiplier to consider a spike (e.g., 2x average)
        lookback: Period for calculating baseline volume
        
    Returns:
        DataFrame with spike indicators
    """
    # Calculate baseline volume
    vol_mean = volume.rolling(lookback).mean()
    vol_std = volume.rolling(lookback).std()
    
    # Volume z-score
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-10)
    
    # Spike detection
    is_spike = volume > (vol_mean * threshold)
    
    # Spike magnitude
    spike_magnitude = np.where(is_spike, volume / vol_mean, 0)
    
    # Consecutive spike count
    spike_count = is_spike.astype(int)
    spike_cumsum = spike_count.groupby((~spike_count.astype(bool)).cumsum()).cumsum()
    
    results = pd.DataFrame({
        'volume_zscore': vol_zscore,
        'volume_spike': is_spike.astype(int),
        'volume_spike_magnitude': spike_magnitude,
        'volume_spike_count': spike_cumsum,
        
        # Extreme volume (both high and low)
        'volume_extreme_high': (vol_zscore > 3).astype(int),
        'volume_extreme_low': (vol_zscore < -2).astype(int),
        
        # Volume dry-up (very low volume)
        'volume_dryup': (volume < vol_mean * 0.5).astype(int)
    })
    
    return results


def calculate_vwap(
    df: pd.DataFrame,
    period: int = None,
    cumulative: bool = False
) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price.
    
    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    
    For intraday: reset at each session
    For rolling: use specified period
    
    Args:
        df: OHLCV DataFrame
        period: Rolling period (None for cumulative)
        cumulative: Whether to use cumulative VWAP
        
    Returns:
        DataFrame with VWAP features
    """
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Price-volume product
    tp_vol = tp * df['volume']
    
    if cumulative:
        # Session-based cumulative VWAP
        # For crypto 24/7, use daily reset
        df_temp = df.copy()
        df_temp['date'] = df_temp.index.date
        
        cum_tp_vol = tp_vol.groupby(df_temp['date']).cumsum()
        cum_vol = df['volume'].groupby(df_temp['date']).cumsum()
        vwap = cum_tp_vol / (cum_vol + 1e-10)
    else:
        # Rolling VWAP
        period = period or 20
        cum_tp_vol = tp_vol.rolling(period).sum()
        cum_vol = df['volume'].rolling(period).sum()
        vwap = cum_tp_vol / (cum_vol + 1e-10)
    
    # VWAP bands (standard deviation)
    if period:
        vwap_std = (tp - vwap).rolling(period).std()
    else:
        vwap_std = (tp - vwap).expanding().std()
    
    results = pd.DataFrame({
        'vwap': vwap,
        'vwap_upper': vwap + 2 * vwap_std,
        'vwap_lower': vwap - 2 * vwap_std,
        
        # Price position relative to VWAP
        'price_vs_vwap': (df['close'] - vwap) / (vwap + 1e-10) * 100,
        'price_above_vwap': (df['close'] > vwap).astype(int),
        
        # VWAP trend
        'vwap_slope': vwap.diff(5) / (vwap.shift(5) + 1e-10)
    })
    
    return results


def calculate_obv(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV is a cumulative indicator that adds volume on up days
    and subtracts volume on down days.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with OBV features
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate direction
    direction = np.sign(close.diff())
    
    # OBV
    obv = (direction * volume).cumsum()
    
    # OBV moving averages
    obv_ma_10 = obv.rolling(10).mean()
    obv_ma_20 = obv.rolling(20).mean()
    
    # OBV trend
    obv_slope = obv.diff(10)
    
    results = pd.DataFrame({
        'obv': obv,
        'obv_ma_10': obv_ma_10,
        'obv_ma_20': obv_ma_20,
        'obv_slope': obv_slope,
        'obv_normalized': (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10),
        
        # OBV divergence from price
        'obv_above_ma': (obv > obv_ma_20).astype(int)
    })
    
    # Check for OBV-Price divergence
    price_trend = close.diff(10) > 0
    obv_trend = obv_slope > 0
    
    results['obv_bullish_div'] = (~price_trend & obv_trend).astype(int)
    results['obv_bearish_div'] = (price_trend & ~obv_trend).astype(int)
    
    return results


def calculate_mfi(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI).
    
    MFI is a volume-weighted RSI.
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    
    Args:
        df: OHLCV DataFrame
        period: MFI period
        
    Returns:
        DataFrame with MFI features
    """
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Raw money flow
    raw_mf = tp * df['volume']
    
    # Positive and negative money flow
    tp_diff = tp.diff()
    pos_mf = np.where(tp_diff > 0, raw_mf, 0)
    neg_mf = np.where(tp_diff < 0, raw_mf, 0)
    
    # Money flow ratio
    pos_mf_sum = pd.Series(pos_mf, index=df.index).rolling(period).sum()
    neg_mf_sum = pd.Series(neg_mf, index=df.index).rolling(period).sum()
    
    mf_ratio = pos_mf_sum / (neg_mf_sum + 1e-10)
    
    # MFI
    mfi = 100 - (100 / (1 + mf_ratio))
    
    results = pd.DataFrame({
        'mfi': mfi,
        'mfi_overbought': (mfi > 80).astype(int),
        'mfi_oversold': (mfi < 20).astype(int),
        'mfi_momentum': mfi.diff(5)
    })
    
    return results


def calculate_ad_line(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Accumulation/Distribution Line.
    
    AD = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with AD Line features
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # AD Line (cumulative)
    ad = mfv.cumsum()
    
    results = pd.DataFrame({
        'ad_line': ad,
        'ad_line_ma': ad.rolling(20).mean(),
        'ad_line_slope': ad.diff(10),
        'mfm': mfm  # Money Flow Multiplier
    })
    
    return results


def calculate_volume_profile(
    df: pd.DataFrame,
    lookback: int = 100,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate Volume Profile features.
    
    Volume Profile shows the distribution of volume at different price levels.
    
    Args:
        df: OHLCV DataFrame
        lookback: Period for profile calculation
        n_bins: Number of price bins
        
    Returns:
        DataFrame with volume profile features
    """
    results = {}
    
    # Rolling volume profile
    def calc_poc_vah_val(window_df):
        """Calculate Point of Control, Value Area High/Low."""
        if len(window_df) < n_bins:
            return pd.Series([np.nan, np.nan, np.nan])
        
        # Create price bins
        price_min = window_df['low'].min()
        price_max = window_df['high'].max()
        bins = np.linspace(price_min, price_max, n_bins + 1)
        
        # Assign volume to bins based on close price
        bin_idx = np.digitize(window_df['close'], bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        vol_by_bin = np.zeros(n_bins)
        for i, v in zip(bin_idx, window_df['volume']):
            vol_by_bin[i] += v
        
        # Point of Control (price level with most volume)
        poc_idx = np.argmax(vol_by_bin)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_vol = vol_by_bin.sum()
        target_vol = total_vol * 0.7
        
        # Start from POC and expand
        cum_vol = vol_by_bin[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while cum_vol < target_vol and (low_idx > 0 or high_idx < n_bins - 1):
            add_low = vol_by_bin[low_idx - 1] if low_idx > 0 else 0
            add_high = vol_by_bin[high_idx + 1] if high_idx < n_bins - 1 else 0
            
            if add_low >= add_high and low_idx > 0:
                low_idx -= 1
                cum_vol += add_low
            elif high_idx < n_bins - 1:
                high_idx += 1
                cum_vol += add_high
            else:
                low_idx -= 1
                cum_vol += add_low
        
        val = bins[low_idx]  # Value Area Low
        vah = bins[high_idx + 1]  # Value Area High
        
        return pd.Series([poc, vah, val])
    
    # Calculate for each point (simplified - using last lookback candles)
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = df.iloc[i-lookback:i]
        profile = calc_poc_vah_val(window)
        poc[i] = profile.iloc[0]
        vah[i] = profile.iloc[1]
        val[i] = profile.iloc[2]
    
    results = pd.DataFrame({
        'vp_poc': poc,
        'vp_vah': vah,
        'vp_val': val,
        'price_vs_poc': (df['close'].values - poc) / (poc + 1e-10) * 100,
        'price_in_value_area': ((df['close'].values >= val) & 
                                (df['close'].values <= vah)).astype(int)
    }, index=df.index)
    
    return results


class VolumeIndicatorGenerator:
    """
    Generates all volume-related features.
    """
    
    def __init__(
        self,
        volume_ma_windows: List[int] = [10, 20, 50],
        volume_spike_threshold: float = 2.0,
        vwap_period: int = 20,
        mfi_period: int = 14,
        volume_profile_lookback: int = 100
    ):
        """
        Initialize the generator.
        """
        self.volume_ma_windows = volume_ma_windows
        self.volume_spike_threshold = volume_spike_threshold
        self.vwap_period = vwap_period
        self.mfi_period = mfi_period
        self.volume_profile_lookback = volume_profile_lookback
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volume features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with volume features added
        """
        features = df.copy()
        
        # Volume MAs
        vol_ma = calculate_volume_ma(df['volume'], self.volume_ma_windows)
        features = pd.concat([features, vol_ma], axis=1)
        
        # Volume spikes
        vol_spikes = calculate_volume_spikes(
            df['volume'], self.volume_spike_threshold
        )
        features = pd.concat([features, vol_spikes], axis=1)
        
        # VWAP
        vwap = calculate_vwap(df, self.vwap_period)
        features = pd.concat([features, vwap], axis=1)
        
        # OBV
        obv = calculate_obv(df)
        features = pd.concat([features, obv], axis=1)
        
        # MFI
        mfi = calculate_mfi(df, self.mfi_period)
        features = pd.concat([features, mfi], axis=1)
        
        # AD Line
        ad = calculate_ad_line(df)
        features = pd.concat([features, ad], axis=1)
        
        # Volume Profile (computationally expensive, optional)
        # vol_profile = calculate_volume_profile(df, self.volume_profile_lookback)
        # features = pd.concat([features, vol_profile], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} volume features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        names = []
        
        # Volume MA features
        for w in self.volume_ma_windows:
            names.extend([
                f'volume_ma_{w}', f'volume_ratio_{w}',
                f'volume_trend_{w}', f'volume_percentile_{w}'
            ])
        names.append('volume_relative')
        
        # Volume spike features
        names.extend([
            'volume_zscore', 'volume_spike', 'volume_spike_magnitude',
            'volume_spike_count', 'volume_extreme_high', 'volume_extreme_low',
            'volume_dryup'
        ])
        
        # VWAP features
        names.extend([
            'vwap', 'vwap_upper', 'vwap_lower',
            'price_vs_vwap', 'price_above_vwap', 'vwap_slope'
        ])
        
        # OBV features
        names.extend([
            'obv', 'obv_ma_10', 'obv_ma_20', 'obv_slope',
            'obv_normalized', 'obv_above_ma',
            'obv_bullish_div', 'obv_bearish_div'
        ])
        
        # MFI features
        names.extend(['mfi', 'mfi_overbought', 'mfi_oversold', 'mfi_momentum'])
        
        # AD Line features
        names.extend(['ad_line', 'ad_line_ma', 'ad_line_slope', 'mfm'])
        
        return names
