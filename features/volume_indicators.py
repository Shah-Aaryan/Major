"""
Volume Indicator Generation.

Calculates volume-based indicators including:
- Volume Moving Averages
- Volume Spikes
- VWAP
- On-Balance Volume (OBV)
- Money Flow Index (MFI)
- Accumulation/Distribution Line (AD Line)
- Chaikin Money Flow (CMF)
- Volume Oscillator
- Ease of Movement (EOM)
- Force Index
- Price Volume Trend (PVT)
- Negative Volume Index (NVI)
- Positive Volume Index (PVI)
- Volume Profile (POC/VAH/VAL)
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
    """Calculate Volume Moving Averages and related features."""
    results = {}
    for window in windows:
        vol_ma = volume.rolling(window).mean()
        results[f'volume_ma_{window}'] = vol_ma
        results[f'volume_ratio_{window}'] = volume / (vol_ma + 1e-10)
        vol_slope = vol_ma.diff(5)
        results[f'volume_trend_{window}'] = np.sign(vol_slope)
    
    if len(windows) >= 2:
        short_ma = results[f'volume_ma_{windows[0]}']
        long_ma = results[f'volume_ma_{windows[-1]}']
        results['volume_relative'] = short_ma / (long_ma + 1e-10)
    
    return pd.DataFrame(results, index=volume.index)


def calculate_volume_spikes(
    volume: pd.Series,
    threshold: float = 2.0,
    lookback: int = 20
) -> pd.DataFrame:
    """Detect volume spikes and anomalies."""
    vol_mean = volume.rolling(lookback).mean()
    vol_std = volume.rolling(lookback).std()
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-10)
    is_spike = volume > (vol_mean * threshold)
    spike_magnitude = np.where(is_spike, volume / (vol_mean + 1e-10), 0)
    
    return pd.DataFrame({
        'volume_zscore': vol_zscore,
        'volume_spike': is_spike.astype(int),
        'volume_spike_magnitude': spike_magnitude,
        'volume_extreme_high': (vol_zscore > 3).astype(int),
        'volume_extreme_low': (vol_zscore < -2).astype(int),
        'volume_dryup': (volume < vol_mean * 0.5).astype(int)
    }, index=volume.index)


def calculate_vwap(
    df: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """Calculate Volume Weighted Average Price."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_vol = tp * df['volume']
    cum_tp_vol = tp_vol.rolling(period).sum()
    cum_vol = df['volume'].rolling(period).sum()
    vwap = cum_tp_vol / (cum_vol + 1e-10)
    vwap_std = (tp - vwap).rolling(period).std()
    
    return pd.DataFrame({
        'vwap': vwap,
        'vwap_upper': vwap + 2 * vwap_std,
        'vwap_lower': vwap - 2 * vwap_std,
        'price_vs_vwap': (df['close'] - vwap) / (vwap + 1e-10) * 100,
        'price_above_vwap': (df['close'] > vwap).astype(int),
        'vwap_slope': vwap.diff(5) / (vwap.shift(5) + 1e-10)
    }, index=df.index)


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate On-Balance Volume (OBV)."""
    close = df['close']
    volume = df['volume']
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    obv_ma_20 = obv.rolling(20).mean()
    obv_slope = obv.diff(10)
    
    results = pd.DataFrame({
        'obv': obv,
        'obv_ma_20': obv_ma_20,
        'obv_slope': obv_slope,
        'obv_above_ma': (obv > obv_ma_20).astype(int)
    }, index=df.index)
    return results


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Money Flow Index (MFI)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = tp * df['volume']
    tp_diff = tp.diff()
    pos_mf = np.where(tp_diff > 0, raw_mf, 0)
    neg_mf = np.where(tp_diff < 0, raw_mf, 0)
    pos_mf_sum = pd.Series(pos_mf, index=df.index).rolling(period).sum()
    neg_mf_sum = pd.Series(neg_mf, index=df.index).rolling(period).sum()
    mf_ratio = pos_mf_sum / (neg_mf_sum + 1e-10)
    mfi = 100 - (100 / (1 + mf_ratio))
    return pd.DataFrame({
        'mfi': mfi,
        'mfi_overbought': (mfi > 80).astype(int),
        'mfi_oversold': (mfi < 20).astype(int)
    }, index=df.index)


def calculate_ad_line(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Accumulation/Distribution Line."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    ad = mfv.cumsum()
    return pd.DataFrame({
        'ad_line': ad,
        'mfm': mfm
    }, index=df.index)


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow (CMF)."""
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    mfv = mfm * df['volume']
    return mfv.rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)


def calculate_volume_oscillator(volume: pd.Series, fast_window: int = 14, slow_window: int = 28) -> pd.Series:
    """Calculate Volume Oscillator."""
    fast_ma = volume.rolling(fast_window).mean()
    slow_ma = volume.rolling(slow_window).mean()
    return (fast_ma - slow_ma) / (slow_ma + 1e-10) * 100


def calculate_eom(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Ease of Movement (EOM)."""
    dist = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
    box_ratio = (df['volume'] / 10000.0) / (df['high'] - df['low'] + 1e-10)
    emv = dist / (box_ratio + 1e-10)
    return emv.ewm(span=period, adjust=False).mean()


def calculate_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    """Calculate Force Index."""
    raw_fi = df['close'].diff() * df['volume']
    return raw_fi.ewm(span=period, adjust=False).mean()


def calculate_pvt(df: pd.DataFrame) -> pd.Series:
    """Calculate Price Volume Trend (PVT)."""
    pct = df['close'].pct_change()
    return (pct * df['volume']).cumsum()


def calculate_nvi(df: pd.DataFrame) -> pd.Series:
    """Calculate Negative Volume Index (NVI)."""
    n = len(df)
    nvi = np.ones(n) * 1000.0
    vol = df['volume'].values
    close = df['close'].values
    for i in range(1, n):
        if vol[i] < vol[i-1] and close[i-1] > 0:
            nvi[i] = nvi[i-1] + nvi[i-1] * ((close[i] - close[i-1]) / close[i-1])
        else:
            nvi[i] = nvi[i-1]
    return pd.Series(nvi, index=df.index)


def calculate_pvi(df: pd.DataFrame) -> pd.Series:
    """Calculate Positive Volume Index (PVI)."""
    n = len(df)
    pvi = np.ones(n) * 1000.0
    vol = df['volume'].values
    close = df['close'].values
    for i in range(1, n):
        if vol[i] > vol[i-1] and close[i-1] > 0:
            pvi[i] = pvi[i-1] + pvi[i-1] * ((close[i] - close[i-1]) / close[i-1])
        else:
            pvi[i] = pvi[i-1]
    return pd.Series(pvi, index=df.index)


class VolumeIndicatorGenerator:
    """Generates all volume-related features."""
    
    def __init__(
        self,
        volume_ma_windows: List[int] = [10, 20, 50],
        volume_spike_threshold: float = 2.0,
        vwap_period: int = 20,
        mfi_period: int = 14
    ):
        self.volume_ma_windows = volume_ma_windows
        self.volume_spike_threshold = volume_spike_threshold
        self.vwap_period = vwap_period
        self.mfi_period = mfi_period
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        features = pd.concat([features, calculate_volume_ma(df['volume'], self.volume_ma_windows)], axis=1)
        features = pd.concat([features, calculate_volume_spikes(df['volume'], self.volume_spike_threshold)], axis=1)
        features = pd.concat([features, calculate_vwap(df, self.vwap_period)], axis=1)
        features = pd.concat([features, calculate_obv(df)], axis=1)
        features = pd.concat([features, calculate_mfi(df, self.mfi_period)], axis=1)
        features = pd.concat([features, calculate_ad_line(df)], axis=1)
        
        # New volume indicators
        features['cmf_20'] = calculate_cmf(df)
        features['volume_oscillator'] = calculate_volume_oscillator(df['volume'])
        features['eom_14'] = calculate_eom(df)
        features['force_index_13'] = calculate_force_index(df)
        features['pvt'] = calculate_pvt(df)
        features['nvi'] = calculate_nvi(df)
        features['pvi'] = calculate_pvi(df)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} volume features")
        return features
