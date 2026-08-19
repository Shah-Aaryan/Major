"""
Volatility Indicator Generation.

Calculates volatility-based indicators including:
- Average True Range (ATR)
- Rolling Volatility
- Bollinger Bands
- Keltner Channels
- Donchian Channels
- Chaikin Volatility
- Ulcer Index
- Parkinson / Garman-Klass / Yang-Zhang Volatility
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_atr(
    df: pd.DataFrame,
    periods: List[int] = [7, 14, 21]
) -> pd.DataFrame:
    """Calculate Average True Range for multiple periods."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    results = {'true_range': true_range}
    
    for period in periods:
        atr = true_range.ewm(span=period, adjust=False).mean()
        results[f'atr_{period}'] = atr
        results[f'atr_{period}_pct'] = atr / (close + 1e-10) * 100
        atr_mean = atr.rolling(period * 5).mean()
        results[f'atr_{period}_ratio'] = atr / (atr_mean + 1e-10)
        results[f'atr_{period}_expanding'] = (atr > atr.shift(1)).astype(int)
    
    return pd.DataFrame(results, index=df.index)


def calculate_rolling_volatility(
    prices: pd.Series,
    windows: List[int] = [10, 20, 60]
) -> pd.DataFrame:
    """Calculate rolling historical volatility."""
    log_returns = np.log(prices / (prices.shift(1) + 1e-10))
    results = {}
    for window in windows:
        vol = log_returns.rolling(window).std()
        results[f'volatility_{window}'] = vol
        ann_factor = np.sqrt(525600)
        results[f'volatility_{window}_ann'] = vol * ann_factor
        vol_mean = vol.rolling(window * 5).mean()
        vol_std = vol.rolling(window * 5).std()
        results[f'volatility_{window}_zscore'] = (vol - vol_mean) / (vol_std + 1e-10)
    
    if len(windows) > 0:
        main_vol = results[f'volatility_{windows[0]}']
        vol_median = main_vol.rolling(200).median()
        results['volatility_regime'] = np.where(
            main_vol > vol_median * 1.5, 'high',
            np.where(main_vol < vol_median * 0.5, 'low', 'normal')
        )
    return pd.DataFrame(results, index=prices.index)


def calculate_chaikin_volatility(df: pd.DataFrame, window: int = 10, roc_window: int = 10) -> pd.Series:
    """Calculate Chaikin Volatility."""
    hl = df['high'] - df['low']
    ema_hl = hl.ewm(span=window, adjust=False).mean()
    return (ema_hl - ema_hl.shift(roc_window)) / (ema_hl.shift(roc_window) + 1e-10) * 100


def calculate_ulcer_index(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Ulcer Index."""
    max_price = prices.rolling(period).max()
    drawdown = (prices - max_price) / (max_price + 1e-10) * 100
    squared_dd = drawdown ** 2
    return np.sqrt(squared_dd.rolling(period).mean())


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    band_width = (upper_band - lower_band) / (sma + 1e-10) * 100
    percent_b = (prices - lower_band) / (upper_band - lower_band + 1e-10)
    
    results = pd.DataFrame({
        'bb_middle': sma,
        'bb_upper': upper_band,
        'bb_lower': lower_band,
        'bb_width': band_width,
        'bb_percent_b': percent_b,
        'bb_above_upper': (prices > upper_band).astype(int),
        'bb_below_lower': (prices < lower_band).astype(int),
        'bb_in_bands': ((prices >= lower_band) & (prices <= upper_band)).astype(int),
        'bb_dist_upper': (upper_band - prices) / (prices + 1e-10) * 100,
        'bb_dist_lower': (prices - lower_band) / (prices + 1e-10) * 100,
    }, index=prices.index)
    return results


def calculate_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0
) -> pd.DataFrame:
    """Calculate Keltner Channels."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    middle = typical_price.ewm(span=ema_period, adjust=False).mean()
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    upper = middle + (atr_multiplier * atr)
    lower = middle - (atr_multiplier * atr)
    
    return pd.DataFrame({
        'kc_middle': middle,
        'kc_upper': upper,
        'kc_lower': lower,
        'kc_width': (upper - lower) / (middle + 1e-10) * 100,
        'kc_position': (df['close'] - lower) / (upper - lower + 1e-10),
    }, index=df.index)


def calculate_donchian_channels(
    df: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """Calculate Donchian Channels."""
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({
        'dc_upper': upper,
        'dc_lower': lower,
        'dc_middle': middle,
        'dc_width': (upper - lower) / (lower + 1e-10) * 100,
        'dc_position': (df['close'] - lower) / (upper - lower + 1e-10),
    }, index=df.index)


def calculate_volatility_metrics(
    df: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """Calculate additional volatility metrics."""
    close = df['close']
    high = df['high']
    low = df['low']
    
    parkinson = np.sqrt((1 / (4 * np.log(2))) * ((np.log(high / (low + 1e-10)) ** 2).rolling(period).mean()))
    log_hl = np.log(high / (low + 1e-10)) ** 2
    log_co = np.log(close / (df['open'] + 1e-10)) ** 2
    gk = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(period).mean())
    
    log_oc = np.log(df['open'] / (close.shift(1) + 1e-10))
    log_co = np.log(close / (df['open'] + 1e-10))
    overnight_var = log_oc.rolling(period).var()
    open_close_var = log_co.rolling(period).var()
    rs_var = (log_hl / (4 * np.log(2))).rolling(period).mean()
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    yz = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
    
    return pd.DataFrame({
        'vol_parkinson': parkinson,
        'vol_garman_klass': gk,
        'vol_yang_zhang': yz,
        'vol_intraday': ((high - low) / (close + 1e-10)).rolling(period).mean(),
        'vol_close_to_close': close.pct_change().rolling(period).std()
    }, index=df.index)


class VolatilityIndicatorGenerator:
    """Generates all volatility-related features."""
    
    def __init__(
        self,
        atr_periods: List[int] = [7, 14, 21],
        rolling_vol_windows: List[int] = [10, 20, 60],
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
        keltner_ema: int = 20,
        keltner_atr: int = 10,
        donchian_period: int = 20
    ):
        self.atr_periods = atr_periods
        self.rolling_vol_windows = rolling_vol_windows
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.keltner_ema = keltner_ema
        self.keltner_atr = keltner_atr
        self.donchian_period = donchian_period
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        features = pd.concat([features, calculate_atr(df, self.atr_periods)], axis=1)
        features = pd.concat([features, calculate_rolling_volatility(df['close'], self.rolling_vol_windows)], axis=1)
        features = pd.concat([features, calculate_bollinger_bands(df['close'], self.bollinger_window, self.bollinger_std)], axis=1)
        features = pd.concat([features, calculate_keltner_channels(df, self.keltner_ema, self.keltner_atr)], axis=1)
        features = pd.concat([features, calculate_donchian_channels(df, self.donchian_period)], axis=1)
        features = pd.concat([features, calculate_volatility_metrics(df)], axis=1)
        
        # New volatility indicators
        features['chaikin_volatility'] = calculate_chaikin_volatility(df)
        features['ulcer_index'] = calculate_ulcer_index(df['close'])
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} volatility features")
        return features
