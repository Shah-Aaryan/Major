"""
Volatility Indicator Generation.

Calculates volatility-based indicators including:
- Average True Range (ATR)
- Rolling Volatility
- Bollinger Bands
- Keltner Channels
- Volatility regime detection
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
    """
    Calculate Average True Range for multiple periods.
    
    ATR measures market volatility by decomposing the entire range of a price.
    High ATR = High volatility
    Low ATR = Low volatility
    
    Args:
        df: OHLCV DataFrame
        periods: List of ATR periods
        
    Returns:
        DataFrame with ATR values and derived features
    """
    # Calculate True Range
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    results = {'true_range': true_range}
    
    for period in periods:
        # ATR using Wilder's smoothing (similar to EMA)
        atr = true_range.ewm(span=period, adjust=False).mean()
        results[f'atr_{period}'] = atr
        
        # ATR as percentage of price
        results[f'atr_{period}_pct'] = atr / close * 100
        
        # ATR ratio (current ATR vs longer-term average)
        atr_mean = atr.rolling(period * 5).mean()
        results[f'atr_{period}_ratio'] = atr / (atr_mean + 1e-10)
        
        # ATR expansion/contraction
        results[f'atr_{period}_expanding'] = (atr > atr.shift(1)).astype(int)
    
    return pd.DataFrame(results)


def calculate_rolling_volatility(
    prices: pd.Series,
    windows: List[int] = [10, 20, 60]
) -> pd.DataFrame:
    """
    Calculate rolling historical volatility.
    
    Uses log returns to calculate annualized volatility.
    For crypto (24/7): annualization factor = sqrt(525600)
    
    Args:
        prices: Price series
        windows: Rolling window sizes
        
    Returns:
        DataFrame with volatility values
    """
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))
    
    results = {}
    
    for window in windows:
        # Standard deviation of returns
        vol = log_returns.rolling(window).std()
        results[f'volatility_{window}'] = vol
        
        # Annualized volatility (assuming 1-minute data)
        # sqrt(525600) for crypto 24/7 market
        ann_factor = np.sqrt(525600)
        results[f'volatility_{window}_ann'] = vol * ann_factor
        
        # Volatility percentile (current vs historical)
        vol_pctl = vol.rolling(window * 10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        results[f'volatility_{window}_percentile'] = vol_pctl
        
        # Volatility Z-score
        vol_mean = vol.rolling(window * 5).mean()
        vol_std = vol.rolling(window * 5).std()
        results[f'volatility_{window}_zscore'] = (vol - vol_mean) / (vol_std + 1e-10)
    
    # Volatility regime
    if len(windows) > 0:
        main_vol = results[f'volatility_{windows[0]}']
        vol_median = main_vol.rolling(200).median()
        results['volatility_regime'] = np.where(
            main_vol > vol_median * 1.5, 'high',
            np.where(main_vol < vol_median * 0.5, 'low', 'normal')
        )
    
    return pd.DataFrame(results)


def calculate_bollinger_bands(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: SMA of price
    - Upper Band: SMA + (num_std * standard deviation)
    - Lower Band: SMA - (num_std * standard deviation)
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations
        
    Returns:
        DataFrame with Bollinger Band values and signals
    """
    # Calculate SMA and standard deviation
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    
    # Calculate bands
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    # Band width (volatility indicator)
    band_width = (upper_band - lower_band) / sma * 100
    
    # %B (price position within bands)
    percent_b = (prices - lower_band) / (upper_band - lower_band + 1e-10)
    
    results = pd.DataFrame({
        'bb_middle': sma,
        'bb_upper': upper_band,
        'bb_lower': lower_band,
        'bb_width': band_width,
        'bb_percent_b': percent_b,
        
        # Position indicators
        'bb_above_upper': (prices > upper_band).astype(int),
        'bb_below_lower': (prices < lower_band).astype(int),
        'bb_in_bands': (
            (prices >= lower_band) & (prices <= upper_band)
        ).astype(int),
        
        # Distance from bands
        'bb_dist_upper': (upper_band - prices) / prices * 100,
        'bb_dist_lower': (prices - lower_band) / prices * 100,
        
        # Squeeze indicator (low volatility)
        'bb_squeeze': (
            band_width < band_width.rolling(100).quantile(0.2)
        ).astype(int),
        
        # Expansion indicator (high volatility)
        'bb_expansion': (
            band_width > band_width.rolling(100).quantile(0.8)
        ).astype(int)
    })
    
    # Band touch signals
    results['bb_touch_upper'] = (
        (prices >= upper_band) & (prices.shift(1) < upper_band.shift(1))
    ).astype(int)
    
    results['bb_touch_lower'] = (
        (prices <= lower_band) & (prices.shift(1) > lower_band.shift(1))
    ).astype(int)
    
    # Mean reversion signals
    results['bb_mean_reversion_long'] = (
        (prices < lower_band) & (prices > prices.shift(1))
    ).astype(int)
    
    results['bb_mean_reversion_short'] = (
        (prices > upper_band) & (prices < prices.shift(1))
    ).astype(int)
    
    return results


def calculate_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Keltner Channels.
    
    Similar to Bollinger Bands but uses ATR instead of standard deviation,
    making it less sensitive to extreme price movements.
    
    Args:
        df: OHLCV DataFrame
        ema_period: EMA period for middle line
        atr_period: ATR period
        atr_multiplier: ATR multiplier for bands
        
    Returns:
        DataFrame with Keltner Channel values
    """
    # Calculate middle line (EMA)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    middle = typical_price.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    
    # Calculate channels
    upper = middle + (atr_multiplier * atr)
    lower = middle - (atr_multiplier * atr)
    
    results = pd.DataFrame({
        'kc_middle': middle,
        'kc_upper': upper,
        'kc_lower': lower,
        'kc_width': (upper - lower) / middle * 100,
        'kc_position': (df['close'] - lower) / (upper - lower + 1e-10),
        'kc_above_upper': (df['close'] > upper).astype(int),
        'kc_below_lower': (df['close'] < lower).astype(int)
    })
    
    return results


def calculate_donchian_channels(
    df: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """
    Calculate Donchian Channels (Price Channels).
    
    Donchian Channels show the highest high and lowest low over a period.
    Used for breakout trading strategies.
    
    Args:
        df: OHLCV DataFrame
        period: Lookback period
        
    Returns:
        DataFrame with Donchian Channel values
    """
    upper = df['high'].rolling(period).max()
    lower = df['low'].rolling(period).min()
    middle = (upper + lower) / 2
    
    results = pd.DataFrame({
        'dc_upper': upper,
        'dc_lower': lower,
        'dc_middle': middle,
        'dc_width': (upper - lower) / lower * 100,
        'dc_position': (df['close'] - lower) / (upper - lower + 1e-10),
        
        # Breakout signals
        'dc_breakout_up': (df['high'] >= upper.shift(1)).astype(int),
        'dc_breakout_down': (df['low'] <= lower.shift(1)).astype(int)
    })
    
    return results


def calculate_volatility_metrics(
    df: pd.DataFrame,
    period: int = 20
) -> pd.DataFrame:
    """
    Calculate additional volatility metrics.
    
    Args:
        df: OHLCV DataFrame
        period: Calculation period
        
    Returns:
        DataFrame with various volatility metrics
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Parkinson volatility (uses high-low range)
    parkinson = np.sqrt(
        (1 / (4 * np.log(2))) * 
        ((np.log(high / low) ** 2).rolling(period).mean())
    )
    
    # Garman-Klass volatility (uses OHLC)
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / df['open']) ** 2
    gk = np.sqrt(
        (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(period).mean()
    )
    
    # Yang-Zhang volatility
    log_oc = np.log(df['open'] / close.shift(1))
    log_co = np.log(close / df['open'])
    
    overnight_var = log_oc.rolling(period).var()
    open_close_var = log_co.rolling(period).var()
    rs_var = (log_hl / (4 * np.log(2))).rolling(period).mean()
    
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    yz = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
    
    results = pd.DataFrame({
        'vol_parkinson': parkinson,
        'vol_garman_klass': gk,
        'vol_yang_zhang': yz,
        
        # Intraday volatility
        'vol_intraday': ((high - low) / close).rolling(period).mean(),
        
        # Close-to-close volatility
        'vol_close_to_close': close.pct_change().rolling(period).std()
    })
    
    return results


class VolatilityIndicatorGenerator:
    """
    Generates all volatility-related features.
    """
    
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
        """
        Initialize the generator.
        """
        self.atr_periods = atr_periods
        self.rolling_vol_windows = rolling_vol_windows
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.keltner_ema = keltner_ema
        self.keltner_atr = keltner_atr
        self.donchian_period = donchian_period
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        features = df.copy()
        
        # ATR
        atr_features = calculate_atr(df, self.atr_periods)
        features = pd.concat([features, atr_features], axis=1)
        
        # Rolling volatility
        vol_features = calculate_rolling_volatility(
            df['close'], self.rolling_vol_windows
        )
        features = pd.concat([features, vol_features], axis=1)
        
        # Bollinger Bands
        bb_features = calculate_bollinger_bands(
            df['close'], self.bollinger_window, self.bollinger_std
        )
        features = pd.concat([features, bb_features], axis=1)
        
        # Keltner Channels
        kc_features = calculate_keltner_channels(
            df, self.keltner_ema, self.keltner_atr
        )
        features = pd.concat([features, kc_features], axis=1)
        
        # Donchian Channels
        dc_features = calculate_donchian_channels(df, self.donchian_period)
        features = pd.concat([features, dc_features], axis=1)
        
        # Additional volatility metrics
        vol_metrics = calculate_volatility_metrics(df)
        features = pd.concat([features, vol_metrics], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} volatility features")
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be generated."""
        names = ['true_range']
        
        # ATR features
        for p in self.atr_periods:
            names.extend([
                f'atr_{p}', f'atr_{p}_pct', f'atr_{p}_ratio', f'atr_{p}_expanding'
            ])
        
        # Volatility features
        for w in self.rolling_vol_windows:
            names.extend([
                f'volatility_{w}', f'volatility_{w}_ann',
                f'volatility_{w}_percentile', f'volatility_{w}_zscore'
            ])
        names.append('volatility_regime')
        
        # Bollinger Band features
        names.extend([
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent_b',
            'bb_above_upper', 'bb_below_lower', 'bb_in_bands',
            'bb_dist_upper', 'bb_dist_lower', 'bb_squeeze', 'bb_expansion',
            'bb_touch_upper', 'bb_touch_lower',
            'bb_mean_reversion_long', 'bb_mean_reversion_short'
        ])
        
        # Keltner Channel features
        names.extend([
            'kc_middle', 'kc_upper', 'kc_lower', 'kc_width',
            'kc_position', 'kc_above_upper', 'kc_below_lower'
        ])
        
        # Donchian Channel features
        names.extend([
            'dc_upper', 'dc_lower', 'dc_middle', 'dc_width',
            'dc_position', 'dc_breakout_up', 'dc_breakout_down'
        ])
        
        # Additional volatility metrics
        names.extend([
            'vol_parkinson', 'vol_garman_klass', 'vol_yang_zhang',
            'vol_intraday', 'vol_close_to_close'
        ])
        
        return names
