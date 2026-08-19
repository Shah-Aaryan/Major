"""
Momentum Indicator Generation.

Calculates momentum-based indicators including:
- Relative Strength Index (RSI)
- Stochastic RSI
- Stochastic Oscillator
- MACD
- ROC
- CCI
- Williams %R
- TRIX
- Momentum (MOM)
- Ultimate Oscillator
- Percentage Price Oscillator (PPO)
- Awesome Oscillator (AO)
- Chande Momentum Oscillator (CMO)
- True Strength Index (TSI)
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
    """Calculate Relative Strength Index for multiple periods."""
    results = {}
    for period in periods:
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        results[f'rsi_{period}'] = rsi
        results[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
        results[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
        results[f'rsi_{period}_momentum'] = rsi.diff(5)
        results[f'rsi_{period}_from_neutral'] = rsi - 50
    
    if len(periods) > 0:
        main_rsi = results[f'rsi_{periods[0]}']
        price_trend = prices.diff(10) > 0
        rsi_trend = pd.Series(main_rsi).diff(10) > 0
        results['rsi_bearish_divergence'] = (price_trend & ~rsi_trend).astype(int)
        results['rsi_bullish_divergence'] = (~price_trend & rsi_trend).astype(int)
    
    return pd.DataFrame(results, index=prices.index)


def calculate_stoch_rsi(
    prices: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic RSI."""
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    avg_gains = gains.ewm(span=rsi_period, adjust=False).mean()
    avg_losses = losses.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100
    stoch_rsi_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(smooth_d).mean()
    
    results = pd.DataFrame({
        'stoch_rsi': stoch_rsi,
        'stoch_rsi_k': stoch_rsi_k,
        'stoch_rsi_d': stoch_rsi_d,
        'stoch_rsi_signal': (stoch_rsi_k > stoch_rsi_d).astype(int),
        'stoch_rsi_overbought': (stoch_rsi_k > 80).astype(int),
        'stoch_rsi_oversold': (stoch_rsi_k < 20).astype(int)
    }, index=prices.index)
    
    results['stoch_rsi_cross_up'] = ((stoch_rsi_k > stoch_rsi_d) & (stoch_rsi_k.shift(1) <= stoch_rsi_d.shift(1))).astype(int)
    results['stoch_rsi_cross_down'] = ((stoch_rsi_k < stoch_rsi_d) & (stoch_rsi_k.shift(1) >= stoch_rsi_d.shift(1))).astype(int)
    
    return results


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator (%K, %D)."""
    lowest_low = df['low'].rolling(k_period).min()
    highest_high = df['high'].rolling(k_period).max()
    stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(d_period).mean()
    return pd.DataFrame({
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'stoch_overbought': (stoch_k > 80).astype(int),
        'stoch_oversold': (stoch_k < 20).astype(int)
    }, index=df.index)


def calculate_trix(prices: pd.Series, period: int = 15) -> pd.Series:
    """Calculate TRIX (Triple Exponentially Smoothed Moving Average)."""
    ema1 = prices.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 100 * (ema3 - ema3.shift(1)) / (ema3.shift(1) + 1e-10)


def calculate_momentum(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Momentum (MOM)."""
    return prices - prices.shift(period)


def calculate_ultimate_oscillator(df: pd.DataFrame, p1: int = 7, p2: int = 14, p3: int = 28) -> pd.Series:
    """Calculate Ultimate Oscillator."""
    prior_close = df['close'].shift(1)
    bp = df['close'] - pd.concat([df['low'], prior_close], axis=1).min(axis=1)
    tr = pd.concat([df['high'], prior_close], axis=1).max(axis=1) - pd.concat([df['low'], prior_close], axis=1).min(axis=1)
    
    avg7 = bp.rolling(p1).sum() / (tr.rolling(p1).sum() + 1e-10)
    avg14 = bp.rolling(p2).sum() / (tr.rolling(p2).sum() + 1e-10)
    avg28 = bp.rolling(p3).sum() / (tr.rolling(p3).sum() + 1e-10)
    
    return 100 * (4 * avg7 + 2 * avg14 + avg28) / 7.0


def calculate_ppo(prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Calculate Percentage Price Oscillator (PPO)."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    ppo = (ema_fast - ema_slow) / (ema_slow + 1e-10) * 100
    ppo_signal = ppo.ewm(span=signal_period, adjust=False).mean()
    ppo_hist = ppo - ppo_signal
    return pd.DataFrame({
        'ppo': ppo,
        'ppo_signal': ppo_signal,
        'ppo_hist': ppo_hist
    }, index=prices.index)


def calculate_awesome_oscillator(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculate Awesome Oscillator (AO)."""
    median_price = (df['high'] + df['low']) / 2.0
    ao = median_price.rolling(fast_period).mean() - median_price.rolling(slow_period).mean()
    return ao


def calculate_cmo(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Chande Momentum Oscillator (CMO)."""
    delta = prices.diff()
    gains = delta.clip(lower=0).rolling(period).sum()
    losses = (-delta.clip(upper=0)).rolling(period).sum()
    return 100 * (gains - losses) / (gains + losses + 1e-10)


def calculate_tsi(prices: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """Calculate True Strength Index (TSI)."""
    pc = prices.diff()
    double_smoothed_pc = pc.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    double_smoothed_abs_pc = pc.abs().ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
    return 100 * (double_smoothed_pc / (double_smoothed_abs_pc + 1e-10))


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """Calculate MACD."""
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    results = pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
        'macd_normalized': macd_line / (prices + 1e-10) * 100,
        'macd_positive': (macd_line > 0).astype(int),
        'macd_above_signal': (macd_line > signal_line).astype(int),
        'macd_hist_momentum': histogram.diff(),
        'macd_hist_growing': (histogram.diff() > 0).astype(int)
    }, index=prices.index)
    
    results['macd_cross_up'] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
    results['macd_cross_down'] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
    results['macd_zero_cross_up'] = ((macd_line > 0) & (macd_line.shift(1) <= 0)).astype(int)
    results['macd_zero_cross_down'] = ((macd_line < 0) & (macd_line.shift(1) >= 0)).astype(int)
    
    return results


def calculate_roc(
    prices: pd.Series,
    periods: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """Calculate Rate of Change (ROC)."""
    results = {}
    for period in periods:
        roc = ((prices - prices.shift(period)) / (prices.shift(period) + 1e-10)) * 100
        results[f'roc_{period}'] = roc
        results[f'roc_{period}_momentum'] = roc.diff()
        roc_mean = roc.rolling(50).mean()
        roc_std = roc.rolling(50).std()
        results[f'roc_{period}_zscore'] = (roc - roc_mean) / (roc_std + 1e-10)
    return pd.DataFrame(results, index=prices.index)


def calculate_cci(
    df: pd.DataFrame,
    period: int = 20,
    constant: float = 0.015
) -> pd.DataFrame:
    """Calculate Commodity Channel Index (CCI)."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - tp_sma) / (constant * mean_dev + 1e-10)
    
    return pd.DataFrame({
        'cci': cci,
        'cci_overbought': (cci > 100).astype(int),
        'cci_oversold': (cci < -100).astype(int),
        'cci_extreme': (abs(cci) > 200).astype(int)
    }, index=df.index)


def calculate_williams_r(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """Calculate Williams %R."""
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    williams_r = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
    
    return pd.DataFrame({
        'williams_r': williams_r,
        'williams_r_overbought': (williams_r > -20).astype(int),
        'williams_r_oversold': (williams_r < -80).astype(int)
    }, index=df.index)


class MomentumIndicatorGenerator:
    """Generates all momentum-related features."""
    
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
        features = df.copy()
        
        features = pd.concat([features, calculate_rsi(df['close'], self.rsi_periods)], axis=1)
        features = pd.concat([features, calculate_stoch_rsi(df['close'], self.stoch_rsi_period, self.stoch_rsi_smooth, self.stoch_rsi_smooth)], axis=1)
        features = pd.concat([features, calculate_stochastic(df)], axis=1)
        features = pd.concat([features, calculate_macd(df['close'], self.macd_fast, self.macd_slow, self.macd_signal)], axis=1)
        features = pd.concat([features, calculate_roc(df['close'], self.roc_periods)], axis=1)
        features = pd.concat([features, calculate_cci(df, self.cci_period)], axis=1)
        features = pd.concat([features, calculate_williams_r(df, self.williams_period)], axis=1)
        
        # New momentum indicators
        features['trix_15'] = calculate_trix(df['close'])
        features['momentum_14'] = calculate_momentum(df['close'])
        features['ultimate_oscillator'] = calculate_ultimate_oscillator(df)
        ppo = calculate_ppo(df['close'])
        features = pd.concat([features, ppo], axis=1)
        features['awesome_oscillator'] = calculate_awesome_oscillator(df)
        features['cmo_14'] = calculate_cmo(df['close'])
        features['tsi_25_13'] = calculate_tsi(df['close'])
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} momentum features")
        return features
