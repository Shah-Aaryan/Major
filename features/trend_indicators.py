"""
Trend Indicator Generation.

Calculates trend-following indicators including:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)
- Hull Moving Average (HMA)
- Double Exponential Moving Average (DEMA)
- Triple Exponential Moving Average (TEMA)
- Kaufman Adaptive Moving Average (KAMA)
- Volume Weighted Moving Average (VWMA)
- EMA Slope
- Average Directional Index (ADX)
- Aroon Indicator
- Parabolic SAR
- Ichimoku Cloud
- Vortex Indicator
- Trend lines (Linear Regression)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_sma(
    prices: pd.Series,
    windows: List[int] = [5, 10, 20, 50, 100, 200]
) -> pd.DataFrame:
    """Calculate Simple Moving Averages for multiple windows."""
    results = {}
    for window in windows:
        sma = prices.rolling(window=window).mean()
        results[f'sma_{window}'] = sma
        results[f'price_sma_{window}_pct'] = (prices - sma) / (sma + 1e-10)
        results[f'sma_{window}_slope'] = sma.diff(5) / (sma.shift(5) + 1e-10)
    
    if 20 in windows and 50 in windows:
        results['sma_20_50_cross'] = (results['sma_20'] > results['sma_50']).astype(int)
    if 50 in windows and 200 in windows:
        results['sma_50_200_cross'] = (results['sma_50'] > results['sma_200']).astype(int)
    
    return pd.DataFrame(results, index=prices.index)


def calculate_ema(
    prices: pd.Series,
    windows: List[int] = [5, 10, 20, 50, 100]
) -> pd.DataFrame:
    """Calculate Exponential Moving Averages for multiple windows."""
    results = {}
    for window in windows:
        ema = prices.ewm(span=window, adjust=False).mean()
        results[f'ema_{window}'] = ema
        results[f'price_ema_{window}_pct'] = (prices - ema) / (ema + 1e-10)
        ema_diff = ema.diff()
        results[f'ema_{window}_accel'] = ema_diff.diff()
    
    if 9 in windows and 21 in windows:
        results['ema_9_21_cross'] = (results['ema_9'] > results['ema_21']).astype(int)
    if 12 in windows and 26 in windows:
        results['ema_12_26_cross'] = (results['ema_12'] > results['ema_26']).astype(int)
    
    return pd.DataFrame(results, index=prices.index)


def calculate_wma(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, window + 1)
    return prices.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def calculate_hma(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Hull Moving Average."""
    wma_half = calculate_wma(prices, max(1, window // 2))
    wma_full = calculate_wma(prices, window)
    diff = 2 * wma_half - wma_full
    sqrt_window = max(1, int(np.sqrt(window)))
    return calculate_wma(diff, sqrt_window)


def calculate_dema(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Double Exponential Moving Average."""
    ema1 = prices.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    return 2 * ema1 - ema2


def calculate_tema(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Triple Exponential Moving Average."""
    ema1 = prices.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


def calculate_kama(prices: pd.Series, window: int = 10, pow1: int = 2, pow2: int = 30) -> pd.Series:
    """Calculate Kaufman Adaptive Moving Average."""
    change = (prices - prices.shift(window)).abs()
    volatility = prices.diff().abs().rolling(window).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2
    kama = pd.Series(index=prices.index, dtype=float)
    if len(prices) > 0:
        kama.iloc[0] = prices.iloc[0]
        for i in range(1, len(prices)):
            if np.isnan(kama.iloc[i-1]):
                kama.iloc[i] = prices.iloc[i]
            else:
                c_sc = sc.iloc[i] if not np.isnan(sc.iloc[i]) else 0.1
                kama.iloc[i] = kama.iloc[i-1] + c_sc * (prices.iloc[i] - kama.iloc[i-1])
    return kama


def calculate_vwma(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Volume Weighted Moving Average."""
    pv = df['close'] * df['volume']
    return pv.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-10)


def calculate_aroon(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Aroon Indicator (Aroon Up, Aroon Down, Aroon Oscillator)."""
    aroon_up = df['high'].rolling(window + 1).apply(lambda x: float(np.argmax(x)) / window * 100, raw=True)
    aroon_down = df['low'].rolling(window + 1).apply(lambda x: float(np.argmin(x)) / window * 100, raw=True)
    aroon_osc = aroon_up - aroon_down
    return pd.DataFrame({
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'aroon_oscillator': aroon_osc
    }, index=df.index)


def calculate_parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return pd.Series(sar, index=df.index)
    
    is_long = True
    af = af_start
    ep = high[0]
    sar[0] = low[0]
    
    for i in range(1, n):
        prior_sar = sar[i-1]
        if is_long:
            current_sar = prior_sar + af * (ep - prior_sar)
            current_sar = min(current_sar, low[i-1], low[max(0, i-2)])
            if low[i] < current_sar:
                is_long = False
                current_sar = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            current_sar = prior_sar + af * (ep - prior_sar)
            current_sar = max(current_sar, high[i-1], high[max(0, i-2)])
            if high[i] > current_sar:
                is_long = True
                current_sar = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar[i] = current_sar
    return pd.Series(sar, index=df.index)


def calculate_ichimoku(df: pd.DataFrame, conversion_periods: int = 9, base_periods: int = 26, lagging_span_2_periods: int = 52, displacement: int = 26) -> pd.DataFrame:
    """Calculate Ichimoku Kinko Hyo (Cloud)."""
    high = df['high']
    low = df['low']
    tenkan_sen = (high.rolling(conversion_periods).max() + low.rolling(conversion_periods).min()) / 2
    kijun_sen = (high.rolling(base_periods).max() + low.rolling(base_periods).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    senkou_span_b = ((high.rolling(lagging_span_2_periods).max() + low.rolling(lagging_span_2_periods).min()) / 2).shift(displacement)
    chikou_span = df['close'].shift(-displacement)
    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }, index=df.index)


def calculate_vortex(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Vortex Indicator (VI+, VI-)."""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr_sum = tr.rolling(period).sum() + 1e-10
    vi_plus = vm_plus.rolling(period).sum() / tr_sum
    vi_minus = vm_minus.rolling(period).sum() / tr_sum
    return pd.DataFrame({
        'vortex_pos': vi_plus,
        'vortex_neg': vi_minus,
        'vortex_diff': vi_plus - vi_minus
    }, index=df.index)


def calculate_ema_slope(
    prices: pd.Series,
    ema_window: int = 20,
    slope_period: int = 5
) -> pd.DataFrame:
    """Calculate EMA and its slope for trend direction analysis."""
    ema = prices.ewm(span=ema_window, adjust=False).mean()
    slope = ema.diff(slope_period)
    slope_pct = slope / (ema.shift(slope_period) + 1e-10)
    slope_threshold = 0.001
    direction = np.where(slope_pct > slope_threshold, 1, np.where(slope_pct < -slope_threshold, -1, 0))
    momentum = slope.diff()
    return pd.DataFrame({
        'ema': ema,
        'ema_slope': slope,
        'ema_slope_pct': slope_pct,
        'ema_slope_direction': direction,
        'ema_slope_momentum': momentum
    }, index=prices.index)


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) for trend strength."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    results = pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'dx': dx
    }, index=df.index)
    
    results['trend_strength'] = pd.cut(
        results['adx'],
        bins=[-1, 20, 40, 60, 100],
        labels=['weak', 'moderate', 'strong', 'very_strong']
    )
    results['trend_direction'] = np.where(results['plus_di'] > results['minus_di'], 1, -1)
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
    """Calculate linear regression trend lines."""
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
    
    results['trend_line_value'] = results['trend_intercept'] + results['trend_slope'] * (lookback - 1)
    results['dist_from_trend'] = (df['close'] - results['trend_line_value']) / (results['trend_line_value'] + 1e-10)
    results['trend_slope_normalized'] = results['trend_slope'] / (df['close'] + 1e-10)
    
    return results


class TrendIndicatorGenerator:
    """Generates all trend-related features."""
    
    def __init__(
        self,
        sma_windows: List[int] = [5, 10, 20, 50, 100, 200],
        ema_windows: List[int] = [5, 10, 20, 50, 100],
        ema_slope_period: int = 5,
        adx_period: int = 14,
        trend_lookback: int = 20
    ):
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.ema_slope_period = ema_slope_period
        self.adx_period = adx_period
        self.trend_lookback = trend_lookback
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        sma_features = calculate_sma(df['close'], self.sma_windows)
        features = pd.concat([features, sma_features], axis=1)
        
        ema_features = calculate_ema(df['close'], self.ema_windows)
        features = pd.concat([features, ema_features], axis=1)
        
        # New moving average indicators
        features['wma_14'] = calculate_wma(df['close'], 14)
        features['hma_14'] = calculate_hma(df['close'], 14)
        features['dema_14'] = calculate_dema(df['close'], 14)
        features['tema_14'] = calculate_tema(df['close'], 14)
        features['kama_10'] = calculate_kama(df['close'], 10)
        features['vwma_14'] = calculate_vwma(df, 14)
        
        # Aroon, PSAR, Ichimoku, Vortex
        aroon = calculate_aroon(df, 14)
        psar = calculate_parabolic_sar(df)
        ichimoku = calculate_ichimoku(df)
        vortex = calculate_vortex(df)
        
        features = pd.concat([features, aroon, ichimoku, vortex], axis=1)
        features['parabolic_sar'] = psar
        
        ema_slope = calculate_ema_slope(df['close'], ema_window=20, slope_period=self.ema_slope_period)
        ema_slope.columns = ['ema_20_for_slope', 'ema_20_slope', 'ema_20_slope_pct', 'ema_20_slope_direction', 'ema_20_slope_momentum']
        features = pd.concat([features, ema_slope], axis=1)
        
        adx_features = calculate_adx(df, self.adx_period)
        features = pd.concat([features, adx_features], axis=1)
        
        trend_features = calculate_trend_lines(df, self.trend_lookback)
        features = pd.concat([features, trend_features], axis=1)
        
        logger.info(f"Generated {len(features.columns) - len(df.columns)} trend features")
        return features
