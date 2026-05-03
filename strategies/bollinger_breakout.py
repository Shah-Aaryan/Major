"""
Bollinger Bands Breakout Strategy.

A volatility-based strategy that trades breakouts from Bollinger Bands.
The strategy enters positions when price breaks out of the bands after
a period of consolidation (squeeze).

STRATEGY LOGIC (FIXED - cannot be changed by ML):
- Detect Bollinger Band squeeze (low volatility, bands narrow)
- Buy when price breaks above upper band after squeeze
- Sell when price breaks below lower band after squeeze
- Use band width and volume as confirmation
- Apply mean reversion exit when price returns inside bands

PARAMETERS (TUNABLE - can be adjusted by ML):
- Bollinger Band period
- Standard deviation multiplier
- Squeeze detection threshold
- Risk management parameters
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import logging

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    SignalType,
    StrategyParameters
)

logger = logging.getLogger(__name__)


@dataclass
class BollingerBreakoutParams(StrategyParameters):
    """
    Parameters for Bollinger Bands Breakout Strategy.
    
    These parameters can be tuned by ML optimization.
    """
    # Bollinger Band parameters
    bb_period: int = 20              # BB calculation period
    bb_std_dev: float = 2.0          # Standard deviation multiplier
    
    # Squeeze detection
    squeeze_lookback: int = 100       # Lookback for squeeze comparison
    squeeze_percentile: float = 20.0  # Band width percentile for squeeze
    require_squeeze: bool = False      # Allow breakout without prior squeeze
    min_squeeze_candles: int = 3      # Lowered from 5
    
    # Breakout confirmation
    breakout_threshold_pct: float = 0.1  # Min % beyond band for breakout
    volume_confirmation: bool = True     # Require volume spike
    volume_spike_mult: float = 1.2       # Lowered from 1.5
    
    # Mean reversion exit
    use_mean_reversion_exit: bool = True  # Exit when price returns to middle band
    
    # Sentiment-aware biasing (Requested by user)
    use_sentiment_bias: bool = True      # Bias signals towards major trend
    sentiment_ema_period: int = 200      # Long-term trend EMA
    sentiment_confidence_boost: float = 0.2  # Add confidence to trend-aligned signals
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds including strategy-specific ones."""
        bounds = super().get_bounds()
        bounds.update({
            'bb_period': (20, 60),           # Increased minimum to reduce noise
            'bb_std_dev': (2.0, 3.5),        # More conservative multiplier
            'squeeze_lookback': (100, 300),
            'squeeze_percentile': (10.0, 30.0),
            'min_squeeze_candles': (2, 10),
            'breakout_threshold_pct': (0.1, 0.8),
            'volume_spike_mult': (1.1, 2.5)
        })
        return bounds


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Trading Strategy.
    
    This strategy exploits volatility expansions by entering positions
    when price breaks out of Bollinger Bands after a consolidation period.
    
    Trading Logic:
    1. Detect Bollinger Band squeeze (bands narrow)
    2. Wait for breakout above upper band (bullish) or below lower band (bearish)
    3. Confirm with volume spike
    4. Enter position in breakout direction
    5. Exit when price returns to middle band or hits stop/take profit
    
    Best market conditions:
    - After periods of consolidation
    - Pre-news volatility expansion
    - Range breakouts
    
    Avoid:
    - Fake breakouts in choppy markets
    - Low volume breakouts
    - Extended overbought/oversold conditions
    """
    
    def __init__(
        self,
        parameters: Optional[BollingerBreakoutParams] = None,
        initial_capital: float = 100000.0,
        trading_fee_pct: float = 0.001
    ):
        """
        Initialize the Bollinger Breakout Strategy.
        
        Args:
            parameters: Strategy parameters (uses defaults if None)
            initial_capital: Starting capital
            trading_fee_pct: Trading fee percentage
        """
        super().__init__(
            name="Bollinger_Breakout",
            parameters=parameters or BollingerBreakoutParams(),
            initial_capital=initial_capital,
            trading_fee_pct=trading_fee_pct
        )
        
        # State tracking
        self.squeeze_count = 0
        self.was_in_squeeze = False
        self._indicators_initialized = False

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters and reset indicator initialization."""
        super().update_parameters(params)
        self._indicators_initialized = False

    def _ensure_indicators(self, df: pd.DataFrame) -> None:
        """Calculate required indicators if they are missing from the dataframe."""
        # Bollinger Bands
        period = self.params.bb_period
        std_dev = self.params.bb_std_dev
        
        prefix = f'bb_' # Usually bb_upper, bb_lower, etc.
        if f'bb_upper' not in df.columns:
            ma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df['bb_upper'] = ma + (std * std_dev)
            df['bb_lower'] = ma - (std * std_dev)
            df['bb_middle'] = ma
            
        # Band width (for squeeze)
        if 'bb_width' not in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
        self._indicators_initialized = True
    
    @property
    def params(self) -> BollingerBreakoutParams:
        """Type-safe access to parameters."""
        return self.parameters
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """
        Generate trading signal based on Bollinger Bands breakout logic.
        
        FIXED LOGIC - This defines the strategy behavior.
        Only the parameters can be changed.
        
        Args:
            df: DataFrame with OHLCV and features
            current_idx: Current index in the DataFrame
            
        Returns:
            StrategySignal indicating the trading action
        """
        if not self._indicators_initialized:
            self._ensure_indicators(df)
            
        current_row = df.iloc[current_idx]
        current_time = df.index[current_idx]
        current_price = current_row['close']
        
        # Get Bollinger Band values
        bb_upper = current_row.get('bb_upper', None)
        bb_lower = current_row.get('bb_lower', None)
        bb_middle = current_row.get('bb_middle', None)
        bb_width = current_row.get('bb_width', None)
        
        # Check if BB features exist
        if bb_upper is None or pd.isna(bb_upper):
            return self._hold_signal(current_time, current_price, "BB features not available")
        
        # Get volume data
        volume = current_row.get('volume', 0)
        volume_ma = current_row.get('volume_ma_20', volume)
        if pd.isna(volume_ma):
            volume_ma = volume
        
        # === SQUEEZE DETECTION ===
        is_in_squeeze = self._detect_squeeze(df, current_idx, bb_width)
        
        # Track squeeze duration
        if is_in_squeeze:
            self.squeeze_count += 1
        else:
            if self.was_in_squeeze and self.squeeze_count >= self.params.min_squeeze_candles:
                # Just exited a valid squeeze - look for breakout
                pass
            self.squeeze_count = 0
        
        previous_squeeze = self.was_in_squeeze
        self.was_in_squeeze = is_in_squeeze
        
        # === SENTIMENT BIASING (Requested: Bearish -> Sell) ===
        sentiment_is_bullish = True
        if self.params.use_sentiment_bias:
            # We need to calculate EMA 200 for sentiment
            # EMA 200 is available in feature_engine
            sentiment_ema = current_row.get(f'ema_{self.params.sentiment_ema_period}', None)
            if sentiment_ema is not None and not pd.isna(sentiment_ema):
                sentiment_is_bullish = current_price > sentiment_ema
        
        # === CORE STRATEGY LOGIC (FIXED) ===
        
        # Check for breakout conditions
        breakout_upper = current_price > bb_upper * (1 + self.params.breakout_threshold_pct / 100)
        breakout_lower = current_price < bb_lower * (1 - self.params.breakout_threshold_pct / 100)
        
        # Apply squeeze requirement
        if self.params.require_squeeze:
            # Need to have been in squeeze recently
            if not (previous_squeeze or self.squeeze_count >= self.params.min_squeeze_candles):
                if breakout_upper or breakout_lower:
                    return self._hold_signal(
                        current_time, current_price,
                        f"Breakout without prior squeeze (squeeze_count={self.squeeze_count})"
                    )
        
        # Apply volume confirmation
        volume_confirmed = True
        if self.params.volume_confirmation:
            volume_confirmed = volume > volume_ma * self.params.volume_spike_mult
        
        # Apply Sentiment Bias: Prioritize signals that match the major trend
        if self.params.use_sentiment_bias:
            if not sentiment_is_bullish and breakout_upper:
                return self._hold_signal(current_time, current_price, "Bullish breakout against Bearish major trend")
            if sentiment_is_bullish and breakout_lower:
                return self._hold_signal(current_time, current_price, "Bearish breakout against Bullish major trend")
        
        # Generate signals
        if breakout_upper and volume_confirmed:
            signal = self._generate_entry_signal(
                current_time,
                current_price,
                SignalType.LONG,
                bb_upper,
                bb_lower,
                bb_middle,
                bb_width,
                volume / volume_ma if volume_ma > 0 else 1.0
            )
            if sentiment_is_bullish:
                signal.confidence = min(signal.confidence + self.params.sentiment_confidence_boost, 1.0)
            return signal
        
        if breakout_lower and volume_confirmed:
            signal = self._generate_entry_signal(
                current_time,
                current_price,
                SignalType.SHORT,
                bb_upper,
                bb_lower,
                bb_middle,
                bb_width,
                volume / volume_ma if volume_ma > 0 else 1.0
            )
            if not sentiment_is_bullish:
                signal.confidence = min(signal.confidence + self.params.sentiment_confidence_boost, 1.0)
            return signal
        
        # Check for volume failure
        if (breakout_upper or breakout_lower) and not volume_confirmed:
            return self._hold_signal(
                current_time, current_price,
                f"Breakout without volume confirmation (vol_ratio={(volume/volume_ma):.2f})"
            )
        
        # No breakout
        position = "in squeeze" if is_in_squeeze else "normal"
        return self._hold_signal(
            current_time, current_price,
            f"No breakout, price within bands ({position})"
        )
    
    def _detect_squeeze(
        self,
        df: pd.DataFrame,
        current_idx: int,
        current_bb_width: float
    ) -> bool:
        """
        Detect if we're in a Bollinger Band squeeze.
        
        A squeeze occurs when the band width is in the lowest percentile
        of its historical range, indicating low volatility.
        """
        if current_bb_width is None or pd.isna(current_bb_width):
            return False
        
        # Get historical band widths
        lookback_start = max(0, current_idx - self.params.squeeze_lookback)
        
        if 'bb_width' in df.columns:
            historical_widths = df['bb_width'].iloc[lookback_start:current_idx].dropna()
        else:
            return False
        
        if len(historical_widths) < 20:
            return False
        
        # Calculate percentile threshold
        threshold = np.percentile(historical_widths, self.params.squeeze_percentile)
        
        return current_bb_width < threshold
    
    def _generate_entry_signal(
        self,
        timestamp,
        price: float,
        signal_type: SignalType,
        bb_upper: float,
        bb_lower: float,
        bb_middle: float,
        bb_width: float,
        volume_ratio: float
    ) -> StrategySignal:
        """Generate entry signal."""
        
        # Calculate confidence based on breakout strength and volume
        breakout_distance = 0
        if signal_type == SignalType.LONG:
            breakout_distance = (price - bb_upper) / bb_upper
        else:
            breakout_distance = (bb_lower - price) / bb_lower
        
        # Base confidence on breakout distance and volume
        confidence = min(0.5 + breakout_distance * 10 + (volume_ratio - 1) * 0.2, 1.0)
        confidence = max(confidence, 0.3)  # Minimum confidence
        
        # Build reason string
        direction = "LONG (upper band breakout)" if signal_type == SignalType.LONG else "SHORT (lower band breakout)"
        reason = (
            f"BB breakout: {direction}, "
            f"width={bb_width:.2f}%, vol_ratio={volume_ratio:.2f}"
        )
        
        return StrategySignal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            reason=reason,
            metadata={
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_width': bb_width,
                'volume_ratio': volume_ratio,
                'squeeze_count': self.squeeze_count,
                'breakout_distance_pct': breakout_distance * 100
            }
        )
    
    def _hold_signal(
        self,
        timestamp,
        price: float,
        reason: str
    ) -> StrategySignal:
        """Generate a HOLD signal."""
        return StrategySignal(
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            price=price,
            confidence=0.0,
            reason=reason
        )
    
    def get_strategy_specific_params(self) -> Dict[str, Any]:
        """Get Bollinger-specific parameters."""
        return {
            'bb_period': self.params.bb_period,
            'bb_std_dev': self.params.bb_std_dev,
            'squeeze_lookback': self.params.squeeze_lookback,
            'squeeze_percentile': self.params.squeeze_percentile,
            'require_squeeze': self.params.require_squeeze,
            'min_squeeze_candles': self.params.min_squeeze_candles,
            'breakout_threshold_pct': self.params.breakout_threshold_pct,
            'volume_confirmation': self.params.volume_confirmation,
            'volume_spike_mult': self.params.volume_spike_mult,
            'use_mean_reversion_exit': self.params.use_mean_reversion_exit,
            'use_sentiment_bias': self.params.use_sentiment_bias,
            'sentiment_ema_period': self.params.sentiment_ema_period,
            'sentiment_confidence_boost': self.params.sentiment_confidence_boost
        }
    
    def set_strategy_specific_params(self, params: Dict[str, Any]) -> None:
        """Set Bollinger-specific parameters."""
        if 'bb_period' in params:
            self.params.bb_period = int(params['bb_period'])
        if 'bb_std_dev' in params:
            self.params.bb_std_dev = float(params['bb_std_dev'])
        if 'squeeze_lookback' in params:
            self.params.squeeze_lookback = int(params['squeeze_lookback'])
        if 'squeeze_percentile' in params:
            self.params.squeeze_percentile = float(params['squeeze_percentile'])
        if 'require_squeeze' in params:
            self.params.require_squeeze = bool(params['require_squeeze'])
        if 'min_squeeze_candles' in params:
            self.params.min_squeeze_candles = int(params['min_squeeze_candles'])
        if 'breakout_threshold_pct' in params:
            self.params.breakout_threshold_pct = float(params['breakout_threshold_pct'])
        if 'volume_confirmation' in params:
            self.params.volume_confirmation = bool(params['volume_confirmation'])
        if 'volume_spike_mult' in params:
            self.params.volume_spike_mult = float(params['volume_spike_mult'])
        if 'use_mean_reversion_exit' in params:
            self.params.use_mean_reversion_exit = bool(params['use_mean_reversion_exit'])
        if 'use_sentiment_bias' in params:
            self.params.use_sentiment_bias = bool(params['use_sentiment_bias'])
        if 'sentiment_ema_period' in params:
            self.params.sentiment_ema_period = int(params['sentiment_ema_period'])
        if 'sentiment_confidence_boost' in params:
            self.params.sentiment_confidence_boost = float(params['sentiment_confidence_boost'])
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for all parameters."""
        return self.params.get_bounds()
    
    def get_optimal_market_conditions(self) -> Dict[str, Any]:
        """Return the market conditions where this strategy performs best."""
        return {
            'trend_regime': ['ranging', 'pre_breakout', 'consolidation'],
            'volatility_regime': ['low_expanding_to_high'],
            'best_patterns': ['squeeze', 'consolidation', 'triangle'],
            'best_timeframes': ['15m', '1h'],
            'avoid_conditions': [
                'fake_breakouts',
                'extended_trends',
                'low_volume',
                'major_news_events'
            ]
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.squeeze_count = 0
        self.was_in_squeeze = False
    
    def explain_signal(self, signal: StrategySignal) -> str:
        """Generate a human-readable explanation of the signal."""
        if signal.signal_type == SignalType.HOLD:
            return f"No trade: {signal.reason}"
        
        bb_upper = signal.metadata.get('bb_upper', 'N/A')
        bb_lower = signal.metadata.get('bb_lower', 'N/A')
        bb_width = signal.metadata.get('bb_width', 'N/A')
        volume_ratio = signal.metadata.get('volume_ratio', 'N/A')
        squeeze_count = signal.metadata.get('squeeze_count', 'N/A')
        
        if signal.signal_type == SignalType.LONG:
            explanation = (
                f"BUY SIGNAL generated:\n"
                f"- Price broke above upper Bollinger Band ({bb_upper:.2f})\n"
                f"- Prior squeeze period: {squeeze_count} candles\n"
                f"- Band width (volatility): {bb_width:.2f}%\n"
                f"- Volume ratio: {volume_ratio:.2f}x average\n"
                f"- Volatility expansion expected (breakout)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        else:
            explanation = (
                f"SELL/SHORT SIGNAL generated:\n"
                f"- Price broke below lower Bollinger Band ({bb_lower:.2f})\n"
                f"- Prior squeeze period: {squeeze_count} candles\n"
                f"- Band width (volatility): {bb_width:.2f}%\n"
                f"- Volume ratio: {volume_ratio:.2f}x average\n"
                f"- Volatility expansion expected (breakout)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        
        return explanation
