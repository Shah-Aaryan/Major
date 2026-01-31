"""
EMA Crossover Strategy.

A trend-following strategy based on Exponential Moving Average crossovers.
The strategy enters positions when a faster EMA crosses a slower EMA.

STRATEGY LOGIC (FIXED - cannot be changed by ML):
- Buy when fast EMA crosses above slow EMA (bullish crossover)
- Sell when fast EMA crosses below slow EMA (bearish crossover)
- Use MACD as confirmation
- Apply trend strength filters (ADX)
- Manage position with trailing stops

PARAMETERS (TUNABLE - can be adjusted by ML):
- Fast EMA period
- Slow EMA period
- Confirmation requirements
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
class EMACrossoverParams(StrategyParameters):
    """
    Parameters for EMA Crossover Strategy.
    
    These parameters can be tuned by ML optimization.
    """
    # EMA-specific parameters
    ema_fast_period: int = 9         # Fast EMA period
    ema_slow_period: int = 21        # Slow EMA period
    
    # Trend confirmation
    use_macd_confirmation: bool = True   # Use MACD to confirm crossover
    use_adx_filter: bool = True          # Only trade in trending markets
    min_adx: float = 20.0                # Minimum ADX for trend strength
    
    # Crossover confirmation
    crossover_threshold_pct: float = 0.1  # Min distance between EMAs (as % of price)
    require_price_above_ema: bool = True  # Price must be above/below EMA for signal
    
    # Additional filters
    use_ema_slope_filter: bool = True    # Check if EMAs are sloping in signal direction
    min_ema_slope: float = 0.0001        # Minimum EMA slope
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds including strategy-specific ones."""
        bounds = super().get_bounds()
        bounds.update({
            'ema_fast_period': (3, 20),
            'ema_slow_period': (15, 100),
            'min_adx': (15.0, 40.0),
            'crossover_threshold_pct': (0.01, 0.5),
            'min_ema_slope': (0.00001, 0.001)
        })
        return bounds


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Trading Strategy.
    
    This strategy follows trends by entering positions when moving
    averages cross, indicating a potential trend change or continuation.
    
    Trading Logic:
    1. Calculate fast and slow EMAs
    2. Generate BUY signal when fast EMA crosses above slow EMA
    3. Generate SELL signal when fast EMA crosses below slow EMA
    4. Confirm with MACD and ADX filters
    5. Use trailing stops to ride the trend
    
    Best market conditions:
    - Trending markets (ADX > 20)
    - Clear directional moves
    - Moderate volatility
    
    Avoid:
    - Sideways/ranging markets (whipsaws)
    - Very high volatility (false signals)
    - Low liquidity periods
    """
    
    def __init__(
        self,
        parameters: Optional[EMACrossoverParams] = None,
        initial_capital: float = 100000.0,
        trading_fee_pct: float = 0.001
    ):
        """
        Initialize the EMA Crossover Strategy.
        
        Args:
            parameters: Strategy parameters (uses defaults if None)
            initial_capital: Starting capital
            trading_fee_pct: Trading fee percentage
        """
        super().__init__(
            name="EMA_Crossover",
            parameters=parameters or EMACrossoverParams(),
            initial_capital=initial_capital,
            trading_fee_pct=trading_fee_pct
        )
        
        # Track previous crossover state
        self.was_fast_above_slow: Optional[bool] = None
    
    @property
    def params(self) -> EMACrossoverParams:
        """Type-safe access to parameters."""
        return self.parameters
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """
        Generate trading signal based on EMA crossover logic.
        
        FIXED LOGIC - This defines the strategy behavior.
        Only the parameters (periods, thresholds) can be changed.
        
        Args:
            df: DataFrame with OHLCV and features
            current_idx: Current index in the DataFrame
            
        Returns:
            StrategySignal indicating the trading action
        """
        current_row = df.iloc[current_idx]
        current_time = df.index[current_idx]
        current_price = current_row['close']
        
        # Get EMA values
        fast_ema_col = f'ema_{self.params.ema_fast_period}'
        slow_ema_col = f'ema_{self.params.ema_slow_period}'
        
        # Find available EMA columns
        fast_ema = self._get_ema_value(df, current_idx, self.params.ema_fast_period)
        slow_ema = self._get_ema_value(df, current_idx, self.params.ema_slow_period)
        
        if fast_ema is None or slow_ema is None:
            return self._hold_signal(current_time, current_price, "EMA not available")
        
        # Check for NaN
        if pd.isna(fast_ema) or pd.isna(slow_ema):
            return self._hold_signal(current_time, current_price, "EMA is NaN")
        
        # Calculate EMA relationship
        is_fast_above_slow = fast_ema > slow_ema
        ema_diff_pct = (fast_ema - slow_ema) / current_price * 100
        
        # Calculate EMA slopes
        fast_ema_slope = 0
        slow_ema_slope = 0
        if current_idx >= 5:
            prev_fast = self._get_ema_value(df, current_idx - 5, self.params.ema_fast_period)
            prev_slow = self._get_ema_value(df, current_idx - 5, self.params.ema_slow_period)
            if prev_fast is not None and prev_slow is not None:
                fast_ema_slope = (fast_ema - prev_fast) / prev_fast
                slow_ema_slope = (slow_ema - prev_slow) / prev_slow
        
        # Apply ADX filter (only trade in trending markets)
        if self.params.use_adx_filter:
            adx = current_row.get('adx', self.params.min_adx)
            if not pd.isna(adx) and adx < self.params.min_adx:
                self.was_fast_above_slow = is_fast_above_slow
                return self._hold_signal(
                    current_time, current_price,
                    f"Weak trend (ADX={adx:.1f} < {self.params.min_adx})"
                )
        
        # === CORE STRATEGY LOGIC (FIXED) ===
        
        # Detect crossover
        crossover_bullish = False
        crossover_bearish = False
        
        if self.was_fast_above_slow is not None:
            # Bullish crossover: fast EMA crosses above slow EMA
            if is_fast_above_slow and not self.was_fast_above_slow:
                crossover_bullish = True
            # Bearish crossover: fast EMA crosses below slow EMA
            elif not is_fast_above_slow and self.was_fast_above_slow:
                crossover_bearish = True
        
        # Update state for next candle
        self.was_fast_above_slow = is_fast_above_slow
        
        # Apply crossover threshold filter
        if abs(ema_diff_pct) < self.params.crossover_threshold_pct:
            return self._hold_signal(
                current_time, current_price,
                f"EMAs too close ({ema_diff_pct:.3f}%)"
            )
        
        # Apply price position filter
        if self.params.require_price_above_ema:
            if crossover_bullish and current_price < fast_ema:
                crossover_bullish = False
            if crossover_bearish and current_price > fast_ema:
                crossover_bearish = False
        
        # Apply EMA slope filter
        if self.params.use_ema_slope_filter:
            if crossover_bullish and fast_ema_slope < self.params.min_ema_slope:
                crossover_bullish = False
            if crossover_bearish and fast_ema_slope > -self.params.min_ema_slope:
                crossover_bearish = False
        
        # Apply MACD confirmation
        if self.params.use_macd_confirmation:
            macd = current_row.get('macd', 0)
            macd_signal = current_row.get('macd_signal', 0)
            
            if not pd.isna(macd) and not pd.isna(macd_signal):
                macd_bullish = macd > macd_signal
                macd_bearish = macd < macd_signal
                
                if crossover_bullish and not macd_bullish:
                    return self._hold_signal(
                        current_time, current_price,
                        "Bullish crossover not confirmed by MACD"
                    )
                if crossover_bearish and not macd_bearish:
                    return self._hold_signal(
                        current_time, current_price,
                        "Bearish crossover not confirmed by MACD"
                    )
        
        # Generate signal
        if crossover_bullish:
            return self._generate_entry_signal(
                current_time,
                current_price,
                SignalType.LONG,
                fast_ema,
                slow_ema,
                fast_ema_slope,
                ema_diff_pct
            )
        
        if crossover_bearish:
            return self._generate_entry_signal(
                current_time,
                current_price,
                SignalType.SHORT,
                fast_ema,
                slow_ema,
                fast_ema_slope,
                ema_diff_pct
            )
        
        # No crossover
        return self._hold_signal(
            current_time, current_price,
            f"No crossover (fast {'>' if is_fast_above_slow else '<'} slow)"
        )
    
    def _get_ema_value(
        self,
        df: pd.DataFrame,
        idx: int,
        period: int
    ) -> Optional[float]:
        """Get EMA value from DataFrame, trying multiple column names."""
        row = df.iloc[idx]
        
        # Try exact column name
        col = f'ema_{period}'
        if col in df.columns:
            return row[col]
        
        # Try to find closest available EMA
        ema_cols = [c for c in df.columns if c.startswith('ema_') and 
                   not any(x in c for x in ['slope', 'accel', 'cross'])]
        
        if not ema_cols:
            return None
        
        # Find closest period
        available_periods = []
        for c in ema_cols:
            try:
                p = int(c.split('_')[1])
                available_periods.append((p, c))
            except (ValueError, IndexError):
                continue
        
        if not available_periods:
            return None
        
        closest = min(available_periods, key=lambda x: abs(x[0] - period))
        return row[closest[1]]
    
    def _generate_entry_signal(
        self,
        timestamp,
        price: float,
        signal_type: SignalType,
        fast_ema: float,
        slow_ema: float,
        ema_slope: float,
        ema_diff_pct: float
    ) -> StrategySignal:
        """Generate entry signal."""
        
        # Calculate confidence based on crossover strength
        confidence = min(0.5 + abs(ema_diff_pct) / 1.0, 1.0)
        
        # Boost confidence if EMA slope is strong
        if abs(ema_slope) > self.params.min_ema_slope * 2:
            confidence = min(confidence + 0.1, 1.0)
        
        # Build reason string
        direction = "LONG (bullish crossover)" if signal_type == SignalType.LONG else "SHORT (bearish crossover)"
        reason = (
            f"EMA{self.params.ema_fast_period}/{self.params.ema_slow_period} "
            f"crossover, diff={ema_diff_pct:.3f}%, {direction}"
        )
        
        return StrategySignal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            reason=reason,
            metadata={
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'ema_diff_pct': ema_diff_pct,
                'ema_slope': ema_slope,
                'fast_period': self.params.ema_fast_period,
                'slow_period': self.params.ema_slow_period
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
        """Get EMA-specific parameters."""
        return {
            'ema_fast_period': self.params.ema_fast_period,
            'ema_slow_period': self.params.ema_slow_period,
            'use_macd_confirmation': self.params.use_macd_confirmation,
            'use_adx_filter': self.params.use_adx_filter,
            'min_adx': self.params.min_adx,
            'crossover_threshold_pct': self.params.crossover_threshold_pct,
            'require_price_above_ema': self.params.require_price_above_ema,
            'use_ema_slope_filter': self.params.use_ema_slope_filter,
            'min_ema_slope': self.params.min_ema_slope
        }
    
    def set_strategy_specific_params(self, params: Dict[str, Any]) -> None:
        """Set EMA-specific parameters."""
        if 'ema_fast_period' in params:
            self.params.ema_fast_period = int(params['ema_fast_period'])
        if 'ema_slow_period' in params:
            self.params.ema_slow_period = int(params['ema_slow_period'])
            # Ensure slow > fast
            if self.params.ema_slow_period <= self.params.ema_fast_period:
                self.params.ema_slow_period = self.params.ema_fast_period + 5
        if 'use_macd_confirmation' in params:
            self.params.use_macd_confirmation = bool(params['use_macd_confirmation'])
        if 'use_adx_filter' in params:
            self.params.use_adx_filter = bool(params['use_adx_filter'])
        if 'min_adx' in params:
            self.params.min_adx = float(params['min_adx'])
        if 'crossover_threshold_pct' in params:
            self.params.crossover_threshold_pct = float(params['crossover_threshold_pct'])
        if 'require_price_above_ema' in params:
            self.params.require_price_above_ema = bool(params['require_price_above_ema'])
        if 'use_ema_slope_filter' in params:
            self.params.use_ema_slope_filter = bool(params['use_ema_slope_filter'])
        if 'min_ema_slope' in params:
            self.params.min_ema_slope = float(params['min_ema_slope'])
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for all parameters."""
        return self.params.get_bounds()
    
    def get_optimal_market_conditions(self) -> Dict[str, Any]:
        """Return the market conditions where this strategy performs best."""
        return {
            'trend_regime': ['trending_up', 'trending_down'],
            'volatility_regime': ['normal', 'moderate'],
            'adx_range': (self.params.min_adx, 60),
            'best_timeframes': ['15m', '1h', '4h'],
            'avoid_conditions': [
                'sideways_market',
                'very_high_volatility',
                'choppy_price_action',
                'low_volume'
            ]
        }
    
    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.was_fast_above_slow = None
    
    def explain_signal(self, signal: StrategySignal) -> str:
        """Generate a human-readable explanation of the signal."""
        if signal.signal_type == SignalType.HOLD:
            return f"No trade: {signal.reason}"
        
        fast_ema = signal.metadata.get('fast_ema', 'N/A')
        slow_ema = signal.metadata.get('slow_ema', 'N/A')
        diff = signal.metadata.get('ema_diff_pct', 'N/A')
        
        if signal.signal_type == SignalType.LONG:
            explanation = (
                f"BUY SIGNAL generated:\n"
                f"- Fast EMA ({self.params.ema_fast_period}) crossed ABOVE slow EMA ({self.params.ema_slow_period})\n"
                f"- Fast EMA: {fast_ema:.2f}, Slow EMA: {slow_ema:.2f}\n"
                f"- EMA difference: {diff:.3f}%\n"
                f"- Trend is confirmed (ADX filter passed)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        else:
            explanation = (
                f"SELL/SHORT SIGNAL generated:\n"
                f"- Fast EMA ({self.params.ema_fast_period}) crossed BELOW slow EMA ({self.params.ema_slow_period})\n"
                f"- Fast EMA: {fast_ema:.2f}, Slow EMA: {slow_ema:.2f}\n"
                f"- EMA difference: {diff:.3f}%\n"
                f"- Trend is confirmed (ADX filter passed)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        
        return explanation
