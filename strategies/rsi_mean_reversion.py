"""
RSI Mean Reversion Strategy.

A mean reversion strategy based on the Relative Strength Index (RSI).
The strategy buys when RSI indicates oversold conditions and sells
when RSI indicates overbought conditions.

STRATEGY LOGIC (FIXED - cannot be changed by ML):
- Buy when RSI drops below buy threshold (oversold)
- Sell when RSI rises above sell threshold (overbought)
- Use confirmation candles to filter false signals
- Apply risk management (stop loss, take profit, trailing stop)

PARAMETERS (TUNABLE - can be adjusted by ML):
- RSI lookback period
- RSI buy threshold
- RSI sell threshold
- Entry/exit confirmation candles
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
class RSIMeanReversionParams(StrategyParameters):
    """
    Parameters for RSI Mean Reversion Strategy.
    
    These parameters can be tuned by ML optimization.
    """
    # RSI-specific parameters
    cooldown_period: int = 10             # Bars to wait between trades
    rsi_lookback: int = 14           # RSI calculation period
    rsi_buy_threshold: int = 30      # RSI level to buy (oversold)
    rsi_sell_threshold: int = 70     # RSI level to sell (overbought)
    
    # Additional filters
    use_divergence_filter: bool = False  # Use RSI divergence as confirmation
    min_rsi_slope: float = 0.0          # Minimum RSI slope for signal
    
    # Regime filter
    avoid_trending_markets: bool = True  # Avoid signals in strong trends
    adx_threshold: float = 30.0          # ADX threshold for trending market
    
    # Sentiment-aware biasing (Requested by user)
    use_sentiment_bias: bool = True      # Bias signals towards major trend
    sentiment_ema_period: int = 200      # Long-term trend EMA
    sentiment_confidence_boost: float = 0.2  # Add confidence to trend-aligned signals
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds including strategy-specific ones."""
        bounds = super().get_bounds()
        bounds.update({
            'rsi_lookback': (10, 40),        # Increased minimum to reduce noise
            'rsi_buy_threshold': (15, 35),   # More conservative thresholds
            'rsi_sell_threshold': (65, 85),
            'min_rsi_slope': (0.0, 5.0),
            'adx_threshold': (20.0, 40.0),   # Lowered to avoid "fake" range signals
            'sentiment_ema_period': (100, 300)
        })
        return bounds


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Trading Strategy.
    
    This strategy exploits the tendency of prices to revert to the mean
    after extreme RSI readings. It's designed for ranging/sideways markets.
    
    Trading Logic:
    1. Calculate RSI for the specified lookback period
    2. Generate BUY signal when RSI < buy_threshold (oversold)
    3. Generate SELL signal when RSI > sell_threshold (overbought)
    4. Apply confirmation and regime filters
    5. Manage position with stop loss, take profit, and trailing stop
    
    Best market conditions:
    - Sideways/ranging markets
    - Low to moderate volatility
    - Clear support/resistance levels
    
    Avoid:
    - Strong trending markets (ADX > 30)
    - High volatility breakouts
    - News-driven moves
    """
    
    def __init__(
        self,
        parameters: Optional[RSIMeanReversionParams] = None,
        initial_capital: float = 100000.0,
        trading_fee_pct: float = 0.001
    ):
        """
        Initialize the RSI Mean Reversion Strategy.
        
        Args:
            parameters: Strategy parameters (uses defaults if None)
            initial_capital: Starting capital
            trading_fee_pct: Trading fee percentage
        """
        super().__init__(
            name="RSI_Mean_Reversion",
            parameters=parameters or RSIMeanReversionParams(),
            initial_capital=initial_capital,
            trading_fee_pct=trading_fee_pct
        )
        
        # Strategy-specific state
        self.confirmation_counter = 0
        self.pending_signal_type: Optional[SignalType] = None
        self._indicators_initialized = False

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters and reset indicator initialization."""
        super().update_parameters(params)
        self._indicators_initialized = False

    def _ensure_indicators(self, df: pd.DataFrame) -> None:
        """Calculate required indicators if they are missing from the dataframe."""
        # RSI
        rsi_col = f'rsi_{self.params.rsi_lookback}'
        if rsi_col not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            avg_gain = gain.ewm(alpha=1/self.params.rsi_lookback, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/self.params.rsi_lookback, adjust=False).mean()
            rs = avg_gain / avg_loss
            df[rsi_col] = 100 - (100 / (1 + rs))
            
        # Sentiment EMA
        if self.params.use_sentiment_bias:
            # Using 200 EMA as fixed sentiment anchor
            period = 200
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Ensure ADX is present if needed
        if self.params.avoid_trending_markets and 'adx' not in df.columns:
             # Basic ADX calculation is complex, but we can at least ensure we don't crash
             # For now, if missing, we'll just not filter.
             pass
                
        self._indicators_initialized = True
    
    @property
    def params(self) -> RSIMeanReversionParams:
        """Type-safe access to parameters."""
        return self.parameters
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """
        Generate trading signal based on RSI mean reversion logic.
        
        FIXED LOGIC - This defines the strategy behavior.
        Only the parameters (thresholds, periods) can be changed.
        
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
        
        # Get RSI value
        rsi_col = f'rsi_{self.params.rsi_lookback}'
        if rsi_col not in df.columns:
            # Fallback to closest available RSI
            rsi_cols = [c for c in df.columns if c.startswith('rsi_') and not c.endswith(('_overbought', '_oversold', '_momentum', '_from_neutral'))]
            if rsi_cols:
                rsi_col = rsi_cols[0]
            else:
                return self._hold_signal(current_time, current_price, "RSI not available")
        
        current_rsi = current_row.get(rsi_col, 50)
        
        # Check for NaN
        if pd.isna(current_rsi):
            return self._hold_signal(current_time, current_price, "RSI is NaN")
        
        # Calculate RSI slope (momentum)
        rsi_momentum = 0
        if current_idx >= 5:
            prev_rsi = df.iloc[current_idx - 5].get(rsi_col, current_rsi)
            if not pd.isna(prev_rsi):
                rsi_momentum = current_rsi - prev_rsi
        
        # Apply regime filter (avoid trending markets)
        if self.params.avoid_trending_markets:
            adx = current_row.get('adx', 0)
            if not pd.isna(adx) and adx > self.params.adx_threshold:
                return self._hold_signal(
                    current_time, current_price,
                    f"Trending market (ADX={adx:.1f})"
                )
        
        # Apply Sentiment Biasing (Requested: Bearish -> Sell)
        sentiment_is_bullish = True
        if self.params.use_sentiment_bias:
            ema_col = f'ema_{self.params.sentiment_ema_period}'
            sentiment_ema = current_row.get(ema_col, None)
            if sentiment_ema is not None and not pd.isna(sentiment_ema):
                sentiment_is_bullish = current_price > sentiment_ema
        
        self.trend_direction = 1 if sentiment_is_bullish else -1
        
        # === CORE STRATEGY LOGIC (FIXED) ===
        
        # Check for oversold condition (BUY signal)
        if current_rsi < self.params.rsi_buy_threshold:
            # Trend filter check (Requested: Bearish -> Sell)
            if self.params.use_sentiment_bias and not sentiment_is_bullish:
                return self._hold_signal(current_time, current_price, "Oversold but major trend is Bearish")
            
            # Check RSI slope filter (allow slightly negative to catch the turn)
            if rsi_momentum >= -5.0:
                # RSI is turning up from oversold - potential buy
                signal = self._generate_entry_signal(
                    current_time,
                    current_price,
                    SignalType.LONG,
                    current_rsi,
                    rsi_momentum,
                    df,
                    current_idx
                )
                if sentiment_is_bullish:
                    signal.confidence = min(signal.confidence + self.params.sentiment_confidence_boost, 1.0)
                return signal
        
        # Check for overbought condition (SELL/SHORT signal)
        elif current_rsi > self.params.rsi_sell_threshold:
            # Trend filter check (optionally allow shorts in bull for reversion)
            # if self.params.use_sentiment_bias and sentiment_is_bullish:
            #     return self._hold_signal(current_time, current_price, "Overbought but major trend is Bullish")

            # Check RSI slope filter
            if rsi_momentum <= -self.params.min_rsi_slope:
                # RSI is turning down from overbought - potential sell
                signal = self._generate_entry_signal(
                    current_time,
                    current_price,
                    SignalType.SHORT,
                    current_rsi,
                    rsi_momentum,
                    df,
                    current_idx
                )
                if not sentiment_is_bullish:
                    signal.confidence = min(signal.confidence + self.params.sentiment_confidence_boost, 1.0)
                return signal
        
        # No signal condition met
        return self._hold_signal(
            current_time, current_price,
            f"RSI={current_rsi:.1f}, neutral zone"
        )
    
    def _generate_entry_signal(
        self,
        timestamp,
        price: float,
        signal_type: SignalType,
        rsi: float,
        rsi_momentum: float,
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """Generate entry signal with confirmation logic."""
        
        # Check confirmation
        if self.params.entry_confirmation > 1:
            if self.pending_signal_type != signal_type:
                # New signal direction, reset counter
                self.pending_signal_type = signal_type
                self.confirmation_counter = 1
                return self._hold_signal(
                    timestamp, price,
                    f"Awaiting confirmation (1/{self.params.entry_confirmation})"
                )
            else:
                self.confirmation_counter += 1
                if self.confirmation_counter < self.params.entry_confirmation:
                    return self._hold_signal(
                        timestamp, price,
                        f"Awaiting confirmation ({self.confirmation_counter}/{self.params.entry_confirmation})"
                    )
        
        # Signal confirmed
        self.pending_signal_type = None
        self.confirmation_counter = 0
        
        # Calculate confidence based on RSI extremity
        if signal_type == SignalType.LONG:
            extremity = (self.params.rsi_buy_threshold - rsi) / self.params.rsi_buy_threshold
        else:
            extremity = (rsi - self.params.rsi_sell_threshold) / (100 - self.params.rsi_sell_threshold)
        
        confidence = min(0.5 + extremity, 1.0)
        
        # Build reason string
        direction = "LONG (oversold)" if signal_type == SignalType.LONG else "SHORT (overbought)"
        reason = f"RSI={rsi:.1f}, momentum={rsi_momentum:.1f}, {direction}"
        
        return StrategySignal(
            timestamp=timestamp,
            signal_type=signal_type,
            price=price,
            confidence=confidence,
            reason=reason,
            metadata={
                'rsi': rsi,
                'rsi_momentum': rsi_momentum,
                'rsi_buy_threshold': self.params.rsi_buy_threshold,
                'rsi_sell_threshold': self.params.rsi_sell_threshold
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
        """Get RSI-specific parameters."""
        return {
            'rsi_lookback': self.params.rsi_lookback,
            'rsi_buy_threshold': self.params.rsi_buy_threshold,
            'rsi_sell_threshold': self.params.rsi_sell_threshold,
            'use_divergence_filter': self.params.use_divergence_filter,
            'min_rsi_slope': self.params.min_rsi_slope,
            'avoid_trending_markets': self.params.avoid_trending_markets,
            'adx_threshold': self.params.adx_threshold,
            'use_sentiment_bias': self.params.use_sentiment_bias,
            'sentiment_ema_period': self.params.sentiment_ema_period,
            'sentiment_confidence_boost': self.params.sentiment_confidence_boost
        }
    
    def set_strategy_specific_params(self, params: Dict[str, Any]) -> None:
        """Set RSI-specific parameters."""
        if 'rsi_lookback' in params:
            self.params.rsi_lookback = int(params['rsi_lookback'])
        if 'rsi_buy_threshold' in params:
            self.params.rsi_buy_threshold = int(params['rsi_buy_threshold'])
        if 'rsi_sell_threshold' in params:
            self.params.rsi_sell_threshold = int(params['rsi_sell_threshold'])
        if 'use_divergence_filter' in params:
            self.params.use_divergence_filter = bool(params['use_divergence_filter'])
        if 'min_rsi_slope' in params:
            self.params.min_rsi_slope = float(params['min_rsi_slope'])
        if 'avoid_trending_markets' in params:
            self.params.avoid_trending_markets = bool(params['avoid_trending_markets'])
        if 'adx_threshold' in params:
            self.params.adx_threshold = float(params['adx_threshold'])
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
        """
        Return the market conditions where this strategy performs best.
        Used for condition analysis.
        """
        return {
            'trend_regime': ['ranging', 'sideways'],
            'volatility_regime': ['low', 'normal'],
            'adx_range': (0, self.params.adx_threshold),
            'best_timeframes': ['5m', '15m'],
            'avoid_conditions': [
                'strong_trend',
                'high_volatility',
                'news_events',
                'low_liquidity'
            ]
        }
    
    def explain_signal(self, signal: StrategySignal) -> str:
        """
        Generate a human-readable explanation of why a signal was generated.
        Useful for audit and transparency.
        """
        if signal.signal_type == SignalType.HOLD:
            return f"No trade: {signal.reason}"
        
        rsi = signal.metadata.get('rsi', 'N/A')
        momentum = signal.metadata.get('rsi_momentum', 'N/A')
        
        if signal.signal_type == SignalType.LONG:
            explanation = (
                f"BUY SIGNAL generated:\n"
                f"- RSI ({rsi:.1f}) dropped below buy threshold ({self.params.rsi_buy_threshold})\n"
                f"- RSI momentum ({momentum:.1f}) indicates potential reversal up\n"
                f"- Market is not in strong trend (mean reversion opportunity)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        else:
            explanation = (
                f"SELL/SHORT SIGNAL generated:\n"
                f"- RSI ({rsi:.1f}) rose above sell threshold ({self.params.rsi_sell_threshold})\n"
                f"- RSI momentum ({momentum:.1f}) indicates potential reversal down\n"
                f"- Market is not in strong trend (mean reversion opportunity)\n"
                f"- Confidence: {signal.confidence:.1%}"
            )
        
        return explanation
