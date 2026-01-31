"""
Strategy Engine - Orchestrates all trading strategies.

The Strategy Engine serves as the central manager for all trading strategies.
It provides:
1. Unified interface to generate signals from multiple strategies
2. Strategy comparison and ensemble capabilities
3. Parameter management for ML optimization
4. Performance tracking per strategy

IMPORTANT: The Strategy Engine does NOT modify strategy logic.
It only coordinates strategies and manages their parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    SignalType,
    StrategyParameters
)
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy, RSIMeanReversionParams
from strategies.ema_crossover import EMACrossoverStrategy, EMACrossoverParams
from strategies.bollinger_breakout import BollingerBreakoutStrategy, BollingerBreakoutParams

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available strategy types."""
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    EMA_CROSSOVER = "ema_crossover"
    BOLLINGER_BREAKOUT = "bollinger_breakout"


@dataclass
class StrategyPerformance:
    """Track performance metrics for a strategy."""
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    hold_signals: int = 0
    
    # Will be filled by backtester
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def signal_breakdown(self) -> Dict[str, float]:
        """Get signal type breakdown as percentages."""
        if self.total_signals == 0:
            return {'long': 0.0, 'short': 0.0, 'hold': 0.0}
        return {
            'long': self.long_signals / self.total_signals,
            'short': self.short_signals / self.total_signals,
            'hold': self.hold_signals / self.total_signals
        }


class StrategyEngine:
    """
    Central engine for managing and coordinating trading strategies.
    
    The Strategy Engine provides:
    1. Strategy registration and management
    2. Unified signal generation interface
    3. Multi-strategy signal aggregation
    4. Parameter management for ML optimization
    5. Performance tracking and comparison
    
    Usage:
        engine = StrategyEngine()
        engine.register_strategy(StrategyType.RSI_MEAN_REVERSION)
        
        # Generate signals
        signals = engine.generate_all_signals(df, current_idx)
        
        # Update parameters (for ML optimization)
        engine.update_strategy_params('rsi_mean_reversion', new_params)
    """
    
    # Registry mapping strategy types to their classes
    STRATEGY_REGISTRY: Dict[StrategyType, Type[BaseStrategy]] = {
        StrategyType.RSI_MEAN_REVERSION: RSIMeanReversionStrategy,
        StrategyType.EMA_CROSSOVER: EMACrossoverStrategy,
        StrategyType.BOLLINGER_BREAKOUT: BollingerBreakoutStrategy,
    }
    
    PARAM_REGISTRY: Dict[StrategyType, Type[StrategyParameters]] = {
        StrategyType.RSI_MEAN_REVERSION: RSIMeanReversionParams,
        StrategyType.EMA_CROSSOVER: EMACrossoverParams,
        StrategyType.BOLLINGER_BREAKOUT: BollingerBreakoutParams,
    }
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        trading_fee_pct: float = 0.001
    ):
        """
        Initialize the Strategy Engine.
        
        Args:
            initial_capital: Starting capital for all strategies
            trading_fee_pct: Trading fee percentage
        """
        self.initial_capital = initial_capital
        self.trading_fee_pct = trading_fee_pct
        
        # Active strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # Performance tracking
        self.performance: Dict[str, StrategyPerformance] = {}
        
        # Signal history
        self.signal_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"StrategyEngine initialized with capital={initial_capital}, "
            f"fee={trading_fee_pct*100}%"
        )
    
    def register_strategy(
        self,
        strategy_type: StrategyType,
        parameters: Optional[StrategyParameters] = None,
        custom_name: Optional[str] = None
    ) -> str:
        """
        Register a strategy with the engine.
        
        Args:
            strategy_type: Type of strategy to register
            parameters: Optional custom parameters
            custom_name: Optional custom name (otherwise uses strategy type)
            
        Returns:
            Strategy name/identifier
        """
        if strategy_type not in self.STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = self.STRATEGY_REGISTRY[strategy_type]
        
        # Create strategy instance
        strategy = strategy_class(
            parameters=parameters,
            initial_capital=self.initial_capital,
            trading_fee_pct=self.trading_fee_pct
        )
        
        name = custom_name or strategy_type.value
        
        self.strategies[name] = strategy
        self.performance[name] = StrategyPerformance()
        
        logger.info(f"Registered strategy: {name}")
        
        return name
    
    def register_all_strategies(
        self,
        use_default_params: bool = True
    ) -> List[str]:
        """
        Register all available strategies.
        
        Args:
            use_default_params: Use default parameters if True
            
        Returns:
            List of registered strategy names
        """
        names = []
        for strategy_type in StrategyType:
            name = self.register_strategy(strategy_type)
            names.append(name)
        
        logger.info(f"Registered {len(names)} strategies")
        return names
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names."""
        return list(self.strategies.keys())
    
    def generate_signal(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """
        Generate signal for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            df: DataFrame with OHLCV and features
            current_idx: Current index in the DataFrame
            
        Returns:
            StrategySignal from the strategy
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        signal = strategy.generate_signal(df, current_idx)
        
        # Update performance tracking
        self._update_performance(strategy_name, signal)
        
        return signal
    
    def generate_all_signals(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, StrategySignal]:
        """
        Generate signals from all registered strategies.
        
        Args:
            df: DataFrame with OHLCV and features
            current_idx: Current index in the DataFrame
            
        Returns:
            Dictionary mapping strategy names to signals
        """
        signals = {}
        
        for name in self.strategies:
            try:
                signals[name] = self.generate_signal(name, df, current_idx)
            except Exception as e:
                logger.error(f"Error generating signal for {name}: {e}")
                # Generate a hold signal on error
                signals[name] = StrategySignal(
                    timestamp=df.index[current_idx],
                    signal_type=SignalType.HOLD,
                    price=df.iloc[current_idx]['close'],
                    confidence=0.0,
                    reason=f"Error: {str(e)}"
                )
        
        # Store in history
        self.signal_history.append({
            'timestamp': df.index[current_idx],
            'signals': {k: v.to_dict() for k, v in signals.items()}
        })
        
        return signals
    
    def _update_performance(
        self,
        strategy_name: str,
        signal: StrategySignal
    ) -> None:
        """Update performance tracking for a signal."""
        perf = self.performance[strategy_name]
        perf.total_signals += 1
        
        if signal.signal_type == SignalType.LONG:
            perf.long_signals += 1
        elif signal.signal_type == SignalType.SHORT:
            perf.short_signals += 1
        else:
            perf.hold_signals += 1
    
    # ===========================================
    # PARAMETER MANAGEMENT (for ML optimization)
    # ===========================================
    
    def get_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get current parameters for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of current parameters
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_strategy_specific_params()
    
    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters for all strategies."""
        return {
            name: self.get_strategy_params(name)
            for name in self.strategies
        }
    
    def update_strategy_params(
        self,
        strategy_name: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Update parameters for a strategy.
        
        This is the interface for ML optimization to adjust strategy parameters.
        
        Args:
            strategy_name: Name of the strategy
            params: Dictionary of parameter values to update
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        strategy.set_strategy_specific_params(params)
        
        logger.info(f"Updated parameters for {strategy_name}: {params}")
    
    def get_parameter_bounds(
        self,
        strategy_name: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for ML optimization.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary mapping parameter names to (min, max) bounds
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        return strategy.get_parameter_bounds()
    
    def get_all_parameter_bounds(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Get parameter bounds for all strategies."""
        return {
            name: self.get_parameter_bounds(name)
            for name in self.strategies
        }
    
    # ===========================================
    # ENSEMBLE SIGNALS
    # ===========================================
    
    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        current_idx: int,
        method: str = 'majority',
        weights: Optional[Dict[str, float]] = None
    ) -> StrategySignal:
        """
        Generate an ensemble signal from all strategies.
        
        Args:
            df: DataFrame with OHLCV and features
            current_idx: Current index in the DataFrame
            method: Ensemble method ('majority', 'weighted', 'unanimous')
            weights: Optional weights for weighted voting
            
        Returns:
            Combined StrategySignal
        """
        signals = self.generate_all_signals(df, current_idx)
        
        if method == 'majority':
            return self._majority_vote(signals, df, current_idx)
        elif method == 'weighted':
            return self._weighted_vote(signals, df, current_idx, weights)
        elif method == 'unanimous':
            return self._unanimous_vote(signals, df, current_idx)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _majority_vote(
        self,
        signals: Dict[str, StrategySignal],
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """Majority voting ensemble."""
        votes = {SignalType.LONG: 0, SignalType.SHORT: 0, SignalType.HOLD: 0}
        
        for signal in signals.values():
            votes[signal.signal_type] += 1
        
        # Find winning signal type
        winning_type = max(votes, key=votes.get)
        
        # Calculate confidence as proportion of votes
        total_votes = sum(votes.values())
        confidence = votes[winning_type] / total_votes if total_votes > 0 else 0.0
        
        # Combine reasons
        supporting_strategies = [
            name for name, sig in signals.items()
            if sig.signal_type == winning_type
        ]
        
        return StrategySignal(
            timestamp=df.index[current_idx],
            signal_type=winning_type,
            price=df.iloc[current_idx]['close'],
            confidence=confidence,
            reason=f"Majority vote ({votes[winning_type]}/{total_votes}): {supporting_strategies}",
            metadata={
                'vote_breakdown': {k.value: v for k, v in votes.items()},
                'supporting_strategies': supporting_strategies
            }
        )
    
    def _weighted_vote(
        self,
        signals: Dict[str, StrategySignal],
        df: pd.DataFrame,
        current_idx: int,
        weights: Optional[Dict[str, float]] = None
    ) -> StrategySignal:
        """Weighted voting ensemble."""
        if weights is None:
            weights = {name: 1.0 for name in signals}
        
        # Normalize weights
        total_weight = sum(weights.get(name, 0) for name in signals)
        if total_weight == 0:
            return self._majority_vote(signals, df, current_idx)
        
        weighted_votes = {SignalType.LONG: 0.0, SignalType.SHORT: 0.0, SignalType.HOLD: 0.0}
        
        for name, signal in signals.items():
            weight = weights.get(name, 0) / total_weight
            weighted_votes[signal.signal_type] += weight * signal.confidence
        
        winning_type = max(weighted_votes, key=weighted_votes.get)
        
        return StrategySignal(
            timestamp=df.index[current_idx],
            signal_type=winning_type,
            price=df.iloc[current_idx]['close'],
            confidence=weighted_votes[winning_type],
            reason=f"Weighted vote: {winning_type.value}",
            metadata={'weighted_scores': {k.value: v for k, v in weighted_votes.items()}}
        )
    
    def _unanimous_vote(
        self,
        signals: Dict[str, StrategySignal],
        df: pd.DataFrame,
        current_idx: int
    ) -> StrategySignal:
        """Unanimous voting - only signal if all agree."""
        signal_types = [sig.signal_type for sig in signals.values()]
        unique_types = set(signal_types)
        
        if len(unique_types) == 1 and signal_types[0] != SignalType.HOLD:
            # All strategies agree on a non-hold signal
            avg_confidence = np.mean([sig.confidence for sig in signals.values()])
            return StrategySignal(
                timestamp=df.index[current_idx],
                signal_type=signal_types[0],
                price=df.iloc[current_idx]['close'],
                confidence=avg_confidence,
                reason=f"Unanimous agreement: {signal_types[0].value}",
                metadata={'agreeing_strategies': list(signals.keys())}
            )
        
        # No unanimity - hold
        return StrategySignal(
            timestamp=df.index[current_idx],
            signal_type=SignalType.HOLD,
            price=df.iloc[current_idx]['close'],
            confidence=0.0,
            reason=f"No unanimous agreement: {[t.value for t in unique_types]}"
        )
    
    # ===========================================
    # PERFORMANCE AND REPORTING
    # ===========================================
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all strategies."""
        summary = {}
        
        for name, perf in self.performance.items():
            summary[name] = {
                'total_signals': perf.total_signals,
                'signal_breakdown': perf.signal_breakdown(),
                'total_trades': perf.total_trades,
                'win_rate': perf.win_rate(),
                'total_pnl': perf.total_pnl,
                'max_drawdown': perf.max_drawdown,
                'sharpe_ratio': perf.sharpe_ratio
            }
        
        return summary
    
    def get_strategy_comparison(self) -> pd.DataFrame:
        """Get a DataFrame comparing all strategy performances."""
        data = []
        
        for name, perf in self.performance.items():
            data.append({
                'strategy': name,
                'total_signals': perf.total_signals,
                'long_signals': perf.long_signals,
                'short_signals': perf.short_signals,
                'hold_signals': perf.hold_signals,
                'total_trades': perf.total_trades,
                'win_rate': perf.win_rate(),
                'total_pnl': perf.total_pnl,
                'sharpe_ratio': perf.sharpe_ratio
            })
        
        return pd.DataFrame(data)
    
    def get_optimal_conditions_report(self) -> Dict[str, Dict[str, Any]]:
        """Get optimal market conditions for each strategy."""
        return {
            name: strategy.get_optimal_market_conditions()
            for name, strategy in self.strategies.items()
        }
    
    def reset_all(self) -> None:
        """Reset all strategies and performance tracking."""
        for strategy in self.strategies.values():
            strategy.reset()
        
        self.performance = {
            name: StrategyPerformance()
            for name in self.strategies
        }
        
        self.signal_history.clear()
        
        logger.info("Reset all strategies and performance tracking")
    
    def export_signal_history(self, filepath: str) -> None:
        """Export signal history to a CSV file."""
        if not self.signal_history:
            logger.warning("No signal history to export")
            return
        
        # Flatten signal history for CSV
        rows = []
        for entry in self.signal_history:
            timestamp = entry['timestamp']
            for strategy_name, signal_dict in entry['signals'].items():
                rows.append({
                    'timestamp': timestamp,
                    'strategy': strategy_name,
                    'signal_type': signal_dict['signal_type'],
                    'price': signal_dict['price'],
                    'confidence': signal_dict['confidence'],
                    'reason': signal_dict['reason']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported signal history to {filepath}")


def create_default_engine() -> StrategyEngine:
    """
    Create a StrategyEngine with all strategies registered using default parameters.
    
    Returns:
        Configured StrategyEngine instance
    """
    engine = StrategyEngine()
    engine.register_all_strategies()
    return engine


def create_engine_with_params(
    strategy_params: Dict[str, Dict[str, Any]]
) -> StrategyEngine:
    """
    Create a StrategyEngine with custom parameters.
    
    Args:
        strategy_params: Dictionary mapping strategy names to parameter dictionaries
        
    Returns:
        Configured StrategyEngine instance
    """
    engine = StrategyEngine()
    
    for strategy_type in StrategyType:
        name = strategy_type.value
        if name in strategy_params:
            param_class = engine.PARAM_REGISTRY[strategy_type]
            params = param_class(**strategy_params[name])
            engine.register_strategy(strategy_type, parameters=params)
        else:
            engine.register_strategy(strategy_type)
    
    return engine
