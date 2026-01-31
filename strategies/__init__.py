"""
Strategy module for ML Trading Research Project.
Implements rule-based trading strategies with tunable parameters.
"""

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    SignalType,
    Position,
    StrategyParameters,
    TradeResult
)
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.strategy_engine import StrategyEngine

__all__ = [
    'BaseStrategy',
    'StrategySignal',
    'SignalType',
    'Position',
    'StrategyParameters',
    'TradeResult',
    'RSIMeanReversionStrategy',
    'EMACrossoverStrategy',
    'BollingerBreakoutStrategy',
    'StrategyEngine'
]
