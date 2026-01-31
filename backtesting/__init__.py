"""
Backtesting module for strategy evaluation.

Provides comprehensive backtesting capabilities including:
- Event-driven backtest engine
- Performance metrics calculation
- Walk-forward optimization
"""

from backtesting.backtest_engine import BacktestEngine, BacktestResult
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics
from backtesting.walk_forward import WalkForwardValidator, WalkForwardResult

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'PerformanceMetrics',
    'calculate_all_metrics',
    'WalkForwardValidator',
    'WalkForwardResult'
]
