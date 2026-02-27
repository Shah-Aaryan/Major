"""
Parallel Strategy Execution Engine (A2 - CRITICAL)

Runs two strategies side-by-side on identical data:
1. Human-defined (static parameters)
2. ML-adjusted (dynamic parameters)

Produces aligned time-series outputs for comparison:
- Equity curves
- Trade logs
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import copy

from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType
from strategies.strategy_parser import StrategySpec
from backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyRunConfig:
    """Configuration for a single strategy run."""
    name: str
    parameters: Dict[str, Any]
    is_ml_adjusted: bool = False
    description: str = ""


@dataclass
class TradeLog:
    """Individual trade record."""
    timestamp: datetime
    action: str  # "buy", "sell", "exit"
    price: float
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""
    strategy_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "reason": self.reason,
            "strategy_name": self.strategy_name
        }


@dataclass
class ParallelExecutionResult:
    """
    Result of parallel strategy execution.
    
    Contains aligned outputs for both human and ML strategies.
    """
    # Metadata
    start_time: datetime
    end_time: datetime
    data_length: int
    
    # Human strategy results
    human_equity_curve: pd.Series
    human_trades: List[TradeLog]
    human_metrics: PerformanceMetrics
    human_params: Dict[str, Any]
    
    # ML strategy results
    ml_equity_curve: pd.Series
    ml_trades: List[TradeLog]
    ml_metrics: PerformanceMetrics
    ml_params: Dict[str, Any]
    
    # Comparison metrics
    improvement_sharpe: float = 0.0
    improvement_return: float = 0.0
    improvement_drawdown: float = 0.0
    ml_outperformed: bool = False
    
    # Parameter changes
    parameter_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate comparison metrics."""
        if self.human_metrics and self.ml_metrics:
            # Sharpe improvement
            if self.human_metrics.sharpe_ratio != 0:
                self.improvement_sharpe = (
                    (self.ml_metrics.sharpe_ratio - self.human_metrics.sharpe_ratio)
                    / abs(self.human_metrics.sharpe_ratio) * 100
                )
            
            # Return improvement
            self.improvement_return = (
                self.ml_metrics.total_return - self.human_metrics.total_return
            ) * 100
            
            # Drawdown improvement (lower is better)
            self.improvement_drawdown = (
                self.human_metrics.max_drawdown - self.ml_metrics.max_drawdown
            ) * 100
            
            # Overall outperformance
            self.ml_outperformed = self.ml_metrics.sharpe_ratio > self.human_metrics.sharpe_ratio
        
        # Calculate parameter changes
        for param_name in set(self.human_params.keys()) | set(self.ml_params.keys()):
            human_val = self.human_params.get(param_name)
            ml_val = self.ml_params.get(param_name)
            
            if human_val is not None and ml_val is not None:
                if isinstance(human_val, (int, float)) and human_val != 0:
                    change_pct = (ml_val - human_val) / abs(human_val) * 100
                else:
                    change_pct = 0 if human_val == ml_val else 100
                
                self.parameter_changes[param_name] = {
                    "human": human_val,
                    "ml": ml_val,
                    "change_pct": change_pct
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "start_time": str(self.start_time),
                "end_time": str(self.end_time),
                "data_length": self.data_length
            },
            "human": {
                "params": self.human_params,
                "metrics": {
                    "total_return": self.human_metrics.total_return,
                    "sharpe_ratio": self.human_metrics.sharpe_ratio,
                    "max_drawdown": self.human_metrics.max_drawdown,
                    "win_rate": self.human_metrics.win_rate,
                    "profit_factor": self.human_metrics.profit_factor
                },
                "trade_count": len(self.human_trades)
            },
            "ml": {
                "params": self.ml_params,
                "metrics": {
                    "total_return": self.ml_metrics.total_return,
                    "sharpe_ratio": self.ml_metrics.sharpe_ratio,
                    "max_drawdown": self.ml_metrics.max_drawdown,
                    "win_rate": self.ml_metrics.win_rate,
                    "profit_factor": self.ml_metrics.profit_factor
                },
                "trade_count": len(self.ml_trades)
            },
            "comparison": {
                "improvement_sharpe_pct": self.improvement_sharpe,
                "improvement_return_pct": self.improvement_return,
                "improvement_drawdown_pct": self.improvement_drawdown,
                "ml_outperformed": self.ml_outperformed,
                "parameter_changes": self.parameter_changes
            }
        }
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curves as aligned DataFrame."""
        return pd.DataFrame({
            "human_equity": self.human_equity_curve,
            "ml_equity": self.ml_equity_curve
        })
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PARALLEL EXECUTION RESULT",
            "=" * 60,
            f"Period: {self.start_time} to {self.end_time}",
            f"Data points: {self.data_length}",
            "",
            "HUMAN STRATEGY:",
            f"  Total Return: {self.human_metrics.total_return:.2%}",
            f"  Sharpe Ratio: {self.human_metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: {self.human_metrics.max_drawdown:.2%}",
            f"  Win Rate: {self.human_metrics.win_rate:.2%}",
            f"  Trades: {len(self.human_trades)}",
            "",
            "ML-ADJUSTED STRATEGY:",
            f"  Total Return: {self.ml_metrics.total_return:.2%}",
            f"  Sharpe Ratio: {self.ml_metrics.sharpe_ratio:.2f}",
            f"  Max Drawdown: {self.ml_metrics.max_drawdown:.2%}",
            f"  Win Rate: {self.ml_metrics.win_rate:.2%}",
            f"  Trades: {len(self.ml_trades)}",
            "",
            "COMPARISON:",
            f"  Sharpe Improvement: {self.improvement_sharpe:+.2f}%",
            f"  Return Improvement: {self.improvement_return:+.2f}%",
            f"  Drawdown Improvement: {self.improvement_drawdown:+.2f}%",
            f"  ML Outperformed: {'YES' if self.ml_outperformed else 'NO'}",
            "",
            "PARAMETER CHANGES:"
        ]
        
        for param, change in self.parameter_changes.items():
            lines.append(f"  {param}: {change['human']} → {change['ml']} ({change['change_pct']:+.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelStrategyExecutor:
    """
    Executes two strategies (human vs ML) side-by-side on identical data.
    
    This class ensures:
    - Same OHLCV data stream
    - Same execution assumptions
    - Same transaction costs
    - Aligned time-series outputs
    
    Usage:
        executor = ParallelStrategyExecutor(backtest_config)
        
        result = executor.run(
            data=df,
            strategy=my_strategy,
            human_params={"rsi_threshold": 30},
            ml_params={"rsi_threshold": 25}
        )
        
        print(result.summary())
    """
    
    def __init__(
        self,
        backtest_config: Optional[BacktestConfig] = None,
        verbose: bool = True
    ):
        """
        Initialize the parallel executor.
        
        Args:
            backtest_config: Configuration for backtesting
            verbose: Enable logging output
        """
        self.config = backtest_config or BacktestConfig()
        self.verbose = verbose
        self.backtest_engine = BacktestEngine(self.config)
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        human_params: Dict[str, Any],
        ml_params: Dict[str, Any],
        strategy_spec: Optional[StrategySpec] = None
    ) -> ParallelExecutionResult:
        """
        Run parallel execution of human vs ML strategies.
        
        Args:
            data: OHLCV DataFrame with features
            strategy: Strategy instance to use
            human_params: Static human-defined parameters
            ml_params: ML-optimized parameters
            strategy_spec: Optional parsed strategy spec
            
        Returns:
            ParallelExecutionResult with comparison data
        """
        if self.verbose:
            logger.info("Starting parallel strategy execution...")
            logger.info(f"  Data length: {len(data)}")
            logger.info(f"  Human params: {human_params}")
            logger.info(f"  ML params: {ml_params}")
        
        # Create deep copies of strategy to avoid state interference
        human_strategy = copy.deepcopy(strategy)
        ml_strategy = copy.deepcopy(strategy)
        
        # Apply parameters
        human_strategy.update_parameters(human_params)
        ml_strategy.update_parameters(ml_params)
        
        # Run backtests
        if self.verbose:
            logger.info("Running human strategy backtest...")
        human_result = self.backtest_engine.run(human_strategy, data)
        
        if self.verbose:
            logger.info("Running ML strategy backtest...")
        ml_result = self.backtest_engine.run(ml_strategy, data)
        
        # Convert trades to TradeLog
        human_trades = self._convert_trades(human_result, "human")
        ml_trades = self._convert_trades(ml_result, "ml")
        
        # Create result
        result = ParallelExecutionResult(
            start_time=data.index[0] if len(data) > 0 else datetime.now(),
            end_time=data.index[-1] if len(data) > 0 else datetime.now(),
            data_length=len(data),
            human_equity_curve=human_result.equity_curve,
            human_trades=human_trades,
            human_metrics=human_result.metrics,
            human_params=human_params,
            ml_equity_curve=ml_result.equity_curve,
            ml_trades=ml_trades,
            ml_metrics=ml_result.metrics,
            ml_params=ml_params
        )
        
        if self.verbose:
            logger.info("Parallel execution complete")
            logger.info(f"  ML outperformed: {result.ml_outperformed}")
            logger.info(f"  Sharpe improvement: {result.improvement_sharpe:.2f}%")
        
        return result
    
    def run_from_spec(
        self,
        data: pd.DataFrame,
        strategy_spec: StrategySpec,
        strategy_factory: Callable[[Dict[str, Any]], BaseStrategy],
        ml_params: Dict[str, Any]
    ) -> ParallelExecutionResult:
        """
        Run parallel execution using a parsed StrategySpec.
        
        Args:
            data: OHLCV DataFrame with features
            strategy_spec: Parsed strategy specification
            strategy_factory: Function to create strategy from params
            ml_params: ML-optimized parameters
            
        Returns:
            ParallelExecutionResult
        """
        # Human params from the spec (user-provided values)
        human_params = strategy_spec.get_parameter_values()
        
        # Create strategies
        human_strategy = strategy_factory(human_params)
        ml_strategy = strategy_factory(ml_params)
        
        # Run both
        human_result = self.backtest_engine.run(human_strategy, data)
        ml_result = self.backtest_engine.run(ml_strategy, data)
        
        return ParallelExecutionResult(
            start_time=data.index[0],
            end_time=data.index[-1],
            data_length=len(data),
            human_equity_curve=human_result.equity_curve,
            human_trades=self._convert_trades(human_result, "human"),
            human_metrics=human_result.metrics,
            human_params=human_params,
            ml_equity_curve=ml_result.equity_curve,
            ml_trades=self._convert_trades(ml_result, "ml"),
            ml_metrics=ml_result.metrics,
            ml_params=ml_params
        )
    
    def run_streaming(
        self,
        data_stream: pd.DataFrame,
        strategy: BaseStrategy,
        human_params: Dict[str, Any],
        ml_params: Dict[str, Any],
        window_size: int = 100,
        update_frequency: int = 10
    ) -> List[ParallelExecutionResult]:
        """
        Run parallel execution in streaming mode with rolling windows.
        
        This simulates real-time execution where ML params may be updated
        periodically while human params remain static.
        
        Args:
            data_stream: Full data to stream through
            strategy: Strategy to use
            human_params: Static human parameters
            ml_params: Initial ML parameters
            window_size: Size of evaluation window
            update_frequency: How often to output results
            
        Returns:
            List of ParallelExecutionResult at each checkpoint
        """
        results = []
        
        for i in range(window_size, len(data_stream), update_frequency):
            window_data = data_stream.iloc[i-window_size:i]
            
            result = self.run(
                data=window_data,
                strategy=strategy,
                human_params=human_params,
                ml_params=ml_params
            )
            results.append(result)
            
            if self.verbose:
                logger.info(f"Checkpoint {i}/{len(data_stream)}: ML {'outperformed' if result.ml_outperformed else 'underperformed'}")
        
        return results
    
    def _convert_trades(
        self, 
        result: BacktestResult, 
        strategy_name: str
    ) -> List[TradeLog]:
        """Convert BacktestResult trades to TradeLog format."""
        trades = []
        
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades:
                # Entry trade log
                trades.append(TradeLog(
                    timestamp=trade.entry_timestamp,
                    action="buy" if trade.side == "long" else "sell",
                    price=trade.entry_price,
                    quantity=trade.quantity,
                    pnl=0,
                    pnl_pct=0,
                    reason="entry",
                    strategy_name=strategy_name
                ))
                
                # Exit trade log
                trades.append(TradeLog(
                    timestamp=trade.exit_timestamp,
                    action="exit",
                    price=trade.exit_price,
                    quantity=trade.quantity,
                    pnl=trade.pnl,
                    pnl_pct=trade.pnl_pct,
                    reason=trade.exit_reason,
                    strategy_name=strategy_name
                ))
        
        return trades
    
    def get_aligned_timeseries(
        self,
        result: ParallelExecutionResult
    ) -> pd.DataFrame:
        """
        Get aligned time series data for frontend visualization.
        
        Returns DataFrame with:
        - timestamp (index)
        - human_equity
        - ml_equity
        - human_position
        - ml_position
        - relative_performance
        """
        df = result.get_equity_dataframe()
        
        # Calculate relative performance
        df["relative_performance"] = (
            (df["ml_equity"] - df["human_equity"]) / df["human_equity"] * 100
        )
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# FRONTEND OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def format_for_frontend(result: ParallelExecutionResult) -> Dict[str, Any]:
    """
    Format result for frontend consumption.
    
    Schema:
    {
        "metadata": {...},
        "equity_curves": {
            "timestamps": [...],
            "human": [...],
            "ml": [...]
        },
        "metrics_comparison": {...},
        "trade_events": [...],
        "parameter_diff": {...}
    }
    """
    equity_df = result.get_equity_dataframe()
    
    return {
        "metadata": {
            "start": str(result.start_time),
            "end": str(result.end_time),
            "points": result.data_length
        },
        "equity_curves": {
            "timestamps": [str(ts) for ts in equity_df.index],
            "human": equity_df["human_equity"].tolist(),
            "ml": equity_df["ml_equity"].tolist()
        },
        "metrics_comparison": {
            "human": {
                "total_return": result.human_metrics.total_return,
                "sharpe": result.human_metrics.sharpe_ratio,
                "max_dd": result.human_metrics.max_drawdown,
                "win_rate": result.human_metrics.win_rate
            },
            "ml": {
                "total_return": result.ml_metrics.total_return,
                "sharpe": result.ml_metrics.sharpe_ratio,
                "max_dd": result.ml_metrics.max_drawdown,
                "win_rate": result.ml_metrics.win_rate
            },
            "improvement": {
                "sharpe_pct": result.improvement_sharpe,
                "return_pct": result.improvement_return,
                "drawdown_pct": result.improvement_drawdown
            }
        },
        "trade_events": {
            "human": [t.to_dict() for t in result.human_trades],
            "ml": [t.to_dict() for t in result.ml_trades]
        },
        "parameter_diff": result.parameter_changes,
        "ml_outperformed": result.ml_outperformed
    }


if __name__ == "__main__":
    # Example usage
    print("ParallelStrategyExecutor ready")
    print("Usage:")
    print("  executor = ParallelStrategyExecutor()")
    print("  result = executor.run(data, strategy, human_params, ml_params)")
    print("  print(result.summary())")
