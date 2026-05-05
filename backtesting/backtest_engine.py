"""
Backtest Engine for Strategy Evaluation.

Event-driven backtesting engine that simulates trading with:
- Realistic order execution
- Slippage and transaction costs
- Position sizing
- Risk management

Used to evaluate both human-parameterized and ML-optimized strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from strategies.base_strategy import (
    BaseStrategy,
    StrategySignal,
    SignalType,
    Position,
    TradeResult
)
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    timestamp: datetime
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Execution details (filled after execution)
    executed: bool = False
    execution_price: Optional[float] = None
    execution_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    position_size_pct: float = 0.95  # Use 95% of capital per trade
    allow_shorting: bool = True
    max_positions: int = 1
    
    # Risk management
    stop_loss_pct: Optional[float] = 0.05  # Default 5% stop loss to prevent bankruptcy
    take_profit_pct: Optional[float] = None  # e.g., 0.05 for 5%
    
    # Execution
    use_next_bar_open: bool = True  # Execute at next bar's open
    periods_per_year: int = 252 * 24 * 60  # Default to 1-minute


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_timestamp: datetime
    exit_timestamp: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    duration: int  # in bars
    exit_reason: str = ""
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entry_timestamp': self.entry_timestamp,
            'exit_timestamp': self.exit_timestamp,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'duration': self.duration,
            'exit_reason': self.exit_reason
        }


@dataclass
class BacktestResult:
    """
    Complete result of a backtest.
    
    Contains equity curve, trades, and performance metrics.
    """
    # Core results
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: PerformanceMetrics
    
    # Configuration
    config: BacktestConfig
    strategy_name: str
    parameters: Dict[str, Any]
    
    # Data info
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Signal history
    signals: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_trade_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"=== Backtest Result: {self.strategy_name} ===\n"
            f"Period: {self.start_date} to {self.end_date}\n"
            f"Total Bars: {self.total_bars}\n"
            f"Total Trades: {len(self.trades)}\n"
            f"\n{self.metrics.summary()}"
        )


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Simulates trading by processing data bar by bar, generating
    signals from the strategy, and executing orders with realistic
    assumptions about slippage and commissions.
    
    Usage:
        engine = BacktestEngine(config=BacktestConfig())
        result = engine.run(strategy, data)
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # State variables (reset on each run)
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset internal state for a new backtest."""
        self.capital = self.config.initial_capital
        self.position: Optional[Dict[str, Any]] = None
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.signals: List[Dict[str, Any]] = []
        self.pending_orders: List[Order] = []
    
    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameters: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy instance
            data: DataFrame with OHLCV and features
            parameters: Optional parameters to set on strategy
            
        Returns:
            BacktestResult with all metrics and trades
        """
        self._reset_state()

        # Reset strategy state between runs so results are reproducible and
        # independent of prior backtests/optimizers.
        if hasattr(strategy, "reset"):
            try:
                strategy.reset()
            except Exception:
                # Strategy reset is best-effort; backtest engine still proceeds.
                pass

        # Set parameters if provided.
        # Important: many strategies separate base params (risk/execution) from
        # strategy-specific params; use set_all_parameters when available.
        if parameters:
            if hasattr(strategy, "set_all_parameters"):
                strategy.set_all_parameters(parameters)
            else:
                strategy.set_strategy_specific_params(parameters)
        
        # Validate data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logger.info(
            f"Starting backtest for {strategy.name} "
            f"({len(data)} bars)"
        )
        
        # Main backtest loop
        # Optimization: Pre-extract indices and common values
        data_len = len(data)
        close_prices = data['close'].values
        timestamps = data.index
        
        for idx in range(data_len):
            current_price = close_prices[idx]
            timestamp = timestamps[idx]
            
            # Update equity
            equity = self._calculate_equity(current_price)
            self.equity_history.append((timestamp, equity))

            # Liquidation check (Bankruptcy)
            if equity <= (self.config.initial_capital * 0.05):
                logger.warning(f"CRITICAL: Portfolio liquidated at {timestamp} (Equity: {equity:.2f})")
                if self.position is not None:
                    self._close_position(current_bar['close'], timestamp, "liquidation")
                # Fill remaining history with bankruptcy equity
                for remaining_idx in range(idx + 1, len(data)):
                    self.equity_history.append((data.index[remaining_idx], equity))
                break
            
            # Check stop loss / take profit
            if self.position is not None:
                # Still need current_bar for some exit conditions, but it's only once per iteration with position
                current_bar = data.iloc[idx]
                self._check_exit_conditions(current_bar, timestamp, strategy=strategy)
            
            # Generate signal
            signal = strategy.generate_signal(data, idx)
            
            self.signals.append({
                'timestamp': timestamp,
                'signal_type': signal.signal_type.value,
                'price': signal.price,
                'confidence': signal.confidence,
                'reason': signal.reason
            })
            
            # Process signal
            self._process_signal(signal, data, idx)
        
        # Close any remaining position
        if self.position is not None:
            final_bar = data.iloc[-1]
            final_timestamp = data.index[-1]
            self._close_position(
                final_bar['close'],
                final_timestamp,
                "end_of_backtest"
            )
        
        # Build equity curve
        equity_curve = pd.Series(
            [e[1] for e in self.equity_history],
            index=[e[0] for e in self.equity_history]
        )
        
        # Calculate metrics
        trade_returns = [t.pnl_pct for t in self.trades]
        trade_durations = [t.duration for t in self.trades]
        
        metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            trade_returns=trade_returns,
            trade_durations=trade_durations,
            periods_per_year=self.config.periods_per_year
        )
        
        # Build result
        result = BacktestResult(
            equity_curve=equity_curve,
            trades=self.trades,
            metrics=metrics,
            config=self.config,
            strategy_name=strategy.name,
            parameters=strategy.get_strategy_specific_params(),
            start_date=data.index[0],
            end_date=data.index[-1],
            total_bars=len(data),
            signals=self.signals
        )
        
        logger.info(f"Backtest complete: {len(self.trades)} trades")
        
        return result
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current portfolio equity."""
        if self.position is None:
            return self.capital
        
        # Principal that was locked when opening the position
        locked_principal = self.position['quantity'] * self.position['entry_price']
        
        if self.position['side'] == 'long':
            unrealized_pnl = (self.position['quantity'] * current_price) - locked_principal
        else:  # short
            unrealized_pnl = locked_principal - (self.position['quantity'] * current_price)
            
        # Total equity = Remaining cash + the principal we "invested" + the PnL gained/lost
        return self.capital + locked_principal + unrealized_pnl
    
    def _process_signal(
        self,
        signal: StrategySignal,
        data: pd.DataFrame,
        idx: int
    ) -> None:
        """Process a trading signal."""
        if signal.signal_type == SignalType.HOLD:
            return
        
        bar = data.iloc[idx]
        timestamp = data.index[idx]

        # Determine execution price
        if self.config.use_next_bar_open and idx < len(data) - 1:
            # Execute at NEXT bar's open
            exec_bar = data.iloc[idx + 1]
            exec_price = exec_bar['open']
            exec_timestamp = data.index[idx + 1]
        else:
            # Execute at CURRENT bar's close (fallback or same-bar)
            exec_price = bar['close']
            exec_timestamp = timestamp
        
        # Apply slippage
        slippage = exec_price * self.config.slippage_pct
        
        if signal.signal_type == SignalType.LONG:
            if self.position is not None:
                if self.position['side'] == 'short':
                    # Close short position
                    self._close_position(exec_price - slippage, exec_timestamp, "reverse_signal")
                else:
                    # Already long, do nothing
                    return
            
            # Open long position
            self._open_position('long', exec_price + slippage, exec_timestamp, signal)
        
        elif signal.signal_type == SignalType.SHORT:
            if not self.config.allow_shorting:
                if self.position is not None and self.position['side'] == 'long':
                    # Close long position
                    self._close_position(exec_price - slippage, exec_timestamp, "exit_signal")
                return
            
            if self.position is not None:
                if self.position['side'] == 'long':
                    # Close long position
                    self._close_position(exec_price - slippage, exec_timestamp, "reverse_signal")
                else:
                    # Already short, do nothing
                    return
            
            # Open short position
            self._open_position('short', exec_price - slippage, exec_timestamp, signal)
    
    def _open_position(
        self,
        side: str,
        price: float,
        timestamp: datetime,
        signal: StrategySignal
    ) -> None:
        """Open a new position."""
        # Calculate position size
        available_capital = self.capital * self.config.position_size_pct
        commission = available_capital * self.config.commission_pct

        # Funds available to invest after paying open commission
        investable = available_capital - commission
        if investable <= 0:
            logger.warning("Investable capital <= 0 after commission; skipping open position")
            return

        quantity = investable / price
        position_value = quantity * price

        self.position = {
            'side': side,
            'entry_price': price,
            'entry_timestamp': timestamp,
            'quantity': quantity,
            'entry_bar_idx': len(self.equity_history),
            'signal_metadata': signal.metadata
        }

        # Deduct both the invested principal and the commission from capital
        self.capital -= (position_value + commission)

        logger.debug(
            f"Opened {side} position at {price:.4f}, "
            f"quantity={quantity:.6f}, invested={position_value:.2f}, commission={commission:.2f}"
        )
    
    def _close_position(
        self,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> None:
        """Close the current position."""
        if self.position is None:
            return
        
        # Calculate commission
        position_value = self.position['quantity'] * price
        commission = position_value * self.config.commission_pct
        
        # Calculate PnL
        if self.position['side'] == 'long':
            pnl = (price - self.position['entry_price']) * self.position['quantity']
        else:  # short
            pnl = (self.position['entry_price'] - price) * self.position['quantity']
        
        pnl -= commission
        
        # PnL percentage
        entry_value = self.position['quantity'] * self.position['entry_price']
        pnl_pct = pnl / entry_value if entry_value > 0 else 0.0
        
        # Duration
        duration = len(self.equity_history) - self.position['entry_bar_idx']
        
        # Create trade record
        trade = Trade(
            entry_timestamp=self.position['entry_timestamp'],
            exit_timestamp=timestamp,
            side=self.position['side'],
            entry_price=self.position['entry_price'],
            exit_price=price,
            quantity=self.position['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            duration=duration,
            exit_reason=reason,
            signal_metadata=self.position.get('signal_metadata', {})
        )
        
        self.trades.append(trade)
        
        # Update capital
        self.capital += pnl + (self.position['quantity'] * self.position['entry_price'])
        
        logger.debug(
            f"Closed {self.position['side']} at {price:.4f}, "
            f"PnL={pnl:.2f} ({pnl_pct:.2%})"
        )
        
        self.position = None
    
    def _check_exit_conditions(
        self,
        bar: pd.Series,
        timestamp: datetime,
        strategy: Optional[BaseStrategy] = None
    ) -> None:
        """Check stop loss and take profit conditions."""
        if self.position is None:
            return
        
        high = bar['high']
        low = bar['low']
        entry_price = self.position['entry_price']
        
        # Use strategy parameters if available, else fallback to config
        sl_pct = self.config.stop_loss_pct
        tp_pct = self.config.take_profit_pct
        
        if strategy and hasattr(strategy, 'parameters'):
            # Some strategies use 2.0 for 2%, some use 0.02. Normalize to 0.02.
            s_sl = getattr(strategy.parameters, 'stop_loss_pct', None)
            if s_sl is not None:
                sl_pct = s_sl / 100.0 if s_sl > 0.5 else s_sl
                
            s_tp = getattr(strategy.parameters, 'take_profit_pct', None)
            if s_tp is not None:
                tp_pct = s_tp / 100.0 if s_tp > 0.5 else s_tp

        if self.position['side'] == 'long':
            # Stop loss
            if sl_pct is not None:
                stop_price = entry_price * (1 - sl_pct)
                if low <= stop_price:
                    self._close_position(stop_price, timestamp, "stop_loss")
                    return
            
            # Take profit
            if tp_pct is not None:
                tp_price = entry_price * (1 + tp_pct)
                if high >= tp_price:
                    self._close_position(tp_price, timestamp, "take_profit")
                    return
        
        else:  # short
            # Stop loss
            if sl_pct is not None:
                stop_price = entry_price * (1 + sl_pct)
                if high >= stop_price:
                    self._close_position(stop_price, timestamp, "stop_loss")
                    return
            
            # Take profit
            if tp_pct is not None:
                tp_price = entry_price * (1 - tp_pct)
                if low <= tp_price:
                    self._close_position(tp_price, timestamp, "take_profit")
                    return


def run_backtest_for_optimization(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    params: Dict[str, Any],
    config: Optional[BacktestConfig] = None,
    metric: str = 'sharpe_ratio'
) -> float:
    """
    Convenience function for optimization.
    
    Args:
        strategy: Strategy instance
        data: Backtest data
        params: Parameters to set
        config: Backtest configuration
        metric: Metric to return
        
    Returns:
        Value of the specified metric
    """
    engine = BacktestEngine(config or BacktestConfig())
    result = engine.run(strategy, data, params)
    
    if hasattr(result.metrics, metric):
        value = getattr(result.metrics, metric)
        # Handle special values
        if np.isinf(value):
            return 10.0 if value > 0 else -10.0
        if np.isnan(value):
            return -10.0
        return value
    
    raise ValueError(f"Unknown metric: {metric}")


def compare_backtest_results(
    result1: BacktestResult,
    result2: BacktestResult,
    label1: str = "Strategy 1",
    label2: str = "Strategy 2"
) -> Dict[str, Any]:
    """
    Compare two backtest results.
    
    Useful for comparing human params vs ML-optimized params.
    
    Args:
        result1: First backtest result
        result2: Second backtest result
        label1: Label for first result
        label2: Label for second result
        
    Returns:
        Dictionary with comparison data
    """
    comparison = {
        'labels': [label1, label2],
        'metrics': {}
    }
    
    # Compare key metrics
    metrics_to_compare = [
        'total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
        'win_rate', 'profit_factor', 'total_trades'
    ]
    
    for metric in metrics_to_compare:
        val1 = getattr(result1.metrics, metric, None)
        val2 = getattr(result2.metrics, metric, None)
        
        if val1 is not None and val2 is not None:
            improvement = val2 - val1
            improvement_pct = (improvement / abs(val1) * 100) if val1 != 0 else 0
            
            comparison['metrics'][metric] = {
                label1: val1,
                label2: val2,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
    
    # Parameter differences
    comparison['parameter_changes'] = {}
    for key in result1.parameters:
        if key in result2.parameters:
            val1 = result1.parameters[key]
            val2 = result2.parameters[key]
            
            if val1 != val2:
                comparison['parameter_changes'][key] = {
                    label1: val1,
                    label2: val2
                }
    
    return comparison
