"""
Paper Trader.

Simulates trading without real money to test
ML-optimized parameters in real-time conditions.

Tracks positions, P&L, and performance metrics
as if trading live.

NOTE: This is for RESEARCH purposes only.
No real trades are executed.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import threading
import json
import uuid

import pandas as pd
import numpy as np

from realtime.binance_websocket import Candle
from realtime.live_feature_updater import FeatureSnapshot
from strategies.base_strategy import SignalType, StrategySignal
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PaperPosition:
    """Paper trading position."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    entry_time: datetime
    quantity: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - self.current_price) * self.quantity
        return 0.0
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.quantity) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'quantity': self.quantity,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct
        }


@dataclass
class PaperTrade:
    """Completed paper trade."""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    strategy_name: str = ""
    signal_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat(),
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'strategy_name': self.strategy_name,
            'signal_confidence': self.signal_confidence
        }


class PaperTrader:
    """
    Paper trading engine for testing ML-optimized strategies.
    
    Features:
    - Simulates order execution with slippage
    - Tracks positions and P&L
    - Implements stop-loss and take-profit
    - Calculates real-time performance metrics
    - Logs all decisions for audit
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.1,
        max_positions: int = 5,
        default_stop_loss_pct: float = 0.02,
        default_take_profit_pct: float = 0.04,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Size of each position as % of capital
            max_positions: Maximum concurrent positions
            default_stop_loss_pct: Default stop loss percentage
            default_take_profit_pct: Default take profit percentage
            commission_pct: Commission per trade
            slippage_pct: Simulated slippage
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[PaperTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_trade_callbacks: List[Callable[[PaperTrade], None]] = []
        self._on_position_callbacks: List[Callable[[PaperPosition], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': initial_capital
        }
    
    def on_trade(self, callback: Callable[[PaperTrade], None]) -> None:
        """Register callback for completed trades."""
        self._on_trade_callbacks.append(callback)
    
    def on_position(self, callback: Callable[[PaperPosition], None]) -> None:
        """Register callback for position changes."""
        self._on_position_callbacks.append(callback)
    
    def process_signal(
        self,
        symbol: str,
        signal: StrategySignal,
        current_price: float,
        timestamp: datetime = None
    ) -> Optional[str]:
        """
        Process trading signal.
        
        Args:
            symbol: Trading symbol
            signal: Strategy signal
            current_price: Current price
            timestamp: Signal timestamp
            
        Returns:
            Position ID if position opened/closed, None otherwise
        """
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            # Check if we have an existing position
            existing = self.positions.get(symbol)
            
            if signal.signal_type == SignalType.BUY:
                if existing and existing.side == PositionSide.LONG:
                    # Already long, ignore
                    return None
                elif existing and existing.side == PositionSide.SHORT:
                    # Close short and open long
                    self._close_position(symbol, current_price, timestamp, "signal_reversal")
                
                # Open long position
                if len(self.positions) < self.max_positions:
                    return self._open_position(
                        symbol, PositionSide.LONG, current_price, timestamp, signal
                    )
                    
            elif signal.signal_type == SignalType.SELL:
                if existing and existing.side == PositionSide.SHORT:
                    # Already short, ignore
                    return None
                elif existing and existing.side == PositionSide.LONG:
                    # Close long and open short
                    self._close_position(symbol, current_price, timestamp, "signal_reversal")
                
                # Open short position
                if len(self.positions) < self.max_positions:
                    return self._open_position(
                        symbol, PositionSide.SHORT, current_price, timestamp, signal
                    )
                    
            elif signal.signal_type == SignalType.CLOSE:
                if existing:
                    self._close_position(symbol, current_price, timestamp, "exit_signal")
                    return existing.id
        
        return None
    
    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        timestamp: datetime,
        signal: StrategySignal
    ) -> str:
        """Open a new position."""
        # Apply slippage
        if side == PositionSide.LONG:
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)
        
        # Calculate position size
        position_value = self.cash * self.position_size_pct
        quantity = position_value / entry_price
        
        # Apply commission
        commission = position_value * self.commission_pct
        self.cash -= commission
        
        # Calculate stop loss and take profit
        if side == PositionSide.LONG:
            stop_loss = entry_price * (1 - signal.stop_loss_pct) if signal.stop_loss_pct else entry_price * (1 - self.default_stop_loss_pct)
            take_profit = entry_price * (1 + signal.take_profit_pct) if signal.take_profit_pct else entry_price * (1 + self.default_take_profit_pct)
        else:
            stop_loss = entry_price * (1 + signal.stop_loss_pct) if signal.stop_loss_pct else entry_price * (1 + self.default_stop_loss_pct)
            take_profit = entry_price * (1 - signal.take_profit_pct) if signal.take_profit_pct else entry_price * (1 - self.default_take_profit_pct)
        
        # Create position
        position = PaperPosition(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=timestamp,
            quantity=quantity,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # Notify callbacks
        for callback in self._on_position_callbacks:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")
        
        logger.info(
            f"Opened {side.value} position for {symbol} at {entry_price:.2f}, "
            f"qty: {quantity:.4f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}"
        )
        
        return position.id
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> Optional[PaperTrade]:
        """Close an existing position."""
        position = self.positions.get(symbol)
        
        if not position:
            return None
        
        # Apply slippage
        if position.side == PositionSide.LONG:
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)
        
        # Calculate P&L
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        # Apply commission
        commission = exit_price * position.quantity * self.commission_pct
        pnl -= commission
        
        pnl_pct = pnl / (position.entry_price * position.quantity) * 100
        
        # Create trade record
        trade = PaperTrade(
            id=position.id,
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=timestamp,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )
        
        self.closed_trades.append(trade)
        
        # Update cash
        self.cash += position.entry_price * position.quantity + pnl
        
        # Update statistics
        self._stats['total_trades'] += 1
        self._stats['total_pnl'] += pnl
        
        if pnl > 0:
            self._stats['winning_trades'] += 1
        else:
            self._stats['losing_trades'] += 1
        
        # Remove position
        del self.positions[symbol]
        
        # Notify callbacks
        for callback in self._on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
        
        logger.info(
            f"Closed {position.side.value} position for {symbol} at {exit_price:.2f}, "
            f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%), reason: {reason}"
        )
        
        return trade
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime = None) -> List[PaperTrade]:
        """
        Update position prices and check stop-loss/take-profit.
        
        Args:
            prices: Dict of symbol -> current price
            timestamp: Current timestamp
            
        Returns:
            List of trades closed due to SL/TP
        """
        timestamp = timestamp or datetime.now()
        closed = []
        
        with self._lock:
            for symbol, position in list(self.positions.items()):
                if symbol not in prices:
                    continue
                
                price = prices[symbol]
                position.current_price = price
                
                # Check stop loss
                should_close = False
                reason = ""
                
                if position.side == PositionSide.LONG:
                    if position.stop_loss and price <= position.stop_loss:
                        should_close = True
                        reason = "stop_loss"
                    elif position.take_profit and price >= position.take_profit:
                        should_close = True
                        reason = "take_profit"
                else:  # SHORT
                    if position.stop_loss and price >= position.stop_loss:
                        should_close = True
                        reason = "stop_loss"
                    elif position.take_profit and price <= position.take_profit:
                        should_close = True
                        reason = "take_profit"
                
                if should_close:
                    trade = self._close_position(symbol, price, timestamp, reason)
                    if trade:
                        closed.append(trade)
            
            # Update equity curve
            self._update_equity(timestamp)
        
        return closed
    
    def _update_equity(self, timestamp: datetime) -> None:
        """Update equity curve."""
        equity = self.get_equity()
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'n_positions': len(self.positions)
        })
        
        # Update max drawdown
        if equity > self._stats['peak_equity']:
            self._stats['peak_equity'] = equity
        
        drawdown = (self._stats['peak_equity'] - equity) / self._stats['peak_equity']
        if drawdown > self._stats['max_drawdown']:
            self._stats['max_drawdown'] = drawdown
    
    def get_equity(self) -> float:
        """Get current total equity."""
        positions_value = sum(
            p.entry_price * p.quantity + p.unrealized_pnl
            for p in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self.positions
    
    def get_all_positions(self) -> List[PaperPosition]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_trades(self, n: Optional[int] = None) -> List[PaperTrade]:
        """Get closed trades."""
        if n:
            return self.closed_trades[-n:]
        return self.closed_trades
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.closed_trades:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Basic metrics
        n_trades = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        losses = n_trades - wins
        
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return_pct = (self.get_equity() - self.initial_capital) / self.initial_capital * 100
        
        pnls = [t.pnl for t in self.closed_trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        std_pnl = np.std(pnls) if len(pnls) > 1 else 0
        
        # Sharpe ratio (simplified)
        sharpe = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_pnl,
            'total_return_pct': total_return_pct,
            'n_trades': n_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': wins / n_trades if n_trades > 0 else 0,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': np.mean([t.pnl_pct for t in self.closed_trades]),
            'sharpe_ratio': sharpe,
            'max_drawdown': self._stats['max_drawdown'],
            'profit_factor': profit_factor,
            'current_equity': self.get_equity()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        stats = self._stats.copy()
        stats['current_equity'] = self.get_equity()
        stats['n_open_positions'] = len(self.positions)
        stats['total_return_pct'] = (self.get_equity() - self.initial_capital) / self.initial_capital * 100
        return stats
    
    def reset(self) -> None:
        """Reset paper trader to initial state."""
        with self._lock:
            self.cash = self.initial_capital
            self.positions.clear()
            self.closed_trades.clear()
            self.equity_curve.clear()
            self._stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'max_drawdown': 0.0,
                'peak_equity': self.initial_capital
            }
        
        logger.info("Paper trader reset")
    
    def export_trades(self, filepath: str) -> None:
        """Export trades to JSON file."""
        data = {
            'trades': [t.to_dict() for t in self.closed_trades],
            'performance': self.get_performance_metrics(),
            'stats': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.closed_trades)} trades to {filepath}")


class PaperTradingSession:
    """
    Manages a complete paper trading session.
    
    Integrates WebSocket, feature updater, strategy,
    and paper trader for end-to-end simulation.
    """
    
    def __init__(
        self,
        websocket,
        feature_updater,
        strategy,
        paper_trader: PaperTrader
    ):
        """
        Initialize paper trading session.
        
        Args:
            websocket: BinanceWebSocket or SimulatedWebSocket
            feature_updater: LiveFeatureUpdater
            strategy: Trading strategy instance
            paper_trader: PaperTrader instance
        """
        self.websocket = websocket
        self.feature_updater = feature_updater
        self.strategy = strategy
        self.paper_trader = paper_trader
        
        self._running = False
        self._signal_count = 0
    
    def start(self) -> None:
        """Start paper trading session."""
        if self._running:
            return
        
        self._running = True
        
        # Register candle callback
        self.websocket.on_candle(self._on_candle)
        
        # Start WebSocket
        self.websocket.start()
        
        logger.info("Paper trading session started")
    
    def stop(self) -> None:
        """Stop paper trading session."""
        self._running = False
        self.websocket.stop()
        
        logger.info("Paper trading session stopped")
    
    def _on_candle(self, symbol: str, candle: Candle) -> None:
        """Handle new candle from WebSocket."""
        if not self._running:
            return
        
        # Update features
        snapshot = self.feature_updater.process_candle(symbol, candle)
        
        if not snapshot or not snapshot.is_valid:
            return
        
        # Get OHLCV for strategy
        ohlcv = self.feature_updater.get_ohlcv(symbol)
        
        if ohlcv.empty:
            return
        
        # Generate signals
        signals = self.strategy.generate_signals(ohlcv)
        
        if signals.empty:
            return
        
        # Get latest signal
        latest_signal = signals.iloc[-1]
        
        if hasattr(latest_signal, 'signal_type'):
            signal = latest_signal
        else:
            # Convert from signal value if needed
            signal_val = latest_signal.get('signal', 0)
            if signal_val == 1:
                signal = StrategySignal(
                    timestamp=candle.timestamp,
                    signal_type=SignalType.BUY,
                    price=candle.close,
                    confidence=0.5
                )
            elif signal_val == -1:
                signal = StrategySignal(
                    timestamp=candle.timestamp,
                    signal_type=SignalType.SELL,
                    price=candle.close,
                    confidence=0.5
                )
            else:
                return
        
        # Process signal
        self.paper_trader.process_signal(
            symbol=symbol,
            signal=signal,
            current_price=candle.close,
            timestamp=candle.timestamp
        )
        
        self._signal_count += 1
        
        # Update prices for all positions
        self.paper_trader.update_prices(
            {symbol: candle.close},
            timestamp=candle.timestamp
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            'running': self._running,
            'signals_processed': self._signal_count,
            'feature_stats': self.feature_updater.get_stats(),
            'trading_stats': self.paper_trader.get_stats(),
            'performance': self.paper_trader.get_performance_metrics()
        }
