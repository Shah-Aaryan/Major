"""
Binance WebSocket Client.

Provides real-time OHLCV data streaming from Binance
for paper trading and live feature calculation.

NOTE: This is for RESEARCH purposes only.
No real trades are executed.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import threading

import pandas as pd
import numpy as np

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    
try:
    from binance import Client, ThreadedWebsocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class WebSocketState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'is_closed': self.is_closed
        }


@dataclass
class TickerUpdate:
    """Real-time ticker update."""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    price_change_24h: float
    
    
class BinanceWebSocket:
    """
    Real-time data streaming from Binance.
    
    Connects to Binance WebSocket API to receive
    real-time OHLCV candles for paper trading.
    """
    
    # Binance WebSocket endpoints
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    WS_TESTNET_URL = "wss://testnet.binance.vision/ws"
    
    def __init__(
        self,
        symbols: List[str],
        interval: str = "1m",
        use_testnet: bool = True,
        max_candles: int = 1000,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize WebSocket client.
        
        Args:
            symbols: List of symbols to stream (e.g., ['BTCUSDT', 'ETHUSDT'])
            interval: Candle interval (1m, 5m, 15m, etc.)
            use_testnet: Use Binance testnet (safer for research)
            max_candles: Maximum candles to keep in memory per symbol
            reconnect_attempts: Max reconnection attempts
            reconnect_delay: Delay between reconnect attempts
        """
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.use_testnet = use_testnet
        self.max_candles = max_candles
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # State
        self.state = WebSocketState.DISCONNECTED
        self._candle_buffer: Dict[str, deque] = {
            s: deque(maxlen=max_candles) for s in self.symbols
        }
        self._current_candle: Dict[str, Optional[Candle]] = {
            s: None for s in self.symbols
        }
        
        # Callbacks
        self._on_candle_callbacks: List[Callable[[str, Candle], None]] = []
        self._on_ticker_callbacks: List[Callable[[TickerUpdate], None]] = []
        self._on_error_callbacks: List[Callable[[Exception], None]] = []
        
        # Threading
        self._ws = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Statistics
        self._stats = {
            'messages_received': 0,
            'candles_received': 0,
            'errors': 0,
            'reconnects': 0,
            'start_time': None
        }
    
    def on_candle(self, callback: Callable[[str, Candle], None]) -> None:
        """Register callback for new candles."""
        self._on_candle_callbacks.append(callback)
    
    def on_ticker(self, callback: Callable[[TickerUpdate], None]) -> None:
        """Register callback for ticker updates."""
        self._on_ticker_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for errors."""
        self._on_error_callbacks.append(callback)
    
    def start(self) -> None:
        """Start WebSocket connection in background thread."""
        if self._running:
            logger.warning("WebSocket already running")
            return
        
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not installed. Install with: pip install websockets")
            return
        
        self._running = True
        self._stats['start_time'] = datetime.now()
        
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"Started WebSocket for {self.symbols}")
    
    def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=5.0)
        
        self.state = WebSocketState.DISCONNECTED
        logger.info("Stopped WebSocket")
    
    def _run_async_loop(self) -> None:
        """Run async event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
            self._notify_error(e)
        finally:
            self._loop.close()
    
    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and listen for messages."""
        attempt = 0
        
        while self._running and attempt < self.reconnect_attempts:
            try:
                self.state = WebSocketState.CONNECTING
                
                # Build stream URL
                streams = "/".join([f"{s}@kline_{self.interval}" for s in self.symbols])
                url = f"{self.WS_TESTNET_URL if self.use_testnet else self.WS_BASE_URL}/{streams}"
                
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    self.state = WebSocketState.CONNECTED
                    attempt = 0  # Reset on successful connection
                    
                    logger.info(f"Connected to Binance WebSocket")
                    
                    while self._running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            self._process_message(message)
                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            await ws.ping()
                            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self._notify_error(e)
                self.state = WebSocketState.RECONNECTING
                attempt += 1
                self._stats['reconnects'] += 1
                
                if self._running and attempt < self.reconnect_attempts:
                    await asyncio.sleep(self.reconnect_delay)
        
        if self._running:
            self.state = WebSocketState.ERROR
            logger.error("Max reconnection attempts reached")
    
    def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            self._stats['messages_received'] += 1
            
            if 'k' in data:  # Kline/Candle data
                self._process_kline(data)
            elif 'e' in data and data['e'] == '24hrTicker':
                self._process_ticker(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            self._stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self._stats['errors'] += 1
    
    def _process_kline(self, data: Dict[str, Any]) -> None:
        """Process kline/candle data."""
        symbol = data['s'].lower()
        kline = data['k']
        
        candle = Candle(
            timestamp=datetime.fromtimestamp(kline['t'] / 1000),
            open=float(kline['o']),
            high=float(kline['h']),
            low=float(kline['l']),
            close=float(kline['c']),
            volume=float(kline['v']),
            is_closed=kline['x']
        )
        
        self._current_candle[symbol] = candle
        
        if candle.is_closed:
            self._candle_buffer[symbol].append(candle)
            self._stats['candles_received'] += 1
            
            # Notify callbacks
            for callback in self._on_candle_callbacks:
                try:
                    callback(symbol, candle)
                except Exception as e:
                    logger.error(f"Candle callback error: {e}")
    
    def _process_ticker(self, data: Dict[str, Any]) -> None:
        """Process ticker data."""
        ticker = TickerUpdate(
            symbol=data['s'].lower(),
            timestamp=datetime.fromtimestamp(data['E'] / 1000),
            price=float(data['c']),
            volume_24h=float(data['v']),
            price_change_24h=float(data['P'])
        )
        
        for callback in self._on_ticker_callbacks:
            try:
                callback(ticker)
            except Exception as e:
                logger.error(f"Ticker callback error: {e}")
    
    def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        self._stats['errors'] += 1
        for callback in self._on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def get_candles(self, symbol: str) -> pd.DataFrame:
        """Get historical candles as DataFrame."""
        symbol = symbol.lower()
        
        if symbol not in self._candle_buffer:
            return pd.DataFrame()
        
        candles = list(self._candle_buffer[symbol])
        
        if not candles:
            return pd.DataFrame()
        
        df = pd.DataFrame([c.to_dict() for c in candles])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_current_candle(self, symbol: str) -> Optional[Candle]:
        """Get current (not yet closed) candle."""
        return self._current_candle.get(symbol.lower())
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol."""
        candle = self._current_candle.get(symbol.lower())
        return candle.close if candle else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        stats = self._stats.copy()
        stats['state'] = self.state.value
        stats['symbols'] = self.symbols
        stats['uptime_seconds'] = (
            (datetime.now() - stats['start_time']).total_seconds()
            if stats['start_time'] else 0
        )
        return stats


class SimulatedWebSocket:
    """
    Simulated WebSocket for testing without real connection.
    
    Replays historical data as if it were streaming in real-time.
    Useful for testing paper trading logic offline.
    """
    
    def __init__(
        self,
        historical_data: pd.DataFrame,
        symbol: str = 'BTCUSDT',
        replay_speed: float = 1.0
    ):
        """
        Initialize simulated WebSocket.
        
        Args:
            historical_data: DataFrame with OHLCV data
            symbol: Symbol name
            replay_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        """
        self.data = historical_data
        self.symbol = symbol.lower()
        self.replay_speed = replay_speed
        
        self.state = WebSocketState.DISCONNECTED
        self._candle_buffer: deque = deque(maxlen=1000)
        self._current_idx = 0
        
        self._on_candle_callbacks: List[Callable] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def on_candle(self, callback: Callable[[str, Candle], None]) -> None:
        """Register candle callback."""
        self._on_candle_callbacks.append(callback)
    
    def start(self) -> None:
        """Start replay."""
        if self._running:
            return
        
        self._running = True
        self.state = WebSocketState.CONNECTED
        self._current_idx = 0
        
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()
        
        logger.info("Started simulated WebSocket")
    
    def stop(self) -> None:
        """Stop replay."""
        self._running = False
        self.state = WebSocketState.DISCONNECTED
        
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _replay_loop(self) -> None:
        """Replay historical data."""
        while self._running and self._current_idx < len(self.data):
            row = self.data.iloc[self._current_idx]
            
            candle = Candle(
                timestamp=row.name if isinstance(row.name, datetime) else datetime.now(),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                is_closed=True
            )
            
            self._candle_buffer.append(candle)
            
            for callback in self._on_candle_callbacks:
                try:
                    callback(self.symbol, candle)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            self._current_idx += 1
            
            # Simulate time passing
            time.sleep(1.0 / self.replay_speed)
        
        self.state = WebSocketState.DISCONNECTED
        logger.info("Simulated WebSocket replay complete")
    
    def get_candles(self, symbol: str = None) -> pd.DataFrame:
        """Get replayed candles."""
        if not self._candle_buffer:
            return pd.DataFrame()
        
        return pd.DataFrame([c.to_dict() for c in self._candle_buffer])
    
    def get_progress(self) -> float:
        """Get replay progress (0-1)."""
        if len(self.data) == 0:
            return 1.0
        return self._current_idx / len(self.data)
