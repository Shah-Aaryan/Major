"""
Live Feature Updater.

Calculates technical indicators in real-time as new
candles arrive from the WebSocket stream.

Maintains a rolling window of features for immediate
use in trading decisions.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import threading

import pandas as pd
import numpy as np

from realtime.binance_websocket import Candle
from features.feature_engine import FeatureEngine
from config.settings import FeatureConfig

logger = logging.getLogger(__name__)


class LiveFeatureState(Enum):
    """Live feature updater state."""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    READY = "ready"
    ERROR = "error"


@dataclass
class FeatureSnapshot:
    """Snapshot of all features at a point in time."""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    is_valid: bool = True
    warmup_progress: float = 1.0


class LiveFeatureUpdater:
    """
    Real-time feature calculator.
    
    Maintains rolling OHLCV window and recalculates features
    as each new candle arrives from the WebSocket.
    """
    
    def __init__(
        self,
        symbols: List[str],
        feature_config: Optional[FeatureConfig] = None,
        min_warmup_candles: int = 200,
        max_window_size: int = 500
    ):
        """
        Initialize live feature updater.
        
        Args:
            symbols: List of symbols to track
            feature_config: Feature configuration (uses default if None)
            min_warmup_candles: Minimum candles needed before features are valid
            max_window_size: Maximum candles to keep in memory
        """
        self.symbols = [s.lower() for s in symbols]
        self.feature_config = feature_config or FeatureConfig()
        self.min_warmup_candles = min_warmup_candles
        self.max_window_size = max_window_size
        
        # Data buffers per symbol
        self._ohlcv_buffer: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.symbols}
        self._feature_buffer: Dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in self.symbols}
        
        # State per symbol
        self._state: Dict[str, LiveFeatureState] = {s: LiveFeatureState.IDLE for s in self.symbols}
        self._warmup_count: Dict[str, int] = {s: 0 for s in self.symbols}
        
        # Feature engine
        self._feature_engine = FeatureEngine(self.feature_config)
        
        # Callbacks
        self._on_feature_callbacks: List[Callable[[FeatureSnapshot], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'updates_processed': 0,
            'features_generated': 0,
            'errors': 0
        }
    
    def on_feature_update(self, callback: Callable[[FeatureSnapshot], None]) -> None:
        """Register callback for feature updates."""
        self._on_feature_callbacks.append(callback)
    
    def process_candle(self, symbol: str, candle: Candle) -> Optional[FeatureSnapshot]:
        """
        Process new candle and update features.
        
        Args:
            symbol: Symbol the candle is for
            candle: New closed candle
            
        Returns:
            FeatureSnapshot if features were calculated
        """
        symbol = symbol.lower()
        
        if symbol not in self.symbols:
            logger.warning(f"Unknown symbol: {symbol}")
            return None
        
        with self._lock:
            try:
                # Add candle to buffer
                self._add_candle(symbol, candle)
                
                # Update state
                self._update_state(symbol)
                
                # Calculate features if we have enough data
                if self._state[symbol] in [LiveFeatureState.WARMING_UP, LiveFeatureState.READY]:
                    return self._calculate_features(symbol)
                
            except Exception as e:
                logger.error(f"Error processing candle for {symbol}: {e}")
                self._state[symbol] = LiveFeatureState.ERROR
                self._stats['errors'] += 1
        
        return None
    
    def _add_candle(self, symbol: str, candle: Candle) -> None:
        """Add candle to OHLCV buffer."""
        new_row = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        }])
        
        if self._ohlcv_buffer[symbol].empty:
            new_row.set_index('timestamp', inplace=True)
            self._ohlcv_buffer[symbol] = new_row
        else:
            new_row.set_index('timestamp', inplace=True)
            self._ohlcv_buffer[symbol] = pd.concat([
                self._ohlcv_buffer[symbol],
                new_row
            ])
        
        # Trim to max window size
        if len(self._ohlcv_buffer[symbol]) > self.max_window_size:
            self._ohlcv_buffer[symbol] = self._ohlcv_buffer[symbol].iloc[-self.max_window_size:]
        
        self._warmup_count[symbol] = len(self._ohlcv_buffer[symbol])
        self._stats['updates_processed'] += 1
    
    def _update_state(self, symbol: str) -> None:
        """Update warmup state for symbol."""
        count = self._warmup_count[symbol]
        
        if count == 0:
            self._state[symbol] = LiveFeatureState.IDLE
        elif count < self.min_warmup_candles:
            self._state[symbol] = LiveFeatureState.WARMING_UP
        else:
            self._state[symbol] = LiveFeatureState.READY
    
    def _calculate_features(self, symbol: str) -> FeatureSnapshot:
        """Calculate all features for symbol."""
        ohlcv = self._ohlcv_buffer[symbol].copy()
        
        # Generate features using feature engine
        features_df = self._feature_engine.generate_features(ohlcv)
        
        # Get latest feature row
        if features_df.empty:
            return FeatureSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                features={},
                is_valid=False,
                warmup_progress=self._warmup_count[symbol] / self.min_warmup_candles
            )
        
        latest_features = features_df.iloc[-1].to_dict()
        
        # Remove NaN values
        latest_features = {k: v for k, v in latest_features.items() if pd.notna(v)}
        
        # Store features
        self._feature_buffer[symbol] = features_df
        self._stats['features_generated'] += 1
        
        # Create snapshot
        snapshot = FeatureSnapshot(
            timestamp=ohlcv.index[-1],
            symbol=symbol,
            features=latest_features,
            is_valid=self._state[symbol] == LiveFeatureState.READY,
            warmup_progress=min(1.0, self._warmup_count[symbol] / self.min_warmup_candles)
        )
        
        # Notify callbacks
        for callback in self._on_feature_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Feature callback error: {e}")
        
        return snapshot
    
    def get_features(self, symbol: str, n_bars: int = 1) -> pd.DataFrame:
        """
        Get recent features for symbol.
        
        Args:
            symbol: Symbol to get features for
            n_bars: Number of most recent bars
            
        Returns:
            DataFrame with features
        """
        symbol = symbol.lower()
        
        with self._lock:
            if symbol not in self._feature_buffer:
                return pd.DataFrame()
            
            df = self._feature_buffer[symbol]
            
            if df.empty:
                return pd.DataFrame()
            
            return df.iloc[-n_bars:].copy()
    
    def get_latest_features(self, symbol: str) -> Dict[str, float]:
        """Get most recent features as dict."""
        df = self.get_features(symbol, n_bars=1)
        
        if df.empty:
            return {}
        
        return df.iloc[-1].to_dict()
    
    def get_ohlcv(self, symbol: str, n_bars: Optional[int] = None) -> pd.DataFrame:
        """Get OHLCV data for symbol."""
        symbol = symbol.lower()
        
        with self._lock:
            if symbol not in self._ohlcv_buffer:
                return pd.DataFrame()
            
            df = self._ohlcv_buffer[symbol]
            
            if n_bars:
                return df.iloc[-n_bars:].copy()
            
            return df.copy()
    
    def get_state(self, symbol: str) -> LiveFeatureState:
        """Get current state for symbol."""
        return self._state.get(symbol.lower(), LiveFeatureState.IDLE)
    
    def get_warmup_progress(self, symbol: str) -> float:
        """Get warmup progress (0-1) for symbol."""
        count = self._warmup_count.get(symbol.lower(), 0)
        return min(1.0, count / self.min_warmup_candles)
    
    def is_ready(self, symbol: str) -> bool:
        """Check if features are ready for trading."""
        return self._state.get(symbol.lower()) == LiveFeatureState.READY
    
    def load_historical_data(self, symbol: str, ohlcv: pd.DataFrame) -> None:
        """
        Pre-load historical data for faster warmup.
        
        Args:
            symbol: Symbol to load data for
            ohlcv: Historical OHLCV DataFrame
        """
        symbol = symbol.lower()
        
        if symbol not in self.symbols:
            logger.warning(f"Unknown symbol: {symbol}")
            return
        
        with self._lock:
            # Trim to max window
            if len(ohlcv) > self.max_window_size:
                ohlcv = ohlcv.iloc[-self.max_window_size:]
            
            self._ohlcv_buffer[symbol] = ohlcv.copy()
            self._warmup_count[symbol] = len(ohlcv)
            self._update_state(symbol)
            
            # Pre-calculate features
            if self._warmup_count[symbol] >= self.min_warmup_candles:
                self._calculate_features(symbol)
        
        logger.info(f"Loaded {len(ohlcv)} historical candles for {symbol}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get updater statistics."""
        stats = self._stats.copy()
        stats['symbols'] = {
            s: {
                'state': self._state[s].value,
                'warmup_progress': self.get_warmup_progress(s),
                'candle_count': self._warmup_count[s]
            }
            for s in self.symbols
        }
        return stats
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset buffers and state.
        
        Args:
            symbol: Specific symbol to reset, or None for all
        """
        with self._lock:
            if symbol:
                symbol = symbol.lower()
                if symbol in self.symbols:
                    self._ohlcv_buffer[symbol] = pd.DataFrame()
                    self._feature_buffer[symbol] = pd.DataFrame()
                    self._state[symbol] = LiveFeatureState.IDLE
                    self._warmup_count[symbol] = 0
            else:
                for s in self.symbols:
                    self._ohlcv_buffer[s] = pd.DataFrame()
                    self._feature_buffer[s] = pd.DataFrame()
                    self._state[s] = LiveFeatureState.IDLE
                    self._warmup_count[s] = 0
        
        logger.info(f"Reset {'all symbols' if not symbol else symbol}")


class MultiTimeframeUpdater:
    """
    Manages feature updates across multiple timeframes.
    
    Useful when strategies need features from different
    intervals (e.g., 1m, 5m, 15m simultaneously).
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str] = ['1m', '5m', '15m'],
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize multi-timeframe updater.
        
        Args:
            symbols: Symbols to track
            timeframes: Timeframes to maintain
            feature_config: Feature configuration
        """
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Create updater for each timeframe
        self.updaters: Dict[str, LiveFeatureUpdater] = {}
        
        for tf in timeframes:
            self.updaters[tf] = LiveFeatureUpdater(
                symbols=symbols,
                feature_config=feature_config,
                min_warmup_candles=200
            )
        
        # Candle aggregation buffers
        self._aggregation_buffer: Dict[str, Dict[str, List[Candle]]] = {
            s: {tf: [] for tf in timeframes[1:]} for s in symbols
        }
        
        # Timeframe mappings (in minutes)
        self._tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }
    
    def process_1m_candle(self, symbol: str, candle: Candle) -> Dict[str, Optional[FeatureSnapshot]]:
        """
        Process 1-minute candle and aggregate to higher timeframes.
        
        Args:
            symbol: Symbol
            candle: 1-minute candle
            
        Returns:
            Dict of snapshots for each timeframe that was updated
        """
        symbol = symbol.lower()
        results = {}
        
        # Always update 1m
        results['1m'] = self.updaters['1m'].process_candle(symbol, candle)
        
        # Aggregate to higher timeframes
        for tf in self.timeframes[1:]:
            buffer = self._aggregation_buffer[symbol][tf]
            buffer.append(candle)
            
            # Check if we have enough candles to form a higher TF candle
            tf_size = self._tf_minutes[tf]
            
            if len(buffer) >= tf_size:
                # Aggregate candles
                agg_candle = self._aggregate_candles(buffer[:tf_size])
                self._aggregation_buffer[symbol][tf] = buffer[tf_size:]
                
                # Process aggregated candle
                results[tf] = self.updaters[tf].process_candle(symbol, agg_candle)
            else:
                results[tf] = None
        
        return results
    
    def _aggregate_candles(self, candles: List[Candle]) -> Candle:
        """Aggregate multiple candles into one."""
        return Candle(
            timestamp=candles[0].timestamp,
            open=candles[0].open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
            close=candles[-1].close,
            volume=sum(c.volume for c in candles),
            is_closed=True
        )
    
    def get_features(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """Get latest features for symbol at specific timeframe."""
        if timeframe not in self.updaters:
            return {}
        
        return self.updaters[timeframe].get_latest_features(symbol)
    
    def get_all_features(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Get latest features for symbol across all timeframes."""
        return {
            tf: self.updaters[tf].get_latest_features(symbol)
            for tf in self.timeframes
        }
    
    def is_ready(self, symbol: str) -> bool:
        """Check if all timeframes are ready."""
        return all(
            self.updaters[tf].is_ready(symbol)
            for tf in self.timeframes
        )
