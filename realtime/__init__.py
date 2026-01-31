"""
Real-time Trading Module.

Provides real-time data streaming, live feature updates,
and paper trading capabilities for testing ML-optimized
parameters in simulated live conditions.

Components:
- BinanceWebSocket: Real-time OHLCV data streaming
- LiveFeatureUpdater: Real-time feature calculation
- PaperTrader: Simulated trading without real money
"""

from realtime.binance_websocket import BinanceWebSocket, WebSocketState
from realtime.live_feature_updater import LiveFeatureUpdater, LiveFeatureState
from realtime.paper_trader import PaperTrader, PaperPosition, PaperTrade

__all__ = [
    'BinanceWebSocket',
    'WebSocketState',
    'LiveFeatureUpdater',
    'LiveFeatureState',
    'PaperTrader',
    'PaperPosition',
    'PaperTrade'
]
