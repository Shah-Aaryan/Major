"""
Real-time Trading Module.

Provides real-time data streaming, live feature updates,
and paper trading capabilities for testing ML-optimized
parameters in simulated live conditions.

Components:
- BinanceWebSocket: Real-time OHLCV data streaming
- CoinGeckoClient: Real-time price data from CoinGecko API
- LiveFeatureUpdater: Real-time feature calculation
- PaperTrader: Simulated trading without real money
"""

from realtime.binance_websocket import BinanceWebSocket, WebSocketState
from realtime.live_feature_updater import LiveFeatureUpdater, LiveFeatureState
from realtime.paper_trader import PaperTrader, PaperPosition, PaperTrade
from realtime.coingecko_client import (
    CoinGeckoClient, 
    CoinGeckoConfig, 
    CoinGeckoDataFetcher,
    get_crypto_price
)

__all__ = [
    'BinanceWebSocket',
    'WebSocketState',
    'CoinGeckoClient',
    'CoinGeckoConfig',
    'CoinGeckoDataFetcher',
    'get_crypto_price',
    'LiveFeatureUpdater',
    'LiveFeatureState',
    'PaperTrader',
    'PaperPosition',
    'PaperTrade'
]
