"""
Unified market data provider layer.

Exposes a single ``MarketDataProvider`` interface with two interchangeable
implementations — ``DatasetProvider`` (historical CSV replay) and
``BinanceLiveProvider`` (live Binance spot market data). Downstream modules
(indicators, optimizers, strategies) depend only on the interface and never
know which concrete source is in use.
"""

from market_data.provider import MarketDataProvider, REQUIRED_COLUMNS, OPTIONAL_COLUMNS
from market_data.dataset_provider import DatasetProvider
from market_data.binance_provider import BinanceLiveProvider
from market_data.factory import create_provider

__all__ = [
    "MarketDataProvider",
    "DatasetProvider",
    "BinanceLiveProvider",
    "create_provider",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
]
