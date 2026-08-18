"""
Market data provider interface.

Defines the contract every market data source (historical dataset replay,
live Binance spot market) must implement, plus the shared OHLCV DataFrame
schema all providers return. No downstream module (rolling window engine,
indicator engine, optimizers, strategies) should ever branch on which
concrete provider is in use.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Columns every provider MUST populate.
REQUIRED_COLUMNS: List[str] = ["timestamp", "open", "high", "low", "close", "volume"]

# Columns a provider MAY populate when the underlying source has them.
OPTIONAL_COLUMNS: List[str] = [
    "quote_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
]

SCHEMA_COLUMNS: List[str] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def validate_candle_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if ``df`` does not satisfy the required OHLCV schema.

    Args:
        df: Candidate candle DataFrame.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Candle DataFrame missing required columns: {sorted(missing)}")


def normalize_candle_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw OHLCV DataFrame into the canonical provider schema.

    Ensures every optional column exists (filled with NaN when absent),
    orders columns deterministically (required first, then optional), and
    sorts rows chronologically by ``timestamp``. This is what lets
    ``DatasetProvider`` and ``BinanceLiveProvider`` return byte-for-byte
    identical schemas regardless of source.

    Args:
        df: Raw OHLCV DataFrame containing at least the required columns.

    Returns:
        A new DataFrame with columns ``REQUIRED_COLUMNS + OPTIONAL_COLUMNS``,
        sorted by timestamp with a fresh RangeIndex.
    """
    validate_candle_schema(df)

    out = df.copy()
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[SCHEMA_COLUMNS]
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


class MarketDataProvider(ABC):
    """Abstract interface shared by every market data source.

    Concrete implementations (``DatasetProvider``, ``BinanceLiveProvider``)
    must return DataFrames matching ``REQUIRED_COLUMNS``/``OPTIONAL_COLUMNS``
    from both ``get_next_candle()`` and ``get_latest_window()`` so that no
    downstream code needs to know which source produced the data.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the provider to start serving candles.

        For ``DatasetProvider`` this loads and validates the CSV. For
        ``BinanceLiveProvider`` this fetches the initial REST window and
        opens the WebSocket connection.
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_candle(self) -> Optional[pd.DataFrame]:
        """Return the next closed candle as a single-row DataFrame.

        Returns:
            A one-row DataFrame matching the provider schema, or ``None``
            when no further candles are available (dataset exhausted, or
            provider shut down).
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_window(self) -> pd.DataFrame:
        """Return the current rolling window of candles.

        Returns:
            A DataFrame of up to ``WINDOW_SIZE`` rows, oldest first,
            matching the provider schema.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the provider to its initial state (for dataset providers,
        rewinds the replay cursor; for live providers, clears the buffer).
        """
        raise NotImplementedError

    @abstractmethod
    def is_finished(self) -> bool:
        """Whether the provider has no more candles to serve.

        Always ``False`` for a live provider unless it has been shut down
        or fully drained a fallback dataset provider.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Release any held resources (open sockets, threads, file handles)."""
        raise NotImplementedError
