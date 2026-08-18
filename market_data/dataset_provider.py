"""
Historical dataset replay provider.

Replays a historical OHLCV CSV file exactly like a live feed — one closed
candle at a time, in chronological order, with a rolling window and zero
future leakage (the provider never exposes any row beyond the one just
"arrived").
"""

import logging
from collections import deque
from typing import Deque, Optional

import pandas as pd

from data.loader import DataLoader
from market_data.provider import MarketDataProvider, normalize_candle_frame

logger = logging.getLogger(__name__)

# Maps DataLoader's internal column names onto the shared provider schema.
_DATASET_COLUMN_ALIASES = {
    "trades_count": "number_of_trades",
    "taker_buy_volume": "taker_buy_base",
    "taker_buy_quote_volume": "taker_buy_quote",
}


class DatasetProvider(MarketDataProvider):
    """Replays a historical CSV file candle-by-candle.

    Deterministic: given the same CSV and window size, ``get_next_candle()``
    always yields candles in the same order. Never looks ahead — the
    rolling window returned by ``get_latest_window()`` only ever contains
    candles already emitted by ``get_next_candle()``.
    """

    def __init__(self, csv_path: str, window_size: int = 500, symbol: str = "BTCUSDT"):
        """
        Args:
            csv_path: Path to the historical OHLCV CSV file.
            window_size: Maximum number of candles kept in the rolling window.
            symbol: Symbol label attached to loaded data (informational only).
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.csv_path = csv_path
        self.window_size = window_size
        self.symbol = symbol

        self._df: Optional[pd.DataFrame] = None
        self._cursor: int = 0
        self._window: Deque[pd.Series] = deque(maxlen=window_size)

    def initialize(self) -> None:
        """Load and normalize the CSV, resetting the replay cursor to 0."""
        loader = DataLoader(data_dir=".")
        raw = loader.load_csv(self.csv_path)

        raw = raw.reset_index()
        raw = raw.rename(columns=_DATASET_COLUMN_ALIASES)

        self._df = normalize_candle_frame(raw)
        self._cursor = 0
        self._window.clear()

        logger.info(
            "DatasetProvider initialized: %s (%d candles) window_size=%d",
            self.csv_path, len(self._df), self.window_size,
        )

    def get_next_candle(self) -> Optional[pd.DataFrame]:
        """Emit the next row of the CSV and advance the rolling window."""
        if self._df is None:
            raise RuntimeError("DatasetProvider.initialize() must be called first")

        if self.is_finished():
            return None

        row = self._df.iloc[self._cursor]
        self._cursor += 1
        self._window.append(row)

        return row.to_frame().T.reset_index(drop=True)

    def get_latest_window(self) -> pd.DataFrame:
        """Return the rolling window of candles emitted so far."""
        if not self._window:
            return pd.DataFrame(columns=self._df.columns if self._df is not None else [])
        return pd.DataFrame(list(self._window)).reset_index(drop=True)

    def reset(self) -> None:
        """Rewind the replay cursor to the beginning and clear the window."""
        self._cursor = 0
        self._window.clear()

    def is_finished(self) -> bool:
        """True once every row of the CSV has been emitted."""
        if self._df is None:
            return True
        return self._cursor >= len(self._df)

    def shutdown(self) -> None:
        """No held resources; provided for interface symmetry."""
        logger.info(
            "DatasetProvider shutdown (cursor=%d/%d)",
            self._cursor, len(self._df) if self._df is not None else 0,
        )
