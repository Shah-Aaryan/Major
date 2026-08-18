"""
Live Binance spot market data provider.

On ``initialize()``: fetches the latest ``WINDOW_SIZE`` candles via REST to
seed the rolling buffer, then opens a WebSocket kline stream. Each closed
candle appends to the buffer and the oldest candle is dropped, keeping
exactly ``WINDOW_SIZE`` candles at all times.

If the WebSocket connection fails or drops permanently, this provider
automatically falls back to ``DatasetProvider`` replay (per the configured
``CSV_PATH``) as its error-recovery strategy, logging the failure and
recording the transition as an audit event.
"""

import logging
import queue
from collections import deque
from typing import Any, Deque, Optional

import pandas as pd

from audit.audit_logger import AuditEventType
from market_data.dataset_provider import DatasetProvider
from market_data.provider import MarketDataProvider, SCHEMA_COLUMNS, normalize_candle_frame
from market_data.rest_client import BinanceRESTClient
from market_data.websocket_client import BinanceKlineStream

logger = logging.getLogger(__name__)


class BinanceLiveProvider(MarketDataProvider):
    """Live Binance spot market data: REST seed + WebSocket streaming.

    Falls back to ``DatasetProvider`` replay automatically if the
    WebSocket connection is permanently lost.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        window_size: int,
        rest_endpoint: str,
        ws_endpoint: str,
        fallback_csv_path: Optional[str] = None,
        use_testnet: bool = False,
        rest_timeout: float = 10.0,
        ws_reconnect_attempts: int = 5,
        ws_reconnect_delay: float = 5.0,
        audit_logger: Optional[Any] = None,
    ):
        """
        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Candle interval, e.g. ``"1m"``.
            window_size: Rolling window size to seed and maintain.
            rest_endpoint: Binance klines REST endpoint URL.
            ws_endpoint: Binance WebSocket base URL.
            fallback_csv_path: CSV path used for automatic fallback replay
                if the WebSocket connection is permanently lost. If unset,
                the provider simply stops yielding candles on disconnect.
            use_testnet: Use Binance testnet when connecting.
            rest_timeout: REST request timeout in seconds.
            ws_reconnect_attempts: Max WebSocket reconnection attempts.
            ws_reconnect_delay: Delay in seconds between reconnect attempts.
            audit_logger: Optional ``audit.audit_logger.AuditLogger``
                instance used to record the fallback transition.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.fallback_csv_path = fallback_csv_path
        self._audit_logger = audit_logger

        self._rest_client = BinanceRESTClient(rest_endpoint, timeout=rest_timeout)
        self._stream = BinanceKlineStream(
            symbol=symbol,
            interval=interval,
            ws_base_url=ws_endpoint,
            use_testnet=use_testnet,
            reconnect_attempts=ws_reconnect_attempts,
            reconnect_delay=ws_reconnect_delay,
        )

        self._window: Deque[pd.Series] = deque(maxlen=window_size)
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._fallback: Optional[DatasetProvider] = None
        self._using_fallback = False
        self._shutdown_flag = False

    def initialize(self) -> None:
        """Seed the window via REST, then connect the WebSocket stream."""
        seed_df = self._rest_client.get_klines(self.symbol, self.interval, limit=self.window_size)
        seed_df = normalize_candle_frame(seed_df)
        for _, row in seed_df.iterrows():
            self._window.append(row)

        logger.info(
            "BinanceLiveProvider seeded %d candles for %s@%s via REST",
            len(self._window), self.symbol, self.interval,
        )

        self._stream.on_candle(self._on_candle)
        self._stream.on_disconnect(self._on_disconnect)
        self._stream.on_error(self._on_error)
        self._stream.start()

    def _on_candle(self, row: dict) -> None:
        self._queue.put(row)

    def _on_error(self, error: Exception) -> None:
        logger.warning("BinanceLiveProvider transient websocket error: %s", error)

    def _on_disconnect(self) -> None:
        logger.error(
            "BinanceLiveProvider websocket connection permanently lost for %s; "
            "falling back to dataset replay mode", self.symbol,
        )
        self._activate_fallback("websocket_disconnected")

    def _activate_fallback(self, reason: str) -> None:
        """Switch to dataset replay mode and record the transition."""
        if self._using_fallback:
            return
        self._using_fallback = True

        if self._audit_logger is not None:
            try:
                self._audit_logger.log_event(
                    AuditEventType.MARKET_DATA_FALLBACK,
                    {
                        "reason": reason,
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "fallback_csv_path": self.fallback_csv_path,
                    },
                    explanation=(
                        f"Live Binance market data unavailable ({reason}); "
                        f"switched to dataset replay mode"
                    ),
                )
            except Exception:
                logger.exception("Failed to log market data fallback audit event")

        if not self.fallback_csv_path:
            logger.error(
                "No fallback_csv_path configured; BinanceLiveProvider will "
                "stop yielding candles after disconnect"
            )
            return

        self._fallback = DatasetProvider(
            self.fallback_csv_path, window_size=self.window_size, symbol=self.symbol
        )
        self._fallback.initialize()

    def get_next_candle(self) -> Optional[pd.DataFrame]:
        """Return the next closed candle, blocking until the WebSocket
        delivers one (or the fallback dataset provider is consulted).
        """
        while not self._using_fallback and not self._shutdown_flag:
            try:
                row = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            candle_series = pd.Series(row)
            self._window.append(candle_series)
            return candle_series.to_frame().T.reset_index(drop=True)

        if self._using_fallback and self._fallback is not None:
            candle = self._fallback.get_next_candle()
            if candle is not None:
                self._window.append(candle.iloc[0])
            return candle

        return None

    def get_latest_window(self) -> pd.DataFrame:
        """Return the current rolling window (up to ``window_size`` candles)."""
        if not self._window:
            return pd.DataFrame(columns=SCHEMA_COLUMNS)
        return pd.DataFrame(list(self._window)).reset_index(drop=True)

    def reset(self) -> None:
        """Clear the rolling window and pending candle queue."""
        self._window.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._fallback is not None:
            self._fallback.reset()

    def is_finished(self) -> bool:
        """False for a live stream; True only if the fallback dataset is exhausted."""
        if self._using_fallback and self._fallback is not None:
            return self._fallback.is_finished()
        return self._shutdown_flag

    def shutdown(self) -> None:
        """Stop the WebSocket and release the fallback provider, if any."""
        self._shutdown_flag = True
        self._stream.stop()
        if self._fallback is not None:
            self._fallback.shutdown()
        logger.info("BinanceLiveProvider shutdown (symbol=%s)", self.symbol)
