"""
Binance Spot WebSocket client — continuous live market data.

Thin adapter over ``realtime.binance_websocket.BinanceWebSocket`` that
speaks the ``market_data`` provider schema (rows, not ``Candle`` objects)
and exposes only what ``BinanceLiveProvider`` needs: closed-candle events
and a permanent-disconnect signal. No order/account/user-data streams are
touched — this wraps the public ``<symbol>@kline_<interval>`` stream only.
"""

import logging
from typing import Callable, Optional

import pandas as pd

from realtime.binance_websocket import BinanceWebSocket, Candle, WebSocketState

logger = logging.getLogger(__name__)


def _candle_to_row(candle: Candle) -> dict:
    """Convert a realtime.Candle into a market_data schema row."""
    return {
        "timestamp": pd.to_datetime(candle.timestamp, utc=True),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "quote_volume": pd.NA,
        "number_of_trades": pd.NA,
        "taker_buy_base": pd.NA,
        "taker_buy_quote": pd.NA,
    }


class BinanceKlineStream:
    """Streams closed 1-symbol klines from Binance over WebSocket.

    Wraps ``BinanceWebSocket`` (which already implements reconnect-with-backoff
    and closed-candle filtering) rather than reimplementing socket handling.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        ws_base_url: Optional[str] = None,
        use_testnet: bool = False,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 5.0,
    ):
        """
        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Candle interval, e.g. ``"1m"``.
            ws_base_url: Config-driven WEBSOCKET_ENDPOINT base URL.
            use_testnet: Use Binance testnet when ``ws_base_url`` is unset.
            reconnect_attempts: Max reconnection attempts before giving up.
            reconnect_delay: Delay in seconds between reconnect attempts.
        """
        self._ws = BinanceWebSocket(
            symbols=[symbol],
            interval=interval,
            use_testnet=use_testnet,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay=reconnect_delay,
            ws_base_url=ws_base_url,
        )
        self._on_candle_callbacks: list[Callable[[dict], None]] = []
        self._ws.on_candle(self._handle_candle)

    def on_candle(self, callback: Callable[[dict], None]) -> None:
        """Register a callback invoked with a schema-row dict per closed candle."""
        self._on_candle_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked once reconnection is exhausted."""
        self._ws.on_disconnect(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback invoked on every transient socket error."""
        self._ws.on_error(callback)

    def _handle_candle(self, _symbol: str, candle: Candle) -> None:
        row = _candle_to_row(candle)
        for callback in self._on_candle_callbacks:
            try:
                callback(row)
            except Exception:
                logger.exception("BinanceKlineStream candle callback failed")

    def start(self) -> None:
        """Connect and begin streaming in a background thread."""
        self._ws.start()

    def stop(self) -> None:
        """Disconnect and stop the background thread."""
        self._ws.stop()

    def is_connected(self) -> bool:
        """Whether the underlying WebSocket is currently connected."""
        return self._ws.state == WebSocketState.CONNECTED
