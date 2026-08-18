"""
Binance Spot REST client — initialization-only.

Used exactly once per live session, to seed the rolling window with the
most recent ``WINDOW_SIZE`` closed candles before the WebSocket takes over.
No trading, account, margin, futures, or withdrawal endpoints are touched;
this client only ever calls ``GET /api/v3/klines``.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Index positions in each row of the Binance klines REST response.
# https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
_KLINE_OPEN_TIME = 0
_KLINE_OPEN = 1
_KLINE_HIGH = 2
_KLINE_LOW = 3
_KLINE_CLOSE = 4
_KLINE_VOLUME = 5
_KLINE_QUOTE_VOLUME = 7
_KLINE_NUMBER_OF_TRADES = 8
_KLINE_TAKER_BUY_BASE = 9
_KLINE_TAKER_BUY_QUOTE = 10


class BinanceRESTClient:
    """Thin wrapper around Binance's public klines REST endpoint."""

    def __init__(self, endpoint: str = "https://api.binance.com/api/v3/klines", timeout: float = 10.0):
        """
        Args:
            endpoint: Full URL of the klines REST endpoint.
            timeout: Request timeout in seconds.
        """
        self.endpoint = endpoint
        self.timeout = timeout

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        """Fetch the most recent ``limit`` klines for ``symbol``.

        Args:
            symbol: Trading pair, e.g. ``"BTCUSDT"``.
            interval: Candle interval, e.g. ``"1m"``.
            limit: Number of candles to fetch (Binance max is 1000).

        Returns:
            DataFrame with columns
            ``timestamp, open, high, low, close, volume, quote_volume,
            number_of_trades, taker_buy_base, taker_buy_quote``, oldest first.

        Raises:
            requests.RequestException: On network failure or non-2xx response.
            ValueError: If the response body is not a well-formed kline array.
        """
        params: Dict[str, str] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": str(limit),
        }

        logger.info("Fetching %d klines for %s@%s from %s", limit, symbol, interval, self.endpoint)
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        raw: List[List[Any]] = response.json()
        if not isinstance(raw, list):
            raise ValueError(f"Unexpected klines response shape: {type(raw)}")

        return self._parse_klines(raw)

    @staticmethod
    def _parse_klines(raw: List[List[Any]]) -> pd.DataFrame:
        """Convert raw Binance kline rows into the shared provider schema."""
        rows = []
        for k in raw:
            rows.append({
                "timestamp": pd.to_datetime(int(k[_KLINE_OPEN_TIME]), unit="ms", utc=True),
                "open": float(k[_KLINE_OPEN]),
                "high": float(k[_KLINE_HIGH]),
                "low": float(k[_KLINE_LOW]),
                "close": float(k[_KLINE_CLOSE]),
                "volume": float(k[_KLINE_VOLUME]),
                "quote_volume": float(k[_KLINE_QUOTE_VOLUME]),
                "number_of_trades": int(k[_KLINE_NUMBER_OF_TRADES]),
                "taker_buy_base": float(k[_KLINE_TAKER_BUY_BASE]),
                "taker_buy_quote": float(k[_KLINE_TAKER_BUY_QUOTE]),
            })

        return pd.DataFrame(rows)
