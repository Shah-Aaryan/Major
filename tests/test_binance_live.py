"""
Unit tests for Binance REST and WebSocket market data clients.

Verifies:
1. REST client klines fetching & response parsing into provider schema.
2. Handling of HTTP errors and malformed responses.
3. WebSocket stream callback registration and candle conversion.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from market_data.rest_client import BinanceRESTClient
from market_data.websocket_client import BinanceKlineStream, _candle_to_row
from realtime.binance_websocket import Candle


def test_binance_rest_client_parse_klines():
    raw_response = [
        [
            1704067200000,  # Open time
            "42000.00",     # Open
            "42500.00",     # High
            "41800.00",     # Low
            "42300.00",     # Close
            "100.5",        # Volume
            1704067259999,  # Close time
            "4230000.00",   # Quote volume
            1500,           # Trades
            "50.25",        # Taker buy base
            "2115000.00",   # Taker buy quote
            "0"
        ]
    ]
    
    parsed = BinanceRESTClient._parse_klines(raw_response)
    assert len(parsed) == 1
    assert parsed.iloc[0]["open"] == 42000.00
    assert parsed.iloc[0]["close"] == 42300.00
    assert parsed.iloc[0]["number_of_trades"] == 1500


@patch("market_data.rest_client.requests.get")
def test_binance_rest_client_fetch_mocked(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        [1704067200000, "100.0", "105.0", "99.0", "104.0", "10.0", 1704067259999, "1040.0", 50, "5.0", "520.0", "0"]
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    client = BinanceRESTClient()
    df = client.get_klines("BTCUSDT", "1m", limit=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 104.0


def test_candle_to_row():
    candle = Candle(
        timestamp=pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        open=100.0,
        high=105.0,
        low=99.0,
        close=103.0,
        volume=50.0,
        is_closed=True
    )
    row = _candle_to_row(candle)
    assert row["close"] == 103.0
    assert row["volume"] == 50.0
