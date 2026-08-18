"""
Provider factory — the single place that reads ``DATA_SOURCE``.

This is what makes mode-switching config-only: everything else in the
pipeline calls ``create_provider(config)`` and depends only on the
``MarketDataProvider`` interface, never on ``DATA_SOURCE`` itself.
"""

import logging
from typing import Any, Optional

from config.settings import MarketDataConfig
from market_data.binance_provider import BinanceLiveProvider
from market_data.dataset_provider import DatasetProvider
from market_data.provider import MarketDataProvider

logger = logging.getLogger(__name__)


def create_provider(
    config: MarketDataConfig, audit_logger: Optional[Any] = None
) -> MarketDataProvider:
    """Build the correct provider for ``config.DATA_SOURCE``.

    Args:
        config: Unified market data configuration (see ``config/beyondalgo.json``).
        audit_logger: Optional ``audit.audit_logger.AuditLogger`` passed through
            to ``BinanceLiveProvider`` for fallback-transition logging.

    Returns:
        A ``DatasetProvider`` when ``DATA_SOURCE == "dataset"``, or a
        ``BinanceLiveProvider`` when ``DATA_SOURCE == "live"``.

    Raises:
        ValueError: If ``config.DATA_SOURCE`` is neither ``"dataset"`` nor ``"live"``.
    """
    if config.DATA_SOURCE == "dataset":
        logger.info("Creating DatasetProvider (CSV_PATH=%s)", config.CSV_PATH)
        return DatasetProvider(
            csv_path=config.CSV_PATH,
            window_size=config.WINDOW_SIZE,
            symbol=config.SYMBOL,
        )

    if config.DATA_SOURCE == "live":
        logger.info("Creating BinanceLiveProvider (symbol=%s)", config.SYMBOL)
        return BinanceLiveProvider(
            symbol=config.SYMBOL,
            interval=config.TIMEFRAME,
            window_size=config.WINDOW_SIZE,
            rest_endpoint=config.REST_ENDPOINT,
            ws_endpoint=config.WEBSOCKET_ENDPOINT,
            fallback_csv_path=config.CSV_PATH,
            use_testnet=config.USE_TESTNET,
            rest_timeout=config.REST_TIMEOUT_SECONDS,
            ws_reconnect_attempts=config.WS_RECONNECT_ATTEMPTS,
            ws_reconnect_delay=config.WS_RECONNECT_DELAY_SECONDS,
            audit_logger=audit_logger,
        )

    raise ValueError(f"Unknown DATA_SOURCE: {config.DATA_SOURCE!r}")
