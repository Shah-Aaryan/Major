"""Logging helpers for the FastAPI backend."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    """Configure application-wide logging once."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
