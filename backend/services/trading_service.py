"""Service layer for paper trading operations."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.models.request_models import PaperTradingStartRequest
from backend.models.response_models import (
    PaperTradingStartResponse,
    PaperTradingStatsResponse,
    PaperTradingStopResponse,
)
from backend.utils.session_files import get_research_output_dir
from config.settings import FeatureConfig
from realtime.binance_websocket import SimulatedWebSocket
from realtime.live_feature_updater import LiveFeatureUpdater
from realtime.paper_trader import PaperTrader, PaperTradingSession
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

logger = logging.getLogger(__name__)


class TradingService:
    """Manages a single in-process paper trading session."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_research_output_dir()
        self._lock = threading.Lock()
        self._session: PaperTradingSession | None = None
        self._paper_trader: PaperTrader | None = None
        self._metadata: dict[str, Any] | None = None

    def start(self, payload: PaperTradingStartRequest) -> PaperTradingStartResponse:
        """Start a new paper trading replay session."""
        with self._lock:
            if self._session and self._session._running:
                raise RuntimeError("A paper trading session is already running.")

            data = self._load_market_data(payload.data_path)
            strategy = self._build_strategy(payload.strategy)
            ml_params = self._load_latest_ml_params(payload.strategy)
            if ml_params:
                strategy.set_strategy_specific_params(ml_params)

            stop_loss = float(ml_params.get("stop_loss_pct", 2.0) / 100) if ml_params else 0.02
            take_profit = float(ml_params.get("take_profit_pct", 4.0) / 100) if ml_params else 0.04
            position_size = float(ml_params.get("position_size_pct", 10.0) / 100) if ml_params else 0.10

            websocket = SimulatedWebSocket(
                historical_data=data,
                symbol=payload.symbol,
                replay_speed=payload.replay_speed,
            )
            feature_updater = LiveFeatureUpdater(
                symbols=[payload.symbol],
                feature_config=FeatureConfig(),
                min_warmup_candles=50,
            )
            paper_trader = PaperTrader(
                initial_capital=payload.initial_capital,
                position_size_pct=position_size,
                default_stop_loss_pct=stop_loss,
                default_take_profit_pct=take_profit,
            )
            session = PaperTradingSession(
                websocket=websocket,
                feature_updater=feature_updater,
                strategy=strategy,
                paper_trader=paper_trader,
            )
            session.start()

            self._session = session
            self._paper_trader = paper_trader
            self._metadata = {
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "symbol": payload.symbol,
                "strategy": payload.strategy,
                "data_path": str(Path(payload.data_path).resolve()),
                "replay_speed": payload.replay_speed,
                "initial_capital": payload.initial_capital,
                "ml_parameters_loaded": bool(ml_params),
            }

        logger.info("Started paper trading session for %s using %s", payload.symbol, payload.strategy)
        return PaperTradingStartResponse(
            status="running",
            message="Paper trading session started.",
            trading_stats=self.get_stats(),
        )

    def stop(self) -> PaperTradingStopResponse:
        """Stop the active paper trading session."""
        with self._lock:
            if not self._session:
                raise RuntimeError("No paper trading session is active.")

            self._session.stop()
            self._metadata = {
                **(self._metadata or {}),
                "status": "stopped",
                "stopped_at": datetime.utcnow().isoformat(),
            }

        logger.info("Stopped paper trading session")
        return PaperTradingStopResponse(
            status="stopped",
            message="Paper trading session stopped.",
        )

    def get_stats(self) -> PaperTradingStatsResponse:
        """Return current paper trading stats."""
        with self._lock:
            if not self._session or not self._paper_trader:
                raise RuntimeError("No paper trading session is active.")

            session_running = self._session._running
            session_stats = self._session.get_session_stats()
            replay_progress = self._session.websocket.get_progress()
            stats = {
                "status": "running" if session_running else "stopped",
                "session": session_stats,
                "replay_progress": replay_progress,
                "metadata": self._metadata or {},
            }

        return PaperTradingStatsResponse(**stats)

    def _load_market_data(self, data_path: str) -> pd.DataFrame:
        """Load historical OHLCV CSV for simulated replay."""
        target = Path(data_path)
        if not target.is_absolute():
            target = Path.cwd() / target
        target = target.resolve()

        if not target.exists():
            raise FileNotFoundError(f"Data path '{target}' does not exist.")
        if not target.is_file():
            raise ValueError("Paper trading data_path must point to a CSV file.")

        data = pd.read_csv(target)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data.set_index("timestamp", inplace=True)
        return data

    def _build_strategy(self, strategy_name: str):
        """Instantiate a strategy by key."""
        strategy_map = {
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "ema_crossover": EMACrossoverStrategy,
            "bollinger_breakout": BollingerBreakoutStrategy,
        }
        strategy_cls = strategy_map.get(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unsupported strategy '{strategy_name}'.")
        return strategy_cls()

    def _load_latest_ml_params(self, strategy_name: str) -> dict[str, Any]:
        """Load the latest ML parameters for a strategy from audit output."""
        audit_dir = self.output_dir / "audit"
        if not audit_dir.exists():
            return {}

        strategy_class_map = {
            "rsi_mean_reversion": "RSIMeanReversionStrategy",
            "ema_crossover": "EMACrossoverStrategy",
            "bollinger_breakout": "BollingerBreakoutStrategy",
        }
        strategy_class_name = strategy_class_map[strategy_name]

        for audit_file in sorted(audit_dir.glob("*_full.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            try:
                with audit_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                for event in reversed(payload.get("events", [])):
                    if (
                        event.get("strategy_name") == strategy_class_name
                        and event.get("data", {}).get("ml_params")
                    ):
                        return event["data"]["ml_params"]
            except Exception:
                logger.exception("Failed to parse audit file %s", audit_file)

        return {}


trading_service = TradingService()

