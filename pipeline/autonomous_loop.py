"""
Autonomous ML pipeline loop.

Ties every subsystem together into the loop described in the project
README: for each new closed candle, update the rolling window, update
indicators/features, periodically re-optimize parameters, run the human
and ML strategies in parallel, update portfolio/evaluation metrics,
refresh charts, and log a blockchain-hashable audit event.

Re-optimizing (and the parallel backtest that goes with it) on literally
every single candle is not practical — a full walk-forward backtest costs
orders of magnitude more than one candle's worth of work. So steps 1-4
(receive candle, update window, update indicators, feature engineering)
run every candle, while steps 5-13 (optimize, execute, evaluate, chart,
audit) run every ``optimization_interval`` candles, mirroring the existing
``min_adjustment_interval`` throttle already used elsewhere in this
codebase (see ``config.settings.OptimizationConfig``).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from analysis.comparison_report import ComparisonReport, generate_full_report
from analysis.condition_analyzer import ConditionAnalyzer
from analysis.failure_detector import FailureDetector
from audit.audit_logger import AuditEventType, AuditLogger, OptimizationAudit
from backtesting.backtest_engine import BacktestConfig as BTConfig
from backtesting.backtest_engine import BacktestEngine
from config.settings import MarketDataConfig
from evaluation.research_metrics import calculate_parameter_stability, calculate_rolling_accuracy
from features.feature_engine import FeatureConfig, FeatureEngine
from market_data.factory import create_provider
from market_data.provider import MarketDataProvider
from optimization.ml_parameter_adjuster import MLParameterAdjuster
from strategies.base_strategy import BaseStrategy
from strategies.bollinger_breakout import BollingerBreakoutStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.parallel_executor import ParallelExecutionResult, ParallelStrategyExecutor
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from visualization.chart_generator import ChartGenerator

logger = logging.getLogger(__name__)

# Maps the CLI/config-facing strategy name to its concrete class. Every
# concrete strategy overrides __init__ with no required args (unlike the
# abstract BaseStrategy.__init__, which requires `name`) -- mypy can't see
# that through a `Type[BaseStrategy]` variable, hence the ignores below.
STRATEGY_FACTORY: Dict[str, Type[BaseStrategy]] = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "ema_crossover": EMACrossoverStrategy,
    "bollinger_breakout": BollingerBreakoutStrategy,
}


@dataclass
class CycleResult:
    """Outcome of one full optimize -> execute -> evaluate -> audit cycle."""

    cycle_index: int
    timestamp: Any
    regime: str
    human_params: Dict[str, Any]
    ml_params: Dict[str, Any]
    ml_helped: bool
    execution: ParallelExecutionResult
    parameter_stability: Dict[str, float] = field(default_factory=dict)
    rolling_accuracy: Optional[float] = None
    chart_paths: Dict[str, str] = field(default_factory=dict)
    report_path: Optional[str] = None
    report: Optional[ComparisonReport] = None


class AutonomousPipeline:
    """Runs the end-to-end autonomous research loop for one strategy/symbol.

    Data-source agnostic: whether candles come from ``DatasetProvider`` or
    ``BinanceLiveProvider`` is entirely determined by ``market_config`` — this
    class never branches on it (see ``market_data.factory.create_provider``).
    """

    def __init__(
        self,
        market_config: MarketDataConfig,
        strategy_name: str,
        human_params: Optional[Dict[str, Any]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        feature_config: Optional[FeatureConfig] = None,
        optimization_method: str = "random_search",
        n_optimization_iterations: int = 30,
        optimization_interval: int = 50,
        min_optimization_rows: int = 100,
        output_dir: str = "./research_output",
        audit_logger: Optional[AuditLogger] = None,
        audit_backend: str = "local",
    ):
        """
        Args:
            market_config: Unified market data configuration (dataset or live).
            strategy_name: One of ``STRATEGY_FACTORY``'s keys.
            human_params: Baseline human parameters; defaults to the
                strategy's built-in defaults if omitted.
            parameter_bounds: Optimization search bounds; defaults to the
                strategy's ``get_parameter_bounds()`` if omitted.
            feature_config: Optional indicator engine configuration.
            optimization_method: One of ``"bayesian"``, ``"random_search"``,
                ``"evolutionary"``, ``"grid_search"``.
            n_optimization_iterations: Trials per optimization cycle.
            optimization_interval: Re-optimize every N new closed candles.
            min_optimization_rows: Minimum rolling-window rows required
                before the first optimization cycle can run.
            output_dir: Root directory for charts and audit logs.
            audit_logger: Optional pre-built AuditLogger (mainly for tests);
                a new one is created under ``output_dir/audit`` otherwise.
            audit_backend: Storage backend for audit logs (e.g., "local", "db").
        """
        if strategy_name not in STRATEGY_FACTORY:
            raise ValueError(
                f"Unknown strategy_name {strategy_name!r}; expected one of "
                f"{sorted(STRATEGY_FACTORY)}"
            )

        self.market_config = market_config
        self.strategy_cls = STRATEGY_FACTORY[strategy_name]
        self.optimization_method = optimization_method
        self.n_optimization_iterations = n_optimization_iterations
        self.optimization_interval = optimization_interval
        self.min_optimization_rows = min_optimization_rows
        self.output_dir = Path(output_dir)
        self.audit_backend = audit_backend

        seed_strategy = self.strategy_cls()  # type: ignore[call-arg]
        self.human_params: Dict[str, Any] = human_params or seed_strategy.parameters.to_dict()
        self.parameter_bounds: Dict[str, Tuple[float, float]] = (
            parameter_bounds or seed_strategy.get_parameter_bounds()
        )
        self.ml_params: Dict[str, Any] = dict(self.human_params)

        self.audit_logger = audit_logger or AuditLogger(
            output_dir=f"{output_dir}/audit", log_to_file=True, log_to_console=False, audit_backend=self.audit_backend
        )
        self.provider: MarketDataProvider = create_provider(market_config, audit_logger=self.audit_logger)
        self.feature_engine = FeatureEngine(feature_config)
        self.backtest_engine = BacktestEngine(config=BTConfig())
        self.parallel_executor = ParallelStrategyExecutor(verbose=False)
        self.chart_generator = ChartGenerator(output_dir=f"{output_dir}/charts")
        self.adjuster = MLParameterAdjuster(
            objective_function=self._objective_function,
            strategy_bounds={self.strategy_cls.__name__: self.parameter_bounds},
            verbose=False,
        )

        self._cycle_index = 0
        self._candles_since_optimization = 0
        self.parameter_history: List[Dict[str, Any]] = []
        self.prediction_log: List[str] = []
        self.actual_log: List[str] = []
        self.results: List[CycleResult] = []
        self.latest_report: Optional[ComparisonReport] = None

    def _objective_function(self, _strategy_name: str, params: Dict[str, Any], data: pd.DataFrame) -> float:
        """Backtest ``params`` on ``data`` and score by Sharpe ratio."""
        strategy = self.strategy_cls()  # type: ignore[call-arg]
        result = self.backtest_engine.run(strategy, data, parameters=params)
        return result.metrics.sharpe_ratio

    def initialize(self) -> None:
        """Prepare the market data provider for streaming."""
        self.provider.initialize()

    def run_cycle(self) -> Optional[CycleResult]:
        """Process exactly one new closed candle through the autonomous loop.

        Returns:
            A ``CycleResult`` if this candle triggered a full
            optimize/execute/evaluate/audit cycle, ``None`` if it only
            updated the rolling window/features (still-warming-up or not
            yet due for re-optimization), or ``None`` if no candle was
            available.
        """
        candle = self.provider.get_next_candle()
        if candle is None:
            return None

        self._cycle_index += 1
        self._candles_since_optimization += 1

        window = self.provider.get_latest_window()
        if len(window) < self.min_optimization_rows:
            logger.debug("Rolling window warming up (%d/%d rows)", len(window), self.min_optimization_rows)
            return None

        features = self.feature_engine.generate_features(window, drop_na=False)

        if self._candles_since_optimization < self.optimization_interval:
            return None

        self._candles_since_optimization = 0
        return self._run_optimization_cycle(features)

    def _run_optimization_cycle(self, features: pd.DataFrame) -> CycleResult:
        """Run steps 5-13 of the autonomous loop: optimize through audit."""
        regime = self._current_regime(features)

        adjustment = self.adjuster.optimize_strategy(
            strategy_name=self.strategy_cls.__name__,
            train_data=features,
            method=self.optimization_method,
            human_params=self.human_params,
            n_iterations=self.n_optimization_iterations,
            market_condition=regime,
        )
        self.ml_params = adjustment.ml_params
        self.parameter_history.append(dict(self.ml_params))

        strategy = self.strategy_cls()  # type: ignore[call-arg]
        execution = self.parallel_executor.run(
            data=features,
            strategy=strategy,
            human_params=self.human_params,
            ml_params=self.ml_params,
        )

        self._log_audit_event(regime, adjustment, execution)

        stability = calculate_parameter_stability(self.parameter_history)
        self.prediction_log.append("ml" if adjustment.ml_helped else "human")
        self.actual_log.append("ml" if execution.ml_outperformed else "human")
        rolling_acc = calculate_rolling_accuracy(self.prediction_log, self.actual_log, window=10)

        chart_paths = self.chart_generator.generate_session_charts({
            "equity_curve": execution.ml_equity_curve,
            "human_equity_curve": execution.human_equity_curve,
            "rolling_accuracy": rolling_acc,
            "parameter_history": self.parameter_history,
        })

        # Generate comprehensive comparison report
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_stem = f"cycle_{self._cycle_index}_{int(datetime.now().timestamp())}"
        report_md_path = str(reports_dir / f"{report_stem}.md")
        report_json_path = str(reports_dir / f"{report_stem}.json")

        report = generate_full_report(
            strategy_name=self.strategy_cls.__name__,
            human_results=execution.human_metrics.to_dict(),
            ml_results=execution.ml_metrics.to_dict(),
            human_params=self.human_params,
            ml_params=self.ml_params,
            data_period=f"Cycle {self._cycle_index} (Regime: {regime})",
        )
        report.save(report_md_path, format="markdown")
        report.save(report_json_path, format="json")
        self.latest_report = report

        result = CycleResult(
            cycle_index=self._cycle_index,
            timestamp=datetime.now(),
            regime=regime,
            human_params=self.human_params,
            ml_params=self.ml_params,
            ml_helped=adjustment.ml_helped,
            execution=execution,
            parameter_stability=stability,
            rolling_accuracy=float(rolling_acc.iloc[-1]) if len(rolling_acc) else None,
            chart_paths=chart_paths,
            report_path=report_md_path,
            report=report,
        )
        self.results.append(result)
        return result

    def _current_regime(self, features: pd.DataFrame) -> str:
        """Read the most recently computed combined market regime label."""
        if "regime_combined" in features.columns and len(features) > 0:
            value = features["regime_combined"].iloc[-1]
            if pd.notna(value):
                return str(value)
        return "unknown"

    def _log_audit_event(
        self, regime: str, adjustment: Any, execution: ParallelExecutionResult
    ) -> None:
        """Record this cycle's optimization as a blockchain-hashable audit event."""
        self.audit_logger.log_optimization(OptimizationAudit(
            timestamp=datetime.now(),
            strategy_name=self.strategy_cls.__name__,
            optimizer_type=str(self.optimization_method),
            n_trials=adjustment.n_iterations,
            best_objective=adjustment.ml_objective,
            elapsed_time=adjustment.optimization_time_seconds,
            human_params=self.human_params,
            ml_params=self.ml_params,
            parameter_changes=adjustment.parameter_changes,
            human_performance=execution.human_metrics.to_dict(),
            ml_performance=execution.ml_metrics.to_dict(),
            improvement=execution.improvement_sharpe,
            converged=True,
        ))
        self.audit_logger.log_event(
            AuditEventType.REGIME_CHANGE,
            {"regime": regime, "cycle": self._cycle_index},
            explanation=f"Cycle {self._cycle_index} evaluated under regime={regime}",
            strategy_name=self.strategy_cls.__name__,
        )

    def run(self, max_cycles: Optional[int] = None) -> List[CycleResult]:
        """Run the loop until the provider is exhausted (or ``max_cycles`` reached).

        Args:
            max_cycles: Optional cap on the number of full optimization
                cycles to run (mainly for tests and dataset-mode demos).
                Live mode has no natural end, so this is the only way to
                bound a live run.

        Returns:
            All ``CycleResult``s produced during the run.
        """
        self.initialize()
        try:
            while not self.provider.is_finished():
                if max_cycles is not None and len(self.results) >= max_cycles:
                    break
                self.run_cycle()
        finally:
            self.shutdown()

        return self.results

    def generate_final_report(self) -> Optional[ComparisonReport]:
        """Generate and save an aggregate final research comparison report across all completed cycles."""
        if not self.results:
            logger.warning("No cycle results available to generate final report.")
            return None

        latest_cycle = self.results[-1]
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        final_md_path = str(reports_dir / "session_final_report.md")
        final_json_path = str(reports_dir / "session_final_report.json")

        final_report = generate_full_report(
            strategy_name=self.strategy_cls.__name__,
            human_results=latest_cycle.execution.human_metrics.to_dict(),
            ml_results=latest_cycle.execution.ml_metrics.to_dict(),
            human_params=self.human_params,
            ml_params=self.ml_params,
            data_period=f"Full Session ({len(self.results)} Cycles Completed)",
        )
        final_report.save(final_md_path, format="markdown")
        final_report.save(final_json_path, format="json")
        logger.info(f"Generated final session report at {final_md_path}")
        return final_report

    def shutdown(self) -> None:
        """Release the market data provider and finalize the audit log."""
        self.generate_final_report()
        self.provider.shutdown()
        self.audit_logger.close()
