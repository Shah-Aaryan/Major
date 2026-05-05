"""
Walk-Forward Validation for robust strategy evaluation.

Walk-forward analysis is critical for:
1. Avoiding overfitting to historical data
2. Testing ML optimization in realistic conditions
3. Understanding how parameters need to adapt over time

The key insight: Parameters optimized on past data may not work
in the future. Walk-forward tests this systematically.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from strategies.base_strategy import BaseStrategy
from backtesting.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult
)
from backtesting.metrics import PerformanceMetrics, calculate_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Results
    optimized_params: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Optional[PerformanceMetrics] = None
    test_metrics: Optional[PerformanceMetrics] = None
    
    # Comparison
    baseline_test_metrics: Optional[PerformanceMetrics] = None
    ml_improvement: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'window_id': self.window_id,
            'train_start': str(self.train_start),
            'train_end': str(self.train_end),
            'test_start': str(self.test_start),
            'test_end': str(self.test_end),
            'optimized_params': self.optimized_params,
            'train_sharpe': self.train_metrics.sharpe_ratio if self.train_metrics else None,
            'test_sharpe': self.test_metrics.sharpe_ratio if self.test_metrics else None,
            'baseline_test_sharpe': self.baseline_test_metrics.sharpe_ratio if self.baseline_test_metrics else None,
            'ml_improvement': self.ml_improvement
        }


@dataclass
class WalkForwardResult:
    """
    Complete result of walk-forward analysis.
    
    Contains results from all windows plus aggregate metrics
    for understanding ML optimization effectiveness.
    """
    # Window results
    windows: List[WalkForwardWindow]
    
    # Aggregate metrics
    aggregate_train_metrics: PerformanceMetrics
    aggregate_test_metrics: PerformanceMetrics
    aggregate_baseline_metrics: PerformanceMetrics
    
    # Combined equity curves
    combined_equity: pd.Series
    combined_baseline_equity: pd.Series
    
    # Strategy info
    strategy_name: str
    n_windows: int
    train_ratio: float
    
    # ML effectiveness analysis
    ml_helped_windows: int = 0
    avg_ml_improvement: float = 0.0
    ml_consistency: float = 0.0  # How often ML helps
    overfitting_ratio: float = 0.0  # train vs test performance gap
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.windows:
            improvements = [w.ml_improvement for w in self.windows]
            self.ml_helped_windows = sum(1 for i in improvements if i > 0)
            self.avg_ml_improvement = np.mean(improvements)
            self.ml_consistency = self.ml_helped_windows / len(self.windows)
            
            # Calculate overfitting ratio
            train_sharpes = [
                w.train_metrics.sharpe_ratio 
                for w in self.windows if w.train_metrics
            ]
            test_sharpes = [
                w.test_metrics.sharpe_ratio 
                for w in self.windows if w.test_metrics
            ]
            
            if train_sharpes and test_sharpes:
                avg_train = np.mean(train_sharpes)
                avg_test = np.mean(test_sharpes)
                # Bounded gap metric in [-1, 1] for stability/readability.
                denom = abs(avg_train) + abs(avg_test) + 1e-9
                self.overfitting_ratio = (avg_train - avg_test) / denom
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"=== Walk-Forward Analysis: {self.strategy_name} ===\n"
            f"Windows: {self.n_windows}, Train Ratio: {self.train_ratio:.0%}\n"
            f"\n"
            f"ML EFFECTIVENESS:\n"
            f"  ML helped in: {self.ml_helped_windows}/{self.n_windows} windows "
            f"({self.ml_consistency:.1%})\n"
            f"  Avg improvement: {self.avg_ml_improvement:.2%}\n"
            f"  Overfitting ratio: {self.overfitting_ratio:.2%}\n"
            f"\n"
            f"AGGREGATE PERFORMANCE (ML-Optimized):\n"
            f"  Total Return: {self.aggregate_test_metrics.total_return:.2%}\n"
            f"  Sharpe Ratio: {self.aggregate_test_metrics.sharpe_ratio:.2f}\n"
            f"  Max Drawdown: {self.aggregate_test_metrics.max_drawdown:.2%}\n"
            f"\n"
            f"BASELINE (Human Params):\n"
            f"  Total Return: {self.aggregate_baseline_metrics.total_return:.2%}\n"
            f"  Sharpe Ratio: {self.aggregate_baseline_metrics.sharpe_ratio:.2f}\n"
        )
    
    def get_window_dataframe(self) -> pd.DataFrame:
        """Convert window results to DataFrame."""
        return pd.DataFrame([w.to_dict() for w in self.windows])


class WalkForwardValidator:
    """
    Walk-Forward Validation for strategy optimization.
    
    This class implements anchored and rolling walk-forward analysis
    to test ML parameter optimization in a realistic setting.
    
    Key concepts:
    - Train window: Period used to optimize parameters
    - Test window: Out-of-sample period to validate optimization
    - Anchored: Train always starts from beginning
    - Rolling: Train window moves forward
    
    RESEARCH PURPOSE: This answers the critical question of whether
    ML optimization actually helps in realistic forward-testing scenarios.
    
    Usage:
        validator = WalkForwardValidator(
            strategy=my_strategy,
            optimize_function=my_optimizer,
            backtest_engine=BacktestEngine()
        )
        
        result = validator.run(
            data=full_data,
            n_windows=5,
            train_ratio=0.8
        )
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        optimize_function: Callable[[pd.DataFrame], Dict[str, Any]],
        backtest_config: Optional[BacktestConfig] = None,
        baseline_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Walk-Forward Validator.
        
        Args:
            strategy: Trading strategy to test
            optimize_function: Function(train_data) -> optimized_params
            backtest_config: Configuration for backtesting
            baseline_params: Human baseline parameters for comparison
        """
        self.strategy = strategy
        self.optimize_function = optimize_function
        self.backtest_config = backtest_config or BacktestConfig()
        self.baseline_params = baseline_params or {}
        
        self.backtest_engine = BacktestEngine(self.backtest_config)
    
    def run(
        self,
        data: pd.DataFrame,
        n_windows: int = 5,
        train_ratio: float = 0.8,
        anchored: bool = False,
        min_train_size: int = 1000
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.
        
        Args:
            data: Full dataset with OHLCV and features
            n_windows: Number of walk-forward windows
            train_ratio: Ratio of data to use for training in each window
            anchored: If True, train always starts from beginning
            min_train_size: Minimum training data points
            
        Returns:
            WalkForwardResult with all window results and aggregate metrics
        """
        logger.info(
            f"Starting walk-forward analysis with {n_windows} windows "
            f"(train_ratio={train_ratio}, anchored={anchored})"
        )
        
        # Generate windows
        windows = self._generate_windows(
            data, n_windows, train_ratio, anchored, min_train_size
        )
        
        # Store results
        window_results: List[WalkForwardWindow] = []
        test_equity_curves: List[pd.Series] = []
        baseline_equity_curves: List[pd.Series] = []
        
        for window in windows:
            logger.info(
                f"Processing window {window.window_id + 1}/{n_windows}"
            )
            
            # Get train and test data
            train_data = data.loc[window.train_start:window.train_end]
            test_data = data.loc[window.test_start:window.test_end]
            
            if len(train_data) < min_train_size:
                logger.warning(f"Skipping window {window.window_id}: insufficient training data")
                continue
            
            # === OPTIMIZE ON TRAIN DATA ===
            try:
                optimized_params = self.optimize_function(train_data)
                window.optimized_params = optimized_params
            except Exception as e:
                logger.error(f"Optimization failed for window {window.window_id}: {e}")
                continue
            
            # === BACKTEST ON TRAIN (to measure overfitting) ===
            train_result = self.backtest_engine.run(
                self.strategy, train_data, optimized_params
            )
            window.train_metrics = train_result.metrics
            
            # === BACKTEST ON TEST (out-of-sample) ===
            test_result = self.backtest_engine.run(
                self.strategy, test_data, optimized_params
            )
            window.test_metrics = test_result.metrics
            test_equity_curves.append(test_result.equity_curve)
            
            # === BASELINE BACKTEST (human params) ===
            if self.baseline_params:
                baseline_result = self.backtest_engine.run(
                    self.strategy, test_data, self.baseline_params
                )
                window.baseline_test_metrics = baseline_result.metrics
                baseline_equity_curves.append(baseline_result.equity_curve)
                
                # Calculate improvement
                base_sharpe = float(getattr(baseline_result.metrics, "sharpe_ratio", 0.0) or 0.0)
                test_sharpe = float(getattr(test_result.metrics, "sharpe_ratio", 0.0) or 0.0)

                if abs(base_sharpe) > 1e-9:
                    window.ml_improvement = (test_sharpe - base_sharpe) / abs(base_sharpe)
                else:
                    # When baseline Sharpe is 0 (often means no trades), represent improvement
                    # as a bounded proxy from the ML Sharpe itself so it isn't stuck at 0.
                    window.ml_improvement = float(np.tanh(test_sharpe / 2.0))
            
            window_results.append(window)
            
            logger.info(
                f"Window {window.window_id}: "
                f"train_sharpe={window.train_metrics.sharpe_ratio:.2f}, "
                f"test_sharpe={window.test_metrics.sharpe_ratio:.2f}, "
                f"improvement={window.ml_improvement:.2%}"
            )

        if not window_results or not test_equity_curves:
            raise RuntimeError(
                "Walk-forward produced no successful windows (all windows were skipped or failed during optimization/backtesting)"
            )
        
        # Combine equity curves
        combined_equity = self._combine_equity_curves(test_equity_curves)
        combined_baseline = self._combine_equity_curves(baseline_equity_curves)

        if combined_equity is None:
            raise RuntimeError(
                "Walk-forward produced no combined equity curve (no test equity curves to combine)"
            )
        
        # Calculate aggregate metrics
        aggregate_test = calculate_all_metrics(combined_equity)
        aggregate_baseline = calculate_all_metrics(combined_baseline) if combined_baseline is not None else PerformanceMetrics()
        
        # Aggregate train metrics (average)
        train_sharpes = [w.train_metrics.sharpe_ratio for w in window_results if w.train_metrics]
        aggregate_train = PerformanceMetrics(
            sharpe_ratio=np.mean(train_sharpes) if train_sharpes else 0.0
        )
        
        result = WalkForwardResult(
            windows=window_results,
            aggregate_train_metrics=aggregate_train,
            aggregate_test_metrics=aggregate_test,
            aggregate_baseline_metrics=aggregate_baseline,
            combined_equity=combined_equity,
            combined_baseline_equity=combined_baseline,
            strategy_name=self.strategy.name,
            n_windows=len(window_results),
            train_ratio=train_ratio
        )
        
        logger.info(f"Walk-forward analysis complete")
        logger.info(result.summary())
        
        return result
    
    def _generate_windows(
        self,
        data: pd.DataFrame,
        n_windows: int,
        train_ratio: float,
        anchored: bool,
        min_train_size: int
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows."""
        total_len = len(data)
        windows = []
        
        if anchored:
            # Anchored: train always starts from beginning
            test_size = total_len // n_windows
            
            for i in range(n_windows):
                test_start_idx = total_len - (n_windows - i) * test_size
                test_end_idx = total_len - (n_windows - i - 1) * test_size - 1
                
                # Train from start to just before test
                train_start_idx = 0
                train_end_idx = test_start_idx - 1
                
                if train_end_idx - train_start_idx < min_train_size:
                    continue
                
                windows.append(WalkForwardWindow(
                    window_id=i,
                    train_start=data.index[train_start_idx],
                    train_end=data.index[train_end_idx],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[min(test_end_idx, total_len - 1)]
                ))
        else:
            # Rolling: train window moves forward
            window_size = total_len // n_windows
            train_size = int(window_size * train_ratio)
            test_size = window_size - train_size
            
            for i in range(n_windows):
                start_idx = i * window_size
                train_end_idx = start_idx + train_size - 1
                test_start_idx = train_end_idx + 1
                test_end_idx = min(start_idx + window_size - 1, total_len - 1)
                
                if train_size < min_train_size:
                    continue
                
                windows.append(WalkForwardWindow(
                    window_id=i,
                    train_start=data.index[start_idx],
                    train_end=data.index[train_end_idx],
                    test_start=data.index[test_start_idx],
                    test_end=data.index[test_end_idx]
                ))
        
        return windows
    
    def _combine_equity_curves(
        self,
        curves: List[pd.Series]
    ) -> Optional[pd.Series]:
        """Combine multiple equity curves into one continuous curve."""
        if not curves:
            return None
        
        # Normalize each curve to start at 1, then chain them
        combined_values = [1.0]
        combined_index = [curves[0].index[0]]
        
        for curve in curves:
            if len(curve) == 0:
                continue
            
            # Normalize to starting equity of 1
            normalized = curve / curve.iloc[0]
            
            # Scale by ending value of previous segment
            scaled = normalized * combined_values[-1]
            
            # Append (skip first value to avoid duplicate)
            combined_values.extend(scaled.iloc[1:].values)
            combined_index.extend(curve.index[1:])
        
        return pd.Series(combined_values, index=combined_index)


class CrossValidator:
    """
    K-Fold Cross-Validation for strategy parameters.
    
    Unlike walk-forward which respects time ordering,
    cross-validation can help understand parameter stability
    (though it has look-ahead bias issues for time series).
    
    Use with caution - mainly for understanding parameter robustness,
    not for realistic performance estimation.
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        optimize_function: Callable[[pd.DataFrame], Dict[str, Any]],
        backtest_config: Optional[BacktestConfig] = None
    ):
        """Initialize cross-validator."""
        self.strategy = strategy
        self.optimize_function = optimize_function
        self.backtest_config = backtest_config or BacktestConfig()
        self.backtest_engine = BacktestEngine(self.backtest_config)
    
    def run_time_series_cv(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 0
    ) -> Dict[str, Any]:
        """
        Run time-series cross-validation.
        
        This is similar to walk-forward but with sklearn-style splits.
        
        Args:
            data: Full dataset
            n_splits: Number of splits
            gap: Gap between train and test to avoid lookahead
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        results = {
            'train_sharpes': [],
            'test_sharpes': [],
            'params': [],
            'overfit_ratios': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Optimize
            params = self.optimize_function(train_data)
            results['params'].append(params)
            
            # Backtest
            train_result = self.backtest_engine.run(self.strategy, train_data, params)
            test_result = self.backtest_engine.run(self.strategy, test_data, params)
            
            results['train_sharpes'].append(train_result.metrics.sharpe_ratio)
            results['test_sharpes'].append(test_result.metrics.sharpe_ratio)
            
            # Overfitting ratio
            if train_result.metrics.sharpe_ratio != 0:
                overfit = (
                    train_result.metrics.sharpe_ratio - test_result.metrics.sharpe_ratio
                ) / abs(train_result.metrics.sharpe_ratio)
            else:
                overfit = 0.0
            results['overfit_ratios'].append(overfit)
        
        # Summary statistics
        results['summary'] = {
            'mean_train_sharpe': np.mean(results['train_sharpes']),
            'mean_test_sharpe': np.mean(results['test_sharpes']),
            'std_test_sharpe': np.std(results['test_sharpes']),
            'mean_overfit_ratio': np.mean(results['overfit_ratios']),
            'param_stability': self._calculate_param_stability(results['params'])
        }
        
        return results
    
    def _calculate_param_stability(
        self,
        params_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate coefficient of variation for each parameter."""
        if not params_list:
            return {}
        
        stability = {}
        param_names = params_list[0].keys()
        
        for param in param_names:
            values = [p[param] for p in params_list if param in p]
            if values and all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
                stability[param] = {
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv  # Lower CV = more stable
                }
        
        return stability
