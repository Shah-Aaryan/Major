"""
Performance Metrics for Backtesting.

Comprehensive set of performance metrics for evaluating trading strategies.
These metrics are used to:
1. Compare human params vs ML-adjusted params
2. Evaluate strategy performance under different conditions
3. Provide objective functions for optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive trading performance metrics.
    
    Contains all metrics needed to evaluate and compare
    strategy performance.
    """
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0  # vs benchmark
    
    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # in periods
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Trade analysis
    avg_trade_return: float = 0.0
    avg_trade_duration: float = 0.0  # in periods
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Time-based
    best_month_return: float = 0.0
    worst_month_return: float = 0.0
    pct_positive_months: float = 0.0
    
    # Additional
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_trade_return': self.avg_trade_return,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"=== Performance Summary ===\n"
            f"Total Return: {self.total_return:.2%}\n"
            f"Annualized Return: {self.annualized_return:.2%}\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Sortino Ratio: {self.sortino_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Win Rate: {self.win_rate:.2%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Total Trades: {self.total_trades}"
        )


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate percentage returns from equity curve."""
    return equity_curve.pct_change().dropna()


def calculate_total_return(equity_curve: pd.Series) -> float:
    """Calculate total return."""
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


def calculate_annualized_return(
    total_return: float,
    periods: int,
    periods_per_year: int = 252 * 24 * 60  # Assuming minute data
) -> float:
    """Calculate annualized return."""
    if periods <= 0:
        return 0.0
    years = periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252 * 24 * 60
) -> Tuple[float, float]:
    """
    Calculate volatility and annualized volatility.
    
    Returns:
        Tuple of (volatility, annualized_volatility)
    """
    if len(returns) < 2:
        return 0.0, 0.0
    
    vol = returns.std()
    ann_vol = vol * np.sqrt(periods_per_year)
    
    return vol, ann_vol


def calculate_downside_volatility(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """Calculate downside (semi) volatility."""
    downside_returns = returns[returns < target]
    
    if len(downside_returns) < 2:
        return 0.0
    
    downside_vol = downside_returns.std()
    return downside_vol * np.sqrt(periods_per_year)


def calculate_drawdowns(equity_curve: pd.Series) -> Tuple[pd.Series, float, float, int]:
    """
    Calculate drawdown series and statistics.
    
    Returns:
        Tuple of (drawdown_series, max_drawdown, avg_drawdown, max_duration)
    """
    running_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - running_max) / running_max
    
    max_dd = drawdowns.min()
    avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0.0
    
    # Calculate max drawdown duration
    in_drawdown = drawdowns < 0
    if not in_drawdown.any():
        max_duration = 0
    else:
        # Find consecutive drawdown periods
        dd_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_duration = int(dd_lengths.max())
    
    return drawdowns, max_dd, avg_dd, max_duration


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Sharpe = (Return - Risk Free Rate) / Volatility
    """
    if len(returns) < 2:
        return 0.0
    
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    volatility = returns.std()
    
    if volatility == 0:
        return 0.0
    
    sharpe = excess_return / volatility
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 60
) -> float:
    """
    Calculate Sortino Ratio.
    
    Sortino = (Return - Risk Free Rate) / Downside Volatility
    """
    if len(returns) < 2:
        return 0.0
    
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) < 2:
        return float('inf') if excess_return > 0 else 0.0
    
    downside_vol = downside_returns.std()
    
    if downside_vol == 0:
        return float('inf') if excess_return > 0 else 0.0
    
    sortino = excess_return / downside_vol
    return sortino * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = Annualized Return / |Max Drawdown|
    """
    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    return annualized_return / abs(max_drawdown)


def calculate_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega Ratio.
    
    Omega = Sum of gains above threshold / |Sum of losses below threshold|
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    if losses.sum() == 0:
        return float('inf') if gains.sum() > 0 else 1.0
    
    return gains.sum() / losses.sum()


def calculate_trade_statistics(
    trade_returns: List[float]
) -> Dict[str, float]:
    """
    Calculate trade-level statistics.
    
    Args:
        trade_returns: List of return percentages for each trade
        
    Returns:
        Dictionary of trade statistics
    """
    if not trade_returns:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_trade_return': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
    
    returns_arr = np.array(trade_returns)
    
    wins = returns_arr[returns_arr > 0]
    losses = returns_arr[returns_arr < 0]
    
    total_trades = len(trade_returns)
    winning_trades = len(wins)
    losing_trades = len(losses)
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    
    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Consecutive wins/losses
    max_consecutive_wins = _max_consecutive(returns_arr > 0)
    max_consecutive_losses = _max_consecutive(returns_arr < 0)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_trade_return': returns_arr.mean(),
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }


def _max_consecutive(bool_array: np.ndarray) -> int:
    """Find maximum consecutive True values."""
    if len(bool_array) == 0:
        return 0
    
    max_count = 0
    current_count = 0
    
    for val in bool_array:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    
    return max_count


def calculate_var_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Conditional VaR.
    
    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    if len(returns) < 10:
        return 0.0, 0.0
    
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


def calculate_monthly_statistics(
    equity_curve: pd.Series
) -> Dict[str, float]:
    """Calculate monthly return statistics."""
    if len(equity_curve) < 30:  # Assume daily data, need at least a month
        return {
            'best_month_return': 0.0,
            'worst_month_return': 0.0,
            'pct_positive_months': 0.0
        }
    
    # Resample to monthly. Pandas >=3.0 removed 'M' in favor of 'ME'.
    try:
        monthly = equity_curve.resample('ME').last()
    except ValueError:
        monthly = equity_curve.resample('M').last()
    monthly_returns = monthly.pct_change().dropna()
    
    if len(monthly_returns) == 0:
        return {
            'best_month_return': 0.0,
            'worst_month_return': 0.0,
            'pct_positive_months': 0.0
        }
    
    return {
        'best_month_return': monthly_returns.max(),
        'worst_month_return': monthly_returns.min(),
        'pct_positive_months': (monthly_returns > 0).mean()
    }


def calculate_all_metrics(
    equity_curve: pd.Series,
    trade_returns: Optional[List[float]] = None,
    trade_durations: Optional[List[int]] = None,
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = 252 * 24 * 60  # Minute data
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.
    
    Args:
        equity_curve: Series of portfolio values over time
        trade_returns: List of individual trade returns
        trade_durations: List of trade durations in periods
        risk_free_rate: Annual risk-free rate
        benchmark_returns: Optional benchmark returns for comparison
        periods_per_year: Number of trading periods per year
        
    Returns:
        PerformanceMetrics object with all calculations
    """
    metrics = PerformanceMetrics()
    
    if len(equity_curve) < 2:
        logger.warning("Insufficient data for metrics calculation")
        return metrics
    
    # Calculate returns
    returns = calculate_returns(equity_curve)
    
    # Return metrics
    metrics.total_return = calculate_total_return(equity_curve)
    metrics.annualized_return = calculate_annualized_return(
        metrics.total_return, len(equity_curve), periods_per_year
    )
    
    # Volatility
    metrics.volatility, metrics.annualized_volatility = calculate_volatility(
        returns, periods_per_year
    )
    metrics.downside_volatility = calculate_downside_volatility(
        returns, 0.0, periods_per_year
    )
    
    # Drawdowns
    _, metrics.max_drawdown, metrics.avg_drawdown, metrics.max_drawdown_duration = \
        calculate_drawdowns(equity_curve)
    
    # Risk-adjusted metrics
    metrics.sharpe_ratio = calculate_sharpe_ratio(
        returns, risk_free_rate, periods_per_year
    )
    metrics.sortino_ratio = calculate_sortino_ratio(
        returns, risk_free_rate, periods_per_year
    )
    metrics.calmar_ratio = calculate_calmar_ratio(
        metrics.annualized_return, metrics.max_drawdown
    )
    metrics.omega_ratio = calculate_omega_ratio(returns)
    
    # VaR and CVaR
    metrics.var_95, metrics.cvar_95 = calculate_var_cvar(returns, 0.95)
    
    # Distribution metrics
    if len(returns) > 2:
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)
    
    # Trade statistics
    if trade_returns:
        trade_stats = calculate_trade_statistics(trade_returns)
        metrics.total_trades = trade_stats['total_trades']
        metrics.winning_trades = trade_stats['winning_trades']
        metrics.losing_trades = trade_stats['losing_trades']
        metrics.win_rate = trade_stats['win_rate']
        metrics.avg_win = trade_stats['avg_win']
        metrics.avg_loss = trade_stats['avg_loss']
        metrics.profit_factor = trade_stats['profit_factor']
        metrics.expectancy = trade_stats['expectancy']
        metrics.avg_trade_return = trade_stats['avg_trade_return']
        metrics.max_consecutive_wins = trade_stats['max_consecutive_wins']
        metrics.max_consecutive_losses = trade_stats['max_consecutive_losses']
    
    # Trade duration
    if trade_durations:
        metrics.avg_trade_duration = np.mean(trade_durations)
    
    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(
            benchmark_returns, join='inner'
        )
        
        if len(aligned_returns) > 0:
            excess = aligned_returns - aligned_benchmark
            metrics.excess_return = excess.mean() * periods_per_year
            
            tracking_error = excess.std() * np.sqrt(periods_per_year)
            if tracking_error > 0:
                metrics.information_ratio = metrics.excess_return / tracking_error
    
    # Monthly statistics (if we have datetime index)
    if hasattr(equity_curve.index, 'to_period'):
        try:
            monthly_stats = calculate_monthly_statistics(equity_curve)
            metrics.best_month_return = monthly_stats['best_month_return']
            metrics.worst_month_return = monthly_stats['worst_month_return']
            metrics.pct_positive_months = monthly_stats['pct_positive_months']
        except Exception as e:
            logger.debug(f"Could not calculate monthly statistics: {e}")
    
    return metrics


def create_objective_function(
    metric: str = 'sharpe_ratio',
    constraints: Optional[Dict[str, float]] = None
) -> callable:
    """
    Create an objective function for optimization.
    
    Args:
        metric: Metric to optimize (e.g., 'sharpe_ratio', 'sortino_ratio')
        constraints: Dict of constraint metrics and their minimum values
        
    Returns:
        Callable that takes PerformanceMetrics and returns objective value
    """
    constraints = constraints or {}
    
    def objective(metrics: PerformanceMetrics) -> float:
        # Check constraints
        for constraint_metric, min_value in constraints.items():
            if hasattr(metrics, constraint_metric):
                if getattr(metrics, constraint_metric) < min_value:
                    return float('-inf')  # Constraint violated
        
        # Return target metric
        if hasattr(metrics, metric):
            value = getattr(metrics, metric)
            # Handle inf values
            if np.isinf(value):
                return 10.0 if value > 0 else -10.0
            return value
        
        raise ValueError(f"Unknown metric: {metric}")
    
    return objective
