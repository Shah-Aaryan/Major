"""
Condition Analyzer - Analyze when ML optimization helps.

This module identifies and categorizes market conditions to understand
when ML parameter optimization provides value over human-tuned parameters.

Key research questions:
- In which market regimes does ML optimization help?
- How do optimal parameters vary across conditions?
- Can we predict when ML will help?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    """Market trend regimes."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    RANGING = "ranging"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
    """Market volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class VolumeRegime(Enum):
    """Market volume regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class MarketCondition:
    """
    Represents a specific market condition.
    
    Used to categorize data periods and analyze ML effectiveness
    under different conditions.
    """
    trend: TrendRegime
    volatility: VolatilityRegime
    volume: VolumeRegime
    
    # Additional context
    avg_return: float = 0.0
    return_std: float = 0.0
    avg_volume: float = 0.0
    adx_value: float = 0.0
    
    # Period info
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    n_bars: int = 0
    
    def __str__(self) -> str:
        return f"{self.trend.value}_{self.volatility.value}_{self.volume.value}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trend': self.trend.value,
            'volatility': self.volatility.value,
            'volume': self.volume.value,
            'avg_return': self.avg_return,
            'return_std': self.return_std,
            'adx_value': self.adx_value,
            'start_date': str(self.start_date) if self.start_date else None,
            'end_date': str(self.end_date) if self.end_date else None,
            'n_bars': self.n_bars
        }


@dataclass
class ConditionPerformance:
    """Performance metrics for a specific market condition."""
    condition: MarketCondition
    
    # Human baseline performance
    human_sharpe: float = 0.0
    human_return: float = 0.0
    human_max_dd: float = 0.0
    
    # ML-optimized performance
    ml_sharpe: float = 0.0
    ml_return: float = 0.0
    ml_max_dd: float = 0.0
    
    # Comparison
    improvement_sharpe: float = 0.0
    improvement_return: float = 0.0
    ml_helped: bool = False
    
    # Optimal parameters for this condition
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    param_changes: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate improvements."""
        if self.human_sharpe != 0:
            self.improvement_sharpe = (
                (self.ml_sharpe - self.human_sharpe) / abs(self.human_sharpe)
            )
        self.improvement_return = self.ml_return - self.human_return
        self.ml_helped = self.improvement_sharpe > 0.05  # 5% improvement threshold


class ConditionAnalyzer:
    """
    Analyzes ML optimization effectiveness across market conditions.
    
    This is the core analytical tool for understanding WHEN ML helps.
    It:
    1. Segments data into market condition periods
    2. Tests optimization under each condition
    3. Identifies patterns in when ML helps vs fails
    4. Generates actionable insights
    
    Usage:
        analyzer = ConditionAnalyzer()
        
        # Analyze conditions in data
        conditions = analyzer.identify_conditions(data)
        
        # After running optimizations, analyze effectiveness
        analysis = analyzer.analyze_ml_effectiveness(
            conditions=conditions,
            human_results=human_backtest_results,
            ml_results=ml_backtest_results
        )
    """
    
    def __init__(
        self,
        trend_thresholds: Optional[Dict[str, float]] = None,
        volatility_thresholds: Optional[Dict[str, float]] = None,
        volume_thresholds: Optional[Dict[str, float]] = None,
        min_period_bars: int = 100
    ):
        """
        Initialize the Condition Analyzer.
        
        Args:
            trend_thresholds: Thresholds for trend classification
            volatility_thresholds: Thresholds for volatility classification
            volume_thresholds: Thresholds for volume classification
            min_period_bars: Minimum bars for a condition period
        """
        self.trend_thresholds = trend_thresholds or {
            'strong_up': 0.15,    # >15% annualized return
            'weak_up': 0.05,      # 5-15%
            'weak_down': -0.05,   # -5% to 5%
            'strong_down': -0.15  # <-15%
        }
        
        self.volatility_thresholds = volatility_thresholds or {
            'low': 0.10,          # <10% annualized vol
            'normal': 0.25,       # 10-25%
            'high': 0.50,         # 25-50%
            # >50% is extreme
        }
        
        self.volume_thresholds = volume_thresholds or {
            'low': 0.5,           # <50% of average
            'high': 1.5           # >150% of average
        }
        
        self.min_period_bars = min_period_bars
        
        # Store analysis results
        self.condition_performances: List[ConditionPerformance] = []
    
    def identify_conditions(
        self,
        data: pd.DataFrame,
        window_size: int = 500
    ) -> List[MarketCondition]:
        """
        Identify market conditions in the data.
        
        Args:
            data: DataFrame with OHLCV and features
            window_size: Rolling window for condition calculation
            
        Returns:
            List of MarketCondition objects for each period
        """
        conditions = []
        
        # Calculate required metrics if not present
        if 'returns' not in data.columns:
            data = data.copy()
            data['returns'] = data['close'].pct_change()
        
        # Process in windows
        n_windows = len(data) // window_size
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(data))
            
            window_data = data.iloc[start_idx:end_idx]
            
            condition = self._classify_period(window_data)
            condition.start_date = window_data.index[0]
            condition.end_date = window_data.index[-1]
            condition.n_bars = len(window_data)
            
            conditions.append(condition)
        
        logger.info(f"Identified {len(conditions)} condition periods")
        
        return conditions
    
    def _classify_period(self, data: pd.DataFrame) -> MarketCondition:
        """Classify a data period into market conditions."""
        returns = data['returns'].dropna()
        
        # Calculate metrics
        avg_return = returns.mean() * 252 * 24 * 60  # Annualized (minute data)
        return_std = returns.std() * np.sqrt(252 * 24 * 60)
        
        # Get volume info
        avg_volume = data['volume'].mean()
        overall_avg_volume = data['volume'].mean()  # Would need full data for true comparison
        
        # Get ADX if available
        adx_value = data['adx'].mean() if 'adx' in data.columns else 25.0
        
        # Classify trend
        trend = self._classify_trend(avg_return, adx_value)
        
        # Classify volatility
        volatility = self._classify_volatility(return_std)
        
        # Classify volume (relative to period average)
        volume = VolumeRegime.NORMAL  # Simplified
        
        return MarketCondition(
            trend=trend,
            volatility=volatility,
            volume=volume,
            avg_return=avg_return,
            return_std=return_std,
            avg_volume=avg_volume,
            adx_value=adx_value
        )
    
    def _classify_trend(self, avg_return: float, adx: float) -> TrendRegime:
        """Classify trend regime."""
        if avg_return > self.trend_thresholds['strong_up']:
            return TrendRegime.STRONG_UPTREND
        elif avg_return > self.trend_thresholds['weak_up']:
            return TrendRegime.WEAK_UPTREND
        elif avg_return > self.trend_thresholds['weak_down']:
            return TrendRegime.RANGING
        elif avg_return > self.trend_thresholds['strong_down']:
            return TrendRegime.WEAK_DOWNTREND
        else:
            return TrendRegime.STRONG_DOWNTREND
    
    def _classify_volatility(self, vol: float) -> VolatilityRegime:
        """Classify volatility regime."""
        if vol < self.volatility_thresholds['low']:
            return VolatilityRegime.LOW
        elif vol < self.volatility_thresholds['normal']:
            return VolatilityRegime.NORMAL
        elif vol < self.volatility_thresholds['high']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def analyze_ml_effectiveness(
        self,
        condition_results: List[Tuple[MarketCondition, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze ML effectiveness across conditions.
        
        Args:
            condition_results: List of (condition, results_dict) tuples
                where results_dict contains 'human_metrics', 'ml_metrics', 'ml_params'
                
        Returns:
            Analysis dictionary with insights
        """
        self.condition_performances = []
        
        for condition, results in condition_results:
            perf = ConditionPerformance(
                condition=condition,
                human_sharpe=results.get('human_metrics', {}).get('sharpe_ratio', 0),
                human_return=results.get('human_metrics', {}).get('total_return', 0),
                human_max_dd=results.get('human_metrics', {}).get('max_drawdown', 0),
                ml_sharpe=results.get('ml_metrics', {}).get('sharpe_ratio', 0),
                ml_return=results.get('ml_metrics', {}).get('total_return', 0),
                ml_max_dd=results.get('ml_metrics', {}).get('max_drawdown', 0),
                optimal_params=results.get('ml_params', {})
            )
            self.condition_performances.append(perf)
        
        return self._generate_analysis()
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis from condition performances."""
        if not self.condition_performances:
            return {'error': 'No condition performances to analyze'}
        
        analysis = {
            'overall': self._overall_analysis(),
            'by_trend': self._analyze_by_trend(),
            'by_volatility': self._analyze_by_volatility(),
            'best_conditions': self._find_best_conditions(),
            'worst_conditions': self._find_worst_conditions(),
            'parameter_patterns': self._analyze_parameter_patterns(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _overall_analysis(self) -> Dict[str, Any]:
        """Overall analysis across all conditions."""
        total = len(self.condition_performances)
        helped = sum(1 for p in self.condition_performances if p.ml_helped)
        
        improvements = [p.improvement_sharpe for p in self.condition_performances]
        
        return {
            'total_conditions': total,
            'ml_helped_count': helped,
            'ml_helped_rate': helped / total if total > 0 else 0,
            'avg_improvement': np.mean(improvements),
            'median_improvement': np.median(improvements),
            'std_improvement': np.std(improvements),
            'max_improvement': np.max(improvements),
            'max_degradation': np.min(improvements)
        }
    
    def _analyze_by_trend(self) -> Dict[str, Dict[str, Any]]:
        """Analyze ML effectiveness by trend regime."""
        by_trend = {}
        
        for trend in TrendRegime:
            perfs = [p for p in self.condition_performances if p.condition.trend == trend]
            if perfs:
                improvements = [p.improvement_sharpe for p in perfs]
                helped = sum(1 for p in perfs if p.ml_helped)
                
                by_trend[trend.value] = {
                    'n_periods': len(perfs),
                    'ml_helped_rate': helped / len(perfs),
                    'avg_improvement': np.mean(improvements),
                    'avg_human_sharpe': np.mean([p.human_sharpe for p in perfs]),
                    'avg_ml_sharpe': np.mean([p.ml_sharpe for p in perfs])
                }
        
        return by_trend
    
    def _analyze_by_volatility(self) -> Dict[str, Dict[str, Any]]:
        """Analyze ML effectiveness by volatility regime."""
        by_vol = {}
        
        for vol in VolatilityRegime:
            perfs = [p for p in self.condition_performances if p.condition.volatility == vol]
            if perfs:
                improvements = [p.improvement_sharpe for p in perfs]
                helped = sum(1 for p in perfs if p.ml_helped)
                
                by_vol[vol.value] = {
                    'n_periods': len(perfs),
                    'ml_helped_rate': helped / len(perfs),
                    'avg_improvement': np.mean(improvements)
                }
        
        return by_vol
    
    def _find_best_conditions(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Find conditions where ML helped most."""
        sorted_perfs = sorted(
            self.condition_performances,
            key=lambda p: p.improvement_sharpe,
            reverse=True
        )
        
        best = []
        for p in sorted_perfs[:top_n]:
            best.append({
                'condition': str(p.condition),
                'improvement': p.improvement_sharpe,
                'human_sharpe': p.human_sharpe,
                'ml_sharpe': p.ml_sharpe,
                'optimal_params': p.optimal_params
            })
        
        return best
    
    def _find_worst_conditions(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """Find conditions where ML hurt most."""
        sorted_perfs = sorted(
            self.condition_performances,
            key=lambda p: p.improvement_sharpe
        )
        
        worst = []
        for p in sorted_perfs[:top_n]:
            worst.append({
                'condition': str(p.condition),
                'degradation': p.improvement_sharpe,
                'human_sharpe': p.human_sharpe,
                'ml_sharpe': p.ml_sharpe
            })
        
        return worst
    
    def _analyze_parameter_patterns(self) -> Dict[str, Any]:
        """Analyze how optimal parameters vary across conditions."""
        if not self.condition_performances:
            return {}
        
        # Collect all parameters
        all_params = {}
        for perf in self.condition_performances:
            for param, value in perf.optimal_params.items():
                if isinstance(value, (int, float)):
                    if param not in all_params:
                        all_params[param] = []
                    all_params[param].append({
                        'value': value,
                        'condition': str(perf.condition),
                        'ml_helped': perf.ml_helped
                    })
        
        patterns = {}
        for param, values in all_params.items():
            param_values = [v['value'] for v in values]
            
            # Parameter variability
            cv = np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else 0
            
            # Correlation with ML helping
            helped_values = [v['value'] for v in values if v['ml_helped']]
            failed_values = [v['value'] for v in values if not v['ml_helped']]
            
            patterns[param] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'cv': cv,  # Coefficient of variation
                'mean_when_helped': np.mean(helped_values) if helped_values else None,
                'mean_when_failed': np.mean(failed_values) if failed_values else None
            }
        
        return patterns
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        overall = self._overall_analysis()
        
        # Overall effectiveness
        if overall['ml_helped_rate'] < 0.3:
            recommendations.append(
                "ML optimization helped in less than 30% of conditions. "
                "Consider: (1) The human parameters may already be well-tuned, "
                "(2) Increase optimization iterations, or "
                "(3) Focus on specific conditions where ML works better."
            )
        elif overall['ml_helped_rate'] > 0.7:
            recommendations.append(
                "ML optimization helped in most conditions (>70%). "
                "Consider using ML-adjusted parameters as the default."
            )
        
        # Trend-specific recommendations
        by_trend = self._analyze_by_trend()
        for trend, stats in by_trend.items():
            if stats['ml_helped_rate'] > 0.6:
                recommendations.append(
                    f"ML optimization works well in {trend} conditions. "
                    f"Consider always using ML params in these markets."
                )
            elif stats['ml_helped_rate'] < 0.3 and stats['n_periods'] >= 2:
                recommendations.append(
                    f"ML optimization rarely helps in {trend} conditions. "
                    f"Use human parameters in these markets."
                )
        
        # Volatility-specific
        by_vol = self._analyze_by_volatility()
        for vol, stats in by_vol.items():
            if stats['ml_helped_rate'] > 0.7:
                recommendations.append(
                    f"ML particularly helpful in {vol} volatility regimes."
                )
        
        # Parameter stability
        patterns = self._analyze_parameter_patterns()
        for param, stats in patterns.items():
            if stats['cv'] > 0.5:
                recommendations.append(
                    f"Parameter '{param}' varies significantly across conditions "
                    f"(CV={stats['cv']:.2f}). Consider condition-specific tuning."
                )
        
        return recommendations
    
    def get_condition_summary_table(self) -> pd.DataFrame:
        """Get a summary table of all conditions and their performance."""
        data = []
        
        for perf in self.condition_performances:
            data.append({
                'condition': str(perf.condition),
                'trend': perf.condition.trend.value,
                'volatility': perf.condition.volatility.value,
                'human_sharpe': perf.human_sharpe,
                'ml_sharpe': perf.ml_sharpe,
                'improvement': perf.improvement_sharpe,
                'ml_helped': perf.ml_helped
            })
        
        return pd.DataFrame(data)
