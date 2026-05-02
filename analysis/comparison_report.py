"""
Comparison Report Generator.

Generates comprehensive reports comparing:
1. Human baseline parameters vs ML-optimized parameters
2. Performance across different market conditions
3. Different optimization methods
4. Failure analysis and recommendations

This is the main output module for research findings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from analysis.condition_analyzer import ConditionAnalyzer, ConditionPerformance
from analysis.failure_detector import FailureDetector, FailurePattern

logger = logging.getLogger(__name__)


def _param_change_reason(param: str, human_val: Any, ml_val: Any) -> str:
    """Heuristic, human-readable rationale for a parameter change.

    This keeps reports explainable without depending on any single optimizer.
    """

    try:
        # RSI parameters
        if param == 'rsi_lookback':
            return (
                'Longer lookback smooths RSI (fewer false signals)'
                if ml_val > human_val
                else 'Shorter lookback makes RSI more reactive (earlier signals)'
            )

        if param == 'rsi_buy_threshold':
            return (
                'Lower buy threshold waits for deeper oversold (more conservative entry)'
                if ml_val < human_val
                else 'Higher buy threshold enters earlier (more trades, more noise risk)'
            )

        if param == 'rsi_sell_threshold':
            return (
                'Higher sell threshold waits for stronger overbought (later exit)'
                if ml_val > human_val
                else 'Lower sell threshold exits earlier (locks profits sooner)'
            )

        # Regime/trend filter
        if param == 'adx_threshold':
            return (
                'Higher ADX threshold treats fewer markets as "trending" (more signals allowed)'
                if ml_val > human_val
                else 'Lower ADX threshold avoids trends more aggressively (fewer mean-reversion traps)'
            )

        # Risk management
        if param == 'stop_loss_pct':
            return (
                'Wider stop gives trades more room (fewer stop-outs)'
                if ml_val > human_val
                else 'Tighter stop cuts losers faster (lower per-trade downside)'
            )

        if param == 'take_profit_pct':
            return (
                'Higher take-profit aims for larger moves (fewer exits)'
                if ml_val > human_val
                else 'Lower take-profit realizes gains sooner (higher hit-rate, smaller wins)'
            )

        if param == 'trailing_stop_pct':
            return (
                'Wider trailing stop allows trends to run (less whipsaw)'
                if ml_val > human_val
                else 'Tighter trailing stop locks profits faster (more premature exits)'
            )

        # Trading frequency / pacing
        if param == 'cooldown_period':
            return (
                'Longer cooldown reduces overtrading in chop'
                if ml_val > human_val
                else 'Shorter cooldown increases trade frequency'
            )

        if param == 'max_trades_per_day':
            return (
                'Higher cap permits more signals (potentially more noise)'
                if ml_val > human_val
                else 'Lower cap forces selectivity (reduces churn)'
            )

        if param == 'max_holding_time':
            return (
                'Longer holds let winners develop (more exposure to reversals)'
                if ml_val > human_val
                else 'Shorter holds reduce time risk (faster mean reversion capture)'
            )

        # Sizing
        if param == 'position_size_pct':
            return (
                'Larger position increases exposure (higher variance)'
                if ml_val > human_val
                else 'Smaller position reduces exposure (lower variance)'
            )

    except Exception:
        return ""

    return ""


@dataclass
class ComparisonReport:
    """
    Comprehensive comparison report for ML optimization research.
    
    Contains all findings, visualizations, and recommendations
    from comparing human vs ML parameter optimization.
    """
    # Report metadata
    generated_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    data_period: str = ""
    
    # Executive summary
    executive_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    
    # Performance comparison
    human_performance: Dict[str, float] = field(default_factory=dict)
    ml_performance: Dict[str, float] = field(default_factory=dict)
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Condition analysis
    condition_analysis: Dict[str, Any] = field(default_factory=dict)
    best_conditions_for_ml: List[str] = field(default_factory=list)
    worst_conditions_for_ml: List[str] = field(default_factory=list)
    
    # Failure analysis
    failures: List[Dict[str, Any]] = field(default_factory=list)
    failure_summary: str = ""
    
    # Method comparison
    method_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_method: str = ""
    
    # Parameter analysis
    parameter_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    most_impactful_params: List[str] = field(default_factory=list)
    
    # Explainability report
    explainability_report_json: Optional[str] = None
    
    # Strategy compatibility check
    strategy_compatibility_report: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    should_use_ml: bool = False
    confidence: float = 0.0
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        md = []
        
        # Title
        md.append(f"# ML Parameter Optimization Research Report")
        md.append(f"**Strategy:** {self.strategy_name}")
        md.append(f"**Data Period:** {self.data_period}")
        md.append(f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append(self.executive_summary)
        md.append("")
        
        # Key Findings
        md.append("### Key Findings")
        for i, finding in enumerate(self.key_findings, 1):
            md.append(f"{i}. {finding}")
        md.append("")
        
        # Strategy Compatibility Check
        if self.strategy_compatibility_report:
            md.append("## ⚠️ Strategy Compatibility Assessment")
            md.append(self.strategy_compatibility_report)
            md.append("")
        
        # Performance Comparison
        md.append("## Performance Comparison")
        md.append("")
        md.append("| Metric | Human Baseline | ML Optimized | Improvement |")
        md.append("|--------|---------------|--------------|-------------|")
        
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']:
            human_val = self.human_performance.get(metric, 'N/A')
            ml_val = self.ml_performance.get(metric, 'N/A')
            imp = self.improvement_metrics.get(metric, 'N/A')
            
            if isinstance(human_val, float):
                human_str = f"{human_val:.4f}"
            else:
                human_str = str(human_val)
            
            if isinstance(ml_val, float):
                ml_str = f"{ml_val:.4f}"
            else:
                ml_str = str(ml_val)
            
            if isinstance(imp, float):
                imp_str = f"{imp:+.2%}"
            else:
                imp_str = str(imp)
            
            md.append(f"| {metric} | {human_str} | {ml_str} | {imp_str} |")
        md.append("")
        
        # Condition Analysis
        md.append("## Condition Analysis")
        md.append("")
        md.append("### When ML Helps Most")
        for cond in self.best_conditions_for_ml[:5]:
            md.append(f"- {cond}")
        md.append("")
        md.append("### When ML Struggles")
        for cond in self.worst_conditions_for_ml[:5]:
            md.append(f"- {cond}")
        md.append("")
        
        # Method Comparison
        if self.method_comparison:
            md.append("## Optimization Method Comparison")
            md.append("")
            md.append("| Method | Avg Improvement | ML Helped Rate | Avg Time |")
            md.append("|--------|-----------------|----------------|----------|")
            
            for method, stats in self.method_comparison.items():
                imp = stats.get('mean_improvement_pct', 0)
                rate = stats.get('ml_helped_rate', 0)
                time = stats.get('mean_time_seconds', 0)
                md.append(f"| {method} | {imp:.2f}% | {rate:.1%} | {time:.1f}s |")
            md.append("")
            md.append(f"**Best Method:** {self.best_method}")
            md.append("")
        
        # Failure Analysis
        if self.failures:
            md.append("## Failure Analysis")
            md.append("")
            md.append(self.failure_summary)
            md.append("")
            md.append("### Detected Failures")
            for failure in self.failures[:5]:
                md.append(f"- **{failure['failure_type']}** (severity: {failure['severity']:.2f})")
                md.append(f"  - {failure.get('root_cause', 'Unknown')[:100]}...")
            md.append("")
        
        # Parameter Analysis
        if self.parameter_changes:
            md.append("## Parameter Analysis")
            md.append("")
            md.append("### Most Impactful Parameters")
            for param in self.most_impactful_params[:5]:
                change = self.parameter_changes.get(param, {})
                md.append(f"- **{param}**: {change.get('human', 'N/A')} -> {change.get('ml', 'N/A')}")
                reason = change.get('reason')
                if reason:
                    md.append(f"  - Reason: {reason}")
            md.append("")
        
        # Explainability Report
        if self.explainability_report_json:
            try:
                import json
                exp_data = json.loads(self.explainability_report_json) if isinstance(self.explainability_report_json, str) else self.explainability_report_json
                
                md.append("## ML Parameter Optimization Explanation")
                md.append("")
                
                # Parameter sensitivities
                if 'sensitivities' in exp_data:
                    md.append("### Parameter Sensitivity Analysis")
                    sensitivities = exp_data['sensitivities']
                    for param_name, sensitivity in list(sensitivities.items())[:5]:
                        impact = sensitivity.get('directional_impact', 'neutral')
                        stability = sensitivity.get('stability', 0)
                        md.append(f"- **{param_name}**")
                        md.append(f"  - Impact: {impact}")
                        md.append(f"  - Stability: {stability:.1%}")
                    md.append("")
                
                # Insights
                if 'insights' in exp_data:
                    insights = exp_data['insights']
                    md.append("### Key Insights")
                    if insights.get('most_impactful'):
                        md.append(f"- **Most Impactful Parameter**: {insights['most_impactful']}")
                    if insights.get('most_stable'):
                        md.append(f"- **Most Stable Parameter**: {insights['most_stable']}")
                    if insights.get('most_volatile'):
                        md.append(f"- **Most Volatile Parameter**: {insights['most_volatile']}")
                    md.append("")
                
                # Attribution
                if 'attribution' in exp_data:
                    attr = exp_data['attribution']
                    md.append("### Performance Attribution")
                    md.append(f"Total Improvement: {attr.get('total_improvement', 0):.4f}")
                    if attr.get('by_parameter'):
                        md.append("By Parameter:")
                        for param, contrib in list(attr['by_parameter'].items())[:5]:
                            md.append(f"- {param}: {contrib:.4f}")
                    md.append("")
                
                # Warnings
                if exp_data.get('warnings'):
                    md.append("### Warnings")
                    for warning in exp_data['warnings']:
                        md.append(f"⚠ {warning}")
                    md.append("")
            except Exception as e:
                logger.warning(f"Failed to format explainability report: {e}")
        
        # Recommendations
        md.append("## Recommendations")
        md.append("")
        for i, rec in enumerate(self.recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")
        
        # Final Verdict
        md.append("## Final Verdict")
        md.append("")
        verdict = "**USE ML-OPTIMIZED PARAMETERS**" if self.should_use_ml else "**KEEP HUMAN BASELINE**"
        md.append(f"{verdict} (Confidence: {self.confidence:.1%})")
        md.append("")
        
        return "\n".join(md)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'generated_at': self.generated_at.isoformat(),
            'strategy_name': self.strategy_name,
            'data_period': self.data_period,
            'executive_summary': self.executive_summary,
            'key_findings': self.key_findings,
            'human_performance': self.human_performance,
            'ml_performance': self.ml_performance,
            'improvement_metrics': self.improvement_metrics,
            'condition_analysis': self.condition_analysis,
            'best_conditions_for_ml': self.best_conditions_for_ml,
            'worst_conditions_for_ml': self.worst_conditions_for_ml,
            'failures': self.failures,
            'failure_summary': self.failure_summary,
            'method_comparison': self.method_comparison,
            'best_method': self.best_method,
            'parameter_changes': self.parameter_changes,
            'most_impactful_params': self.most_impactful_params,
            'explainability_report': self.explainability_report_json,
            'strategy_compatibility_report': self.strategy_compatibility_report,
            'recommendations': self.recommendations,
            'should_use_ml': self.should_use_ml,
            'confidence': self.confidence
        }
    
    def save(self, filepath: str, format: str = 'markdown') -> None:
        """Save report to file."""
        if format == 'markdown':
            with open(filepath, 'w') as f:
                f.write(self.to_markdown())
        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved report to {filepath}")


def generate_full_report(
    strategy_name: str,
    human_results: Dict[str, Any],
    ml_results: Dict[str, Any],
    condition_analyzer: Optional[ConditionAnalyzer] = None,
    failure_detector: Optional[FailureDetector] = None,
    method_comparison: Optional[Dict[str, Dict[str, float]]] = None,
    best_method_override: Optional[str] = None,
    human_params: Optional[Dict[str, Any]] = None,
    ml_params: Optional[Dict[str, Any]] = None,
    data_period: str = "",
    explainability_report_json: Optional[str] = None,
    strategy_compatibility_report: Optional[str] = None
) -> ComparisonReport:
    """
    Generate a comprehensive comparison report.
    
    Args:
        strategy_name: Name of the strategy
        human_results: Backtest results with human params
        ml_results: Backtest results with ML params
        condition_analyzer: Optional condition analysis results
        failure_detector: Optional failure detection results
        method_comparison: Optional method comparison data
        human_params: Human baseline parameters
        ml_params: ML optimized parameters
        data_period: Description of data period
        
    Returns:
        ComparisonReport with all analysis
    """
    report = ComparisonReport(
        strategy_name=strategy_name,
        data_period=data_period,
        explainability_report_json=explainability_report_json,
        strategy_compatibility_report=strategy_compatibility_report
    )
    
    # Extract metrics
    human_metrics = human_results.get('metrics', {})
    ml_metrics = ml_results.get('metrics', {})
    
    if hasattr(human_metrics, 'to_dict'):
        human_metrics = human_metrics.to_dict()
    if hasattr(ml_metrics, 'to_dict'):
        ml_metrics = ml_metrics.to_dict()
    
    report.human_performance = human_metrics
    report.ml_performance = ml_metrics
    
    # Calculate improvements
    improvements = {}
    for metric in human_metrics:
        if metric in ml_metrics:
            h_val = human_metrics[metric]
            m_val = ml_metrics[metric]
            
            if isinstance(h_val, (int, float)) and isinstance(m_val, (int, float)):
                if h_val != 0:
                    improvements[metric] = (m_val - h_val) / abs(h_val)
                else:
                    improvements[metric] = 0 if m_val == 0 else float('inf')
    
    report.improvement_metrics = improvements
    
    # Parameter changes
    if human_params and ml_params:
        for param in human_params:
            if param in ml_params:
                report.parameter_changes[param] = {
                    'human': human_params[param],
                    'ml': ml_params[param],
                    'change_pct': (
                        (ml_params[param] - human_params[param]) / abs(human_params[param]) * 100
                        if isinstance(human_params[param], (int, float)) and human_params[param] != 0
                        else 0
                    ),
                    'reason': _param_change_reason(param, human_params[param], ml_params[param])
                }
        
        # Find most impactful (largest changes)
        sorted_params = sorted(
            report.parameter_changes.items(),
            key=lambda x: abs(x[1].get('change_pct', 0)),
            reverse=True
        )
        report.most_impactful_params = [p[0] for p in sorted_params[:5]]
    
    # Condition analysis
    if condition_analyzer:
        analysis = condition_analyzer._generate_analysis()
        report.condition_analysis = analysis
        
        # Extract best/worst conditions
        best_conds = analysis.get('best_conditions', [])
        worst_conds = analysis.get('worst_conditions', [])
        
        report.best_conditions_for_ml = [c['condition'] for c in best_conds]
        report.worst_conditions_for_ml = [c['condition'] for c in worst_conds]
    
    # Failure analysis
    if failure_detector:
        report.failures = [f.to_dict() for f in failure_detector.failures]
        
        summary = failure_detector.get_failure_summary()
        report.failure_summary = (
            f"Detected {summary['n_failures']} failures. "
            f"Most severe: {summary.get('most_severe', {}).get('failure_type', 'N/A')}"
        )
    
    # Method comparison
    if method_comparison:
        report.method_comparison = method_comparison
        
        # Find best method
        best_method = max(
            method_comparison.items(),
            key=lambda x: x[1].get('mean_improvement_pct', 0)
        )
        report.best_method = best_method[0]

    if best_method_override:
        report.best_method = best_method_override
    
    # Generate key findings
    key_findings = []
    
    sharpe_imp = improvements.get('sharpe_ratio', 0)
    if sharpe_imp > 0.1:
        key_findings.append(
            f"ML optimization improved Sharpe ratio by {sharpe_imp:.1%}"
        )
    elif sharpe_imp < -0.1:
        key_findings.append(
            f"ML optimization DEGRADED Sharpe ratio by {abs(sharpe_imp):.1%}"
        )
    else:
        key_findings.append(
            "ML optimization provided minimal improvement to Sharpe ratio"
        )
    
    if report.best_conditions_for_ml:
        key_findings.append(
            f"ML works best in: {', '.join(report.best_conditions_for_ml[:3])}"
        )
    
    if report.failures:
        key_findings.append(
            f"Warning: {len(report.failures)} optimization failures detected"
        )
    
    report.key_findings = key_findings
    
    # Executive summary
    ml_helped = sharpe_imp > 0.05
    
    report.executive_summary = (
        f"This report analyzes ML parameter optimization for the {strategy_name} strategy. "
        f"The ML optimization {'improved' if ml_helped else 'did not improve'} "
        f"performance compared to human baseline parameters. "
        f"Sharpe ratio changed from {human_metrics.get('sharpe_ratio', 0):.2f} "
        f"to {ml_metrics.get('sharpe_ratio', 0):.2f} "
        f"({sharpe_imp:+.1%} change)."
    )
    
    # Recommendations
    recommendations = []
    
    if ml_helped:
        recommendations.append(
            "Consider using ML-optimized parameters for this strategy."
        )
    else:
        recommendations.append(
            "Keep using human baseline parameters - ML did not provide value."
        )
    
    if failure_detector and failure_detector.failures:
        should_use_baseline, reason = failure_detector.should_use_baseline()
        if should_use_baseline:
            recommendations.append(f"CAUTION: {reason}")
            recommendations.append("Consider hybrid approach or baseline parameters.")
    
    if condition_analyzer:
        recs = condition_analyzer._generate_recommendations()
        recommendations.extend(recs[:3])
    
    report.recommendations = recommendations
    
    # Final verdict
    report.should_use_ml = ml_helped and not (
        failure_detector and failure_detector.should_use_baseline()[0]
    )
    
    # Confidence based on consistency and lack of failures
    base_confidence = 0.5
    if ml_helped:
        base_confidence += 0.2
    if not report.failures:
        base_confidence += 0.2
    if report.method_comparison and len(report.method_comparison) > 1:
        base_confidence += 0.1
    
    report.confidence = min(base_confidence, 1.0)
    
    return report


class ResearchSummaryGenerator:
    """
    Generates research summaries across multiple strategies and conditions.
    
    Useful for answering broader research questions about ML optimization.
    """
    
    def __init__(self):
        self.reports: List[ComparisonReport] = []
    
    def add_report(self, report: ComparisonReport) -> None:
        """Add a report to the summary."""
        self.reports.append(report)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary across all reports."""
        if not self.reports:
            return {'error': 'No reports to summarize'}
        
        summary = {
            'n_reports': len(self.reports),
            'strategies_analyzed': list(set(r.strategy_name for r in self.reports)),
            'ml_recommended_count': sum(1 for r in self.reports if r.should_use_ml),
            'avg_confidence': np.mean([r.confidence for r in self.reports]),
            'findings': self._aggregate_findings(),
            'overall_recommendation': self._overall_recommendation()
        }
        
        return summary
    
    def _aggregate_findings(self) -> Dict[str, Any]:
        """Aggregate findings across reports."""
        all_improvements = []
        all_failures = 0
        
        for report in self.reports:
            imp = report.improvement_metrics.get('sharpe_ratio', 0)
            all_improvements.append(imp)
            all_failures += len(report.failures)
        
        return {
            'avg_sharpe_improvement': np.mean(all_improvements),
            'std_sharpe_improvement': np.std(all_improvements),
            'total_failures': all_failures,
            'pct_improved': sum(1 for i in all_improvements if i > 0.05) / len(all_improvements)
        }
    
    def _overall_recommendation(self) -> str:
        """Generate overall recommendation."""
        findings = self._aggregate_findings()
        
        if findings['pct_improved'] > 0.6:
            return (
                "ML optimization is generally beneficial. "
                "Consider adopting ML-adjusted parameters with condition-specific tuning."
            )
        elif findings['pct_improved'] < 0.3:
            return (
                "ML optimization rarely helps in this research. "
                "Human parameters appear to be well-tuned already."
            )
        else:
            return (
                "ML optimization shows mixed results. "
                "Use ML selectively based on market conditions."
            )
