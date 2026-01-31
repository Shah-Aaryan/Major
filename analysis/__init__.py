"""
Analysis module for understanding ML optimization effectiveness.

This module answers the core research questions:
- WHEN does ML optimization help?
- HOW does it help?
- WHEN does it fail?
"""

from analysis.condition_analyzer import ConditionAnalyzer, MarketCondition
from analysis.failure_detector import FailureDetector, FailurePattern
from analysis.comparison_report import ComparisonReport, generate_full_report

__all__ = [
    'ConditionAnalyzer',
    'MarketCondition',
    'FailureDetector',
    'FailurePattern',
    'ComparisonReport',
    'generate_full_report'
]
