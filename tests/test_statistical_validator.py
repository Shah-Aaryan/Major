"""
Unit tests for the multi-seed statistical validator.
"""

import pytest
import os
import numpy as np
from analysis.statistical_validator import StatisticalValidator


def test_statistical_validator_calculations(tmp_path):
    validator = StatisticalValidator()
    
    np.random.seed(42)
    baseline = np.random.normal(10.0, 1.0, 30).tolist()
    treatment = np.random.normal(12.0, 1.0, 30).tolist()
    
    res = validator.compare_distributions(baseline, treatment, "Default", "ML-Optimized")
    
    assert res['statistically_significant'] is True
    assert res['cohens_d'] > 0.5
    assert len(res['ci_95_diff']) == 2
    
    report_path = str(tmp_path / "stat_report.md")
    validator.generate_report(res, report_path)
    assert os.path.exists(report_path)
