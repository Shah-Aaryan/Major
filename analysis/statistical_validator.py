"""
Multi-Seed Statistical Validation Engine.

Provides rigorous statistical evaluation for trading strategy parameters and ML optimizers:
- Multi-seed Monte Carlo replication
- Parametric (t-test) and Non-Parametric (Wilcoxon signed-rank) hypothesis testing
- Effect size (Cohen's d) computation
- Bootstrapped 95% Confidence Intervals
- Normality testing (Shapiro-Wilk)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class StatisticalValidator:
    """
    Statistical validation engine for multi-seed strategy evaluation and optimizer comparison.
    """

    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap

    def compute_summary_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculates central tendency and dispersion metrics."""
        arr = np.asarray(data, dtype=float)
        if len(arr) == 0:
            return {}

        q25, q75 = np.percentile(arr, [25, 75])
        return {
            'count': float(len(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'median': float(np.median(arr)),
            'iqr': float(q75 - q25),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'skewness': float(stats.skew(arr)) if len(arr) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(arr)) if len(arr) > 3 else 0.0
        }

    def bootstrap_ci(self, data: List[float], func=np.mean) -> Tuple[float, float]:
        """Calculates non-parametric bootstrap confidence interval."""
        arr = np.asarray(data, dtype=float)
        if len(arr) < 2:
            val = float(arr[0]) if len(arr) == 1 else 0.0
            return val, val

        rng = np.random.default_rng(42)
        boot_dist = [
            func(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(self.n_bootstrap)
        ]
        lower_pct = (1.0 - self.confidence_level) / 2.0 * 100
        upper_pct = (1.0 + self.confidence_level) / 2.0 * 100
        ci_lower = float(np.percentile(boot_dist, lower_pct))
        ci_upper = float(np.percentile(boot_dist, upper_pct))
        return ci_lower, ci_upper

    def compare_distributions(
        self,
        baseline: List[float],
        treatment: List[float],
        label_a: str = "Baseline",
        label_b: str = "Treatment"
    ) -> Dict[str, Any]:
        """
        Executes complete hypothesis test comparing treatment vs baseline performance.
        """
        arr_a = np.asarray(baseline, dtype=float)
        arr_b = np.asarray(treatment, dtype=float)

        stats_a = self.compute_summary_statistics(arr_a)
        stats_b = self.compute_summary_statistics(arr_b)

        # Normality tests
        shapiro_a = stats.shapiro(arr_a).pvalue if len(arr_a) >= 3 else 1.0
        shapiro_b = stats.shapiro(arr_b).pvalue if len(arr_b) >= 3 else 1.0
        is_normal = (shapiro_a > 0.05) and (shapiro_b > 0.05)

        # Hypothesis test
        if is_normal and len(arr_a) == len(arr_b) and len(arr_a) >= 2:
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(arr_b, arr_a)
            test_type = "Paired t-test"
        else:
            # Wilcoxon signed-rank or Mann-Whitney
            if len(arr_a) == len(arr_b) and len(arr_a) >= 3:
                try:
                    t_stat, p_val = stats.wilcoxon(arr_b, arr_a)
                    test_type = "Wilcoxon signed-rank test"
                except Exception:
                    t_stat, p_val = stats.mannwhitneyu(arr_b, arr_a)
                    test_type = "Mann-Whitney U test"
            else:
                t_stat, p_val = stats.mannwhitneyu(arr_b, arr_a)
                test_type = "Mann-Whitney U test"

        # Effect Size (Cohen's d)
        pooled_std = np.sqrt((np.var(arr_a, ddof=1) + np.var(arr_b, ddof=1)) / 2.0) + 1e-10
        cohens_d = float((np.mean(arr_b) - np.mean(arr_a)) / pooled_std)

        # Bootstrap CIs for mean difference
        diffs = arr_b - arr_a if len(arr_a) == len(arr_b) else arr_b
        ci_low, ci_high = self.bootstrap_ci(diffs, np.mean)

        return {
            'label_a': label_a,
            'label_b': label_b,
            'stats_a': stats_a,
            'stats_b': stats_b,
            'test_type': test_type,
            'statistic': float(t_stat),
            'p_value': float(p_val),
            'statistically_significant': float(p_val) < 0.05,
            'cohens_d': cohens_d,
            'cohens_d_interpretation': (
                "large" if abs(cohens_d) >= 0.8 else
                "medium" if abs(cohens_d) >= 0.5 else
                "small" if abs(cohens_d) >= 0.2 else "negligible"
            ),
            'ci_95_diff': (ci_low, ci_high)
        }

    def generate_report(self, comp_results: Dict[str, Any], output_path: str) -> str:
        """Saves a Markdown statistical validation report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("# Multi-Seed Statistical Validation Report\n\n")
            f.write(f"**Baseline ({comp_results['label_a']}) vs Treatment ({comp_results['label_b']})**\n\n")
            
            f.write("## Summary Statistics\n")
            f.write(f"- **{comp_results['label_a']} Mean**: {comp_results['stats_a'].get('mean', 0):.4f} (Std: {comp_results['stats_a'].get('std', 0):.4f})\n")
            f.write(f"- **{comp_results['label_b']} Mean**: {comp_results['stats_b'].get('mean', 0):.4f} (Std: {comp_results['stats_b'].get('std', 0):.4f})\n\n")
            
            f.write("## Hypothesis Test Results\n")
            f.write(f"- **Test Used**: {comp_results['test_type']}\n")
            f.write(f"- **Test Statistic**: {comp_results['statistic']:.4f}\n")
            f.write(f"- **p-value**: {comp_results['p_value']:.4e}\n")
            f.write(f"- **Statistically Significant (p < 0.05)**: {'Yes' if comp_results['statistically_significant'] else 'No'}\n")
            f.write(f"- **Cohen's d Effect Size**: {comp_results['cohens_d']:.4f} ({comp_results['cohens_d_interpretation']})\n")
            f.write(f"- **95% Bootstrap CI of Diff**: [{comp_results['ci_95_diff'][0]:.4f}, {comp_results['ci_95_diff'][1]:.4f}]\n")
            
        logger.info(f"Statistical validation report saved to {output_path}")
        return output_path
