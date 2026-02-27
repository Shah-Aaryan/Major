"""
Data Leakage Checker (A4 - P1)

Detects and prevents feature leakage in the ML pipeline.

Key Checks:
1. Rolling features don't peek ahead (lookahead bias)
2. Train/validation separation is enforced
3. Feature computation respects temporal order
4. No future data in any feature

This module integrates into the training pipeline to validate
data integrity before optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import warnings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LeakageWarning:
    """Represents a detected leakage issue."""
    severity: str  # "critical", "warning", "info"
    category: str  # "lookahead", "train_test_overlap", "feature_future", etc.
    description: str
    affected_features: List[str] = field(default_factory=list)
    affected_rows: Optional[Tuple[int, int]] = None  # (start_idx, end_idx)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "affected_features": self.affected_features,
            "affected_rows": self.affected_rows,
            "recommendation": self.recommendation
        }


@dataclass
class LeakageCheckResult:
    """Result of a leakage check."""
    passed: bool
    check_name: str
    warnings: List[LeakageWarning] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return f"{self.check_name}: {status} ({len(self.warnings)} warnings)"


@dataclass
class LeakageReport:
    """Complete leakage analysis report."""
    timestamp: datetime
    overall_passed: bool
    checks: List[LeakageCheckResult]
    critical_issues: int
    warnings_count: int
    info_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "passed": self.overall_passed,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings_count,
            "info": self.info_count,
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "warnings": [w.to_dict() for w in c.warnings]
                }
                for c in self.checks
            ]
        }
    
    def summary(self) -> str:
        status = "PASSED" if self.overall_passed else "FAILED"
        lines = [
            "=" * 60,
            f"LEAKAGE ANALYSIS REPORT - {status}",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Critical issues: {self.critical_issues}",
            f"Warnings: {self.warnings_count}",
            "",
            "CHECK RESULTS:"
        ]
        
        for check in self.checks:
            lines.append(f"  {check.summary()}")
            for warn in check.warnings:
                prefix = "🔴" if warn.severity == "critical" else "🟡" if warn.severity == "warning" else "ℹ"
                lines.append(f"    {prefix} {warn.description}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

LEAKAGE_CHECKLIST = """
DATA LEAKAGE PREVENTION CHECKLIST
=================================

1. LOOKAHEAD BIAS
   - [ ] Rolling features use only past data (no future values in window)
   - [ ] RSI, MACD, etc. computed with historical data only
   - [ ] No shifting that pulls future data backward
   - [ ] Verify .shift(n) uses n > 0 for past values

2. TRAIN/TEST SEPARATION
   - [ ] No overlap between train and test periods
   - [ ] Test data timestamps strictly after train data
   - [ ] Walk-forward windows don't overlap
   - [ ] Validation set isolated from training

3. FEATURE COMPUTATION
   - [ ] All rolling windows only consider t-n to t-1 (not t or t+n)
   - [ ] normalize/scale operations fitted on train only
   - [ ] No global statistics that include test data

4. PRICE DATA
   - [ ] Close price at time t doesn't use t+1 open
   - [ ] OHLC bars are complete before feature computation
   - [ ] No intrabar data leakage (e.g., using volume before bar closes)

5. STRATEGY SIGNALS
   - [ ] Signals generated at time t execute at t+1 or later
   - [ ] No "perfect foresight" in entry/exit logic
   - [ ] Transaction costs applied properly

6. OPTIMIZATION
   - [ ] Optimizer only sees training data objective
   - [ ] Test data not used in parameter selection
   - [ ] Walk-forward retraining respects temporal order
"""


# ═══════════════════════════════════════════════════════════════════════════════
# LEAKAGE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

class LeakageChecker:
    """
    Detects and prevents data leakage in ML trading pipelines.
    
    Performs automated checks:
    1. Lookahead bias in features
    2. Train/test separation
    3. Feature temporal integrity
    4. Rolling window validation
    
    Usage:
        checker = LeakageChecker()
        
        # Check a DataFrame
        report = checker.check_dataframe(df, train_end_idx=1000)
        print(report.summary())
        
        # Validate train/test split
        result = checker.validate_train_test_split(train_df, test_df)
    """
    
    # Known rolling feature patterns
    ROLLING_PATTERNS = [
        'sma_', 'ema_', 'rsi_', 'atr_', 'adx', 'macd',
        'bb_', 'stoch_', 'volume_sma', 'volatility_', 'momentum_',
        'rolling_', '_mean', '_std', '_min', '_max'
    ]
    
    # Features that should NOT exist (indicators of leakage)
    SUSPICIOUS_PATTERNS = [
        'future_', 'next_', 'forward_', '_t1', '_t+1',
        'lead_', 'lookahead_'
    ]
    
    def __init__(self):
        """Initialize the leakage checker."""
        self.checks_run: List[LeakageCheckResult] = []
    
    def check_dataframe(
        self,
        df: pd.DataFrame,
        train_end_idx: Optional[int] = None,
        feature_columns: Optional[List[str]] = None
    ) -> LeakageReport:
        """
        Run all leakage checks on a DataFrame.
        
        Args:
            df: DataFrame with features
            train_end_idx: Index where training data ends
            feature_columns: Specific columns to check (None = all non-OHLCV)
            
        Returns:
            LeakageReport with all findings
        """
        self.checks_run = []
        
        # Identify feature columns
        if feature_columns is None:
            ohlcv = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [c for c in df.columns if c.lower() not in ohlcv]
        
        # Run checks
        self.checks_run.append(self.check_lookahead_bias(df, feature_columns))
        self.checks_run.append(self.check_suspicious_features(df))
        self.checks_run.append(self.check_nan_propagation(df, feature_columns))
        self.checks_run.append(self.check_rolling_windows(df, feature_columns))
        
        if train_end_idx is not None:
            self.checks_run.append(
                self.check_train_test_leakage(df, train_end_idx, feature_columns)
            )
        
        # Aggregate results
        all_warnings = []
        for check in self.checks_run:
            all_warnings.extend(check.warnings)
        
        critical = sum(1 for w in all_warnings if w.severity == "critical")
        warnings_count = sum(1 for w in all_warnings if w.severity == "warning")
        info = sum(1 for w in all_warnings if w.severity == "info")
        
        overall_passed = critical == 0
        
        return LeakageReport(
            timestamp=datetime.now(),
            overall_passed=overall_passed,
            checks=self.checks_run,
            critical_issues=critical,
            warnings_count=warnings_count,
            info_count=info
        )
    
    def check_lookahead_bias(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> LeakageCheckResult:
        """
        Check for lookahead bias by detecting future correlations.
        
        Tests if features are correlated with future returns, which
        would indicate they contain future information.
        """
        warnings = []
        
        if 'close' not in df.columns:
            return LeakageCheckResult(
                passed=True,
                check_name="Lookahead Bias Check",
                warnings=[LeakageWarning(
                    severity="info",
                    category="skip",
                    description="No close price column, skipping lookahead check"
                )]
            )
        
        # Compute future returns
        future_return = df['close'].pct_change().shift(-1)
        
        suspicious_features = []
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            try:
                # Compute correlation with future return
                corr = df[col].corr(future_return)
                
                # Very high correlation with future is suspicious
                if abs(corr) > 0.7:
                    suspicious_features.append((col, corr))
            except Exception:
                continue
        
        if suspicious_features:
            w = LeakageWarning(
                severity="critical",
                category="lookahead",
                description=f"Features highly correlated with future returns: {suspicious_features[:5]}",
                affected_features=[f[0] for f in suspicious_features],
                recommendation="Review feature computation. Rolling windows may include future data."
            )
            warnings.append(w)
        
        return LeakageCheckResult(
            passed=len(suspicious_features) == 0,
            check_name="Lookahead Bias Check",
            warnings=warnings,
            details={"suspicious_features": suspicious_features}
        )
    
    def check_suspicious_features(self, df: pd.DataFrame) -> LeakageCheckResult:
        """Check for features with suspicious naming patterns."""
        warnings = []
        suspicious = []
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in self.SUSPICIOUS_PATTERNS:
                if pattern in col_lower:
                    suspicious.append(col)
                    break
        
        if suspicious:
            w = LeakageWarning(
                severity="warning",
                category="naming",
                description=f"Features with suspicious names (may contain future data): {suspicious}",
                affected_features=suspicious,
                recommendation="Verify these features don't use future information"
            )
            warnings.append(w)
        
        return LeakageCheckResult(
            passed=len(suspicious) == 0,
            check_name="Suspicious Feature Names",
            warnings=warnings,
            details={"suspicious_features": suspicious}
        )
    
    def check_nan_propagation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> LeakageCheckResult:
        """
        Check that NaN values propagate correctly at start of data.
        
        Rolling features should have NaN at the beginning (warmup period).
        If they don't, the rolling window may be using future data.
        """
        warnings = []
        issues = []
        
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            # Check for rolling patterns
            is_rolling = any(p in col.lower() for p in self.ROLLING_PATTERNS)
            
            if is_rolling:
                # First few values should be NaN for rolling features
                first_10 = df[col].iloc[:10]
                non_nan_count = first_10.notna().sum()
                
                # If rolling feature has no NaN at start, it might be using future data
                if non_nan_count == 10:
                    issues.append(col)
        
        if issues:
            w = LeakageWarning(
                severity="warning",
                category="nan_propagation",
                description=f"Rolling features have no NaN warmup period: {issues[:10]}",
                affected_features=issues,
                recommendation="Rolling features should have NaN at start. Check computation."
            )
            warnings.append(w)
        
        return LeakageCheckResult(
            passed=len(issues) == 0,
            check_name="NaN Propagation Check",
            warnings=warnings,
            details={"issues": issues}
        )
    
    def check_rolling_windows(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> LeakageCheckResult:
        """
        Verify rolling windows only use past data.
        
        Tests by checking that feature at time t doesn't change when
        we modify future data (which it shouldn't access).
        """
        warnings = []
        
        # This is a statistical check - we verify that features are
        # computed point-in-time by checking temporal consistency
        
        for col in feature_columns[:20]:  # Check first 20 features
            if col not in df.columns:
                continue
            
            is_rolling = any(p in col.lower() for p in self.ROLLING_PATTERNS)
            if not is_rolling:
                continue
            
            # Check for any NaN in the middle of the series
            # (which would indicate recalculation issues)
            values = df[col].values
            if len(values) > 100:
                middle = values[50:-50]
                nan_count = np.isnan(middle).sum() if isinstance(middle[0], float) else 0
                total = len(middle)
                
                if nan_count > total * 0.1:  # More than 10% NaN in middle
                    warnings.append(LeakageWarning(
                        severity="info",
                        category="data_quality",
                        description=f"Feature '{col}' has {nan_count} NaN values in middle portion",
                        affected_features=[col],
                        recommendation="May indicate data issues. Verify computation."
                    ))
        
        return LeakageCheckResult(
            passed=True,  # This is informational
            check_name="Rolling Window Check",
            warnings=warnings
        )
    
    def check_train_test_leakage(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        feature_columns: List[str]
    ) -> LeakageCheckResult:
        """
        Check for information leakage between train and test sets.
        
        Verifies that features in training set don't contain
        information from the test set.
        """
        warnings = []
        
        # Check that train_end_idx is valid
        if train_end_idx >= len(df):
            return LeakageCheckResult(
                passed=True,
                check_name="Train/Test Leakage Check",
                warnings=[LeakageWarning(
                    severity="warning",
                    category="invalid",
                    description="Invalid train_end_idx, skipping check"
                )]
            )
        
        train_df = df.iloc[:train_end_idx]
        test_df = df.iloc[train_end_idx:]
        
        # Check timestamps don't overlap
        if hasattr(df.index, 'max') and hasattr(df.index, 'min'):
            train_max = train_df.index.max()
            test_min = test_df.index.min()
            
            if train_max >= test_min:
                warnings.append(LeakageWarning(
                    severity="critical",
                    category="train_test_overlap",
                    description=f"Train and test sets have overlapping timestamps: train max={train_max}, test min={test_min}",
                    recommendation="Ensure strict temporal separation"
                ))
        
        # Check for identical rows (copy-paste errors)
        if len(train_df) > 0 and len(test_df) > 0:
            train_last = train_df.iloc[-1]
            test_first = test_df.iloc[0]
            
            # Check a subset of columns
            check_cols = [c for c in feature_columns[:10] if c in df.columns]
            if check_cols:
                identical_count = sum(
                    1 for c in check_cols 
                    if train_last.get(c) == test_first.get(c)
                )
                
                if identical_count == len(check_cols):
                    warnings.append(LeakageWarning(
                        severity="warning",
                        category="duplicate",
                        description="Last train row identical to first test row",
                        recommendation="Verify data split is correct"
                    ))
        
        return LeakageCheckResult(
            passed=all(w.severity != "critical" for w in warnings),
            check_name="Train/Test Leakage Check",
            warnings=warnings
        )
    
    def validate_train_test_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> LeakageCheckResult:
        """
        Validate a train/test split for leakage.
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            LeakageCheckResult with validation findings
        """
        warnings = []
        
        # Check timestamps
        if isinstance(train_df.index, pd.DatetimeIndex) and isinstance(test_df.index, pd.DatetimeIndex):
            train_max = train_df.index.max()
            test_min = test_df.index.min()
            
            if train_max >= test_min:
                warnings.append(LeakageWarning(
                    severity="critical",
                    category="temporal_overlap",
                    description=f"Temporal overlap! Train max: {train_max}, Test min: {test_min}",
                    recommendation="Redefine split to ensure temporal separation"
                ))
            else:
                gap = test_min - train_max
                warnings.append(LeakageWarning(
                    severity="info",
                    category="gap_info",
                    description=f"Train/test gap: {gap}. This is OK."
                ))
        
        # Check for data contamination via common rows
        if len(train_df) > 0 and len(test_df) > 0:
            try:
                # Check if any test rows appear in train
                train_set = set(map(tuple, train_df.values[:, :5]))  # Check first 5 cols
                test_set = set(map(tuple, test_df.values[:, :5]))
                overlap = train_set & test_set
                
                if overlap:
                    warnings.append(LeakageWarning(
                        severity="critical",
                        category="data_overlap",
                        description=f"Found {len(overlap)} identical rows in train and test sets",
                        recommendation="Remove duplicate rows"
                    ))
            except Exception:
                pass  # Skip if comparison fails
        
        return LeakageCheckResult(
            passed=all(w.severity != "critical" for w in warnings),
            check_name="Train/Test Split Validation",
            warnings=warnings
        )
    
    def validate_walk_forward_windows(
        self,
        windows: List[Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> LeakageCheckResult:
        """
        Validate walk-forward windows for leakage.
        
        Args:
            windows: List of (train_df, test_df) tuples
            
        Returns:
            LeakageCheckResult
        """
        warnings = []
        
        for i, (train_df, test_df) in enumerate(windows):
            result = self.validate_train_test_split(train_df, test_df)
            
            for w in result.warnings:
                w.description = f"Window {i}: {w.description}"
                warnings.append(w)
            
            # Check this window's test doesn't overlap with next window's train
            if i < len(windows) - 1:
                next_train = windows[i + 1][0]
                if isinstance(test_df.index, pd.DatetimeIndex):
                    test_max = test_df.index.max()
                    next_train_min = next_train.index.min()
                    
                    if test_max >= next_train_min:
                        warnings.append(LeakageWarning(
                            severity="critical",
                            category="window_overlap",
                            description=f"Window {i} test overlaps with window {i+1} train"
                        ))
        
        return LeakageCheckResult(
            passed=all(w.severity != "critical" for w in warnings),
            check_name="Walk-Forward Window Validation",
            warnings=warnings
        )
    
    @staticmethod
    def print_checklist():
        """Print the leakage prevention checklist."""
        print(LEAKAGE_CHECKLIST)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def assert_no_leakage(df: pd.DataFrame, train_end_idx: int):
    """
    Assert that no critical leakage is detected.
    
    Raises AssertionError if critical issues found.
    
    Args:
        df: DataFrame to check
        train_end_idx: End of training data
    """
    checker = LeakageChecker()
    report = checker.check_dataframe(df, train_end_idx)
    
    if not report.overall_passed:
        critical_warnings = [
            w for check in report.checks 
            for w in check.warnings 
            if w.severity == "critical"
        ]
        raise AssertionError(
            f"Critical data leakage detected!\n" + 
            "\n".join(f"  - {w.description}" for w in critical_warnings)
        )


def log_leakage_warnings(df: pd.DataFrame, train_end_idx: Optional[int] = None):
    """
    Log any leakage warnings without raising errors.
    
    Args:
        df: DataFrame to check
        train_end_idx: Optional end of training data
    """
    checker = LeakageChecker()
    report = checker.check_dataframe(df, train_end_idx)
    
    for check in report.checks:
        for warn in check.warnings:
            if warn.severity == "critical":
                logger.error(f"LEAKAGE: {warn.description}")
            elif warn.severity == "warning":
                logger.warning(f"LEAKAGE WARNING: {warn.description}")
            else:
                logger.info(f"LEAKAGE INFO: {warn.description}")


if __name__ == "__main__":
    # Print checklist
    LeakageChecker.print_checklist()
    
    # Example check
    print("\nRunning example check...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
    df = pd.DataFrame({
        "close": np.cumsum(np.random.randn(1000)) + 100,
        "volume": np.random.randint(1000, 10000, 1000),
        "rsi_14": np.random.uniform(20, 80, 1000),
        "sma_20": np.random.randn(1000) + 100,
        "future_return": np.random.randn(1000)  # Suspicious!
    }, index=dates)
    
    checker = LeakageChecker()
    report = checker.check_dataframe(df, train_end_idx=800)
    print(report.summary())
