"""
Data Preprocessing Module.

Handles data cleaning, missing value imputation, normalization,
and data quality validation for OHLCV cryptocurrency data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Report on data quality metrics."""
    symbol: str
    total_rows: int
    missing_values: Dict[str, int]
    missing_pct: Dict[str, float]
    duplicates: int
    gaps: int
    gap_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    invalid_candles: int
    outliers: Dict[str, int]
    quality_score: float  # 0-100


class DataPreprocessor:
    """
    Preprocesses OHLCV data for ML trading research.
    
    This class handles:
    - Missing value detection and imputation
    - Outlier detection
    - Data normalization
    - Gap detection and handling
    - Data quality reporting
    """
    
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    
    def __init__(
        self,
        max_missing_pct: float = 0.05,
        forward_fill_limit: int = 5,
        normalize_prices: bool = True,
        normalize_volume: bool = True,
        normalization_window: int = 1440
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_missing_pct: Maximum allowed percentage of missing values
            forward_fill_limit: Maximum consecutive NaNs to forward fill
            normalize_prices: Whether to normalize price columns
            normalize_volume: Whether to normalize volume
            normalization_window: Rolling window for normalization (in candles)
        """
        self.max_missing_pct = max_missing_pct
        self.forward_fill_limit = forward_fill_limit
        self.normalize_prices = normalize_prices
        self.normalize_volume = normalize_volume
        self.normalization_window = normalization_window
        
        self._normalization_params: Dict[str, Dict] = {}
    
    def preprocess(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Full preprocessing pipeline for OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            symbol: Symbol name for reporting
            
        Returns:
            Tuple of (preprocessed DataFrame, quality report)
        """
        # Generate initial quality report
        report = self.analyze_quality(df, symbol)
        
        # Check if data quality is acceptable
        if report.quality_score < 50:
            logger.warning(f"Low data quality score for {symbol}: {report.quality_score:.1f}")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Step 1: Handle duplicates
        df = self._remove_duplicates(df)
        
        # Step 2: Handle missing timestamps (gaps)
        df = self._handle_gaps(df)
        
        # Step 3: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 4: Fix invalid candles
        df = self._fix_invalid_candles(df)
        
        # Step 5: Handle outliers
        df = self._handle_outliers(df)
        
        # Step 6: Normalize data
        if self.normalize_prices or self.normalize_volume:
            df = self._normalize_data(df)
        
        # Update report with cleaned data stats
        report.total_rows = len(df)
        
        logger.info(
            f"Preprocessed {symbol}: {report.total_rows} rows, "
            f"quality score: {report.quality_score:.1f}"
        )
        
        return df, report
    
    def analyze_quality(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> DataQualityReport:
        """
        Analyze data quality and generate a report.
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name
            
        Returns:
            DataQualityReport with quality metrics
        """
        total_rows = len(df)
        
        # Missing values
        missing_values = df[self.OHLCV_COLUMNS].isnull().sum().to_dict()
        missing_pct = {
            col: count / total_rows if total_rows > 0 else 0
            for col, count in missing_values.items()
        }
        
        # Duplicates
        duplicates = df.index.duplicated().sum()
        
        # Gaps (missing timestamps)
        gaps, gap_periods = self._detect_gaps(df)
        
        # Invalid candles (high < low, etc.)
        invalid_candles = self._count_invalid_candles(df)
        
        # Outliers
        outliers = self._detect_outliers(df)
        
        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(
            total_rows=total_rows,
            missing_pct=max(missing_pct.values()) if missing_pct else 0,
            duplicate_pct=duplicates / total_rows if total_rows > 0 else 0,
            gap_pct=gaps / total_rows if total_rows > 0 else 0,
            invalid_pct=invalid_candles / total_rows if total_rows > 0 else 0,
            outlier_pct=sum(outliers.values()) / (total_rows * len(outliers)) if total_rows > 0 and outliers else 0
        )
        
        return DataQualityReport(
            symbol=symbol,
            total_rows=total_rows,
            missing_values=missing_values,
            missing_pct=missing_pct,
            duplicates=duplicates,
            gaps=gaps,
            gap_periods=gap_periods,
            invalid_candles=invalid_candles,
            outliers=outliers,
            quality_score=quality_score
        )
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the last occurrence."""
        if df.index.duplicated().any():
            n_duplicates = df.index.duplicated().sum()
            logger.info(f"Removing {n_duplicates} duplicate timestamps")
            df = df[~df.index.duplicated(keep='last')]
        return df
    
    def _detect_gaps(
        self,
        df: pd.DataFrame
    ) -> Tuple[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """
        Detect gaps (missing timestamps) in the data.
        
        Returns:
            Tuple of (number of missing candles, list of gap periods)
        """
        if len(df) < 2:
            return 0, []
        
        # Infer the expected frequency
        time_diffs = df.index.to_series().diff().dropna()
        expected_freq = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(minutes=1)
        
        # Find gaps (where diff > expected frequency * 1.5)
        tolerance = expected_freq * 1.5
        gaps = time_diffs > tolerance
        
        gap_periods = []
        if gaps.any():
            gap_indices = gaps[gaps].index
            for idx in gap_indices:
                loc = df.index.get_loc(idx)
                if loc > 0:
                    start = df.index[loc - 1]
                    end = idx
                    gap_periods.append((start, end))
        
        # Calculate total missing candles
        total_missing = 0
        for start, end in gap_periods:
            gap_duration = end - start
            missing_candles = int(gap_duration / expected_freq) - 1
            total_missing += missing_candles
        
        return total_missing, gap_periods
    
    def _handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle gaps by reindexing with expected frequency.
        
        For small gaps, forward fill. For large gaps, leave as NaN.
        """
        if len(df) < 2:
            return df
        
        # Infer frequency
        time_diffs = df.index.to_series().diff().dropna()
        freq = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(minutes=1)
        
        # Create complete index
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Reindex
        original_len = len(df)
        df = df.reindex(full_index)
        new_len = len(df)
        
        if new_len > original_len:
            logger.info(f"Added {new_len - original_len} rows to fill gaps")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in OHLCV columns."""
        for col in self.OHLCV_COLUMNS:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(df)
            
            if missing_pct > self.max_missing_pct:
                logger.warning(
                    f"High missing rate in {col}: {missing_pct:.2%} "
                    f"(threshold: {self.max_missing_pct:.2%})"
                )
            
            # Forward fill with limit
            df[col] = df[col].fillna(method='ffill', limit=self.forward_fill_limit)
            
            # Backward fill remaining (at the start)
            df[col] = df[col].fillna(method='bfill', limit=self.forward_fill_limit)
        
        # Drop rows that still have NaN in critical columns
        critical_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=critical_cols)
        
        return df
    
    def _count_invalid_candles(self, df: pd.DataFrame) -> int:
        """Count candles with invalid OHLC relationships."""
        invalid = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        return invalid.sum()
    
    def _fix_invalid_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix candles with invalid OHLC relationships."""
        # Fix high < low by swapping
        mask = df['high'] < df['low']
        if mask.any():
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
            logger.info(f"Fixed {mask.sum()} candles with high < low")
        
        # Ensure high is the maximum
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is the minimum
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect outliers using IQR method on returns.
        
        Returns:
            Dictionary mapping column names to outlier counts
        """
        outliers = {}
        
        # Check price columns using returns
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            q1, q3 = returns.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = (returns < q1 - 3 * iqr) | (returns > q3 + 3 * iqr)
            outliers['close_returns'] = outlier_mask.sum()
        
        # Check volume
        if 'volume' in df.columns:
            volume = df['volume'].dropna()
            q1, q3 = volume.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_mask = volume > q3 + 5 * iqr  # Very high volume might be real
            outliers['volume'] = outlier_mask.sum()
        
        return outliers
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers by clipping extreme values.
        
        Note: We're conservative here since crypto can have extreme moves.
        """
        # Clip extreme returns (> 50% in a single candle is suspicious)
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            extreme_mask = returns.abs() > 0.5  # 50% change
            
            if extreme_mask.any():
                logger.warning(f"Found {extreme_mask.sum()} extreme price moves (>50%)")
                # Don't remove, just flag for now
        
        # Handle zero or negative volume
        if 'volume' in df.columns:
            df.loc[df['volume'] <= 0, 'volume'] = np.nan
            df['volume'] = df['volume'].fillna(method='ffill', limit=1)
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize price and volume data.
        
        Uses rolling z-score normalization to maintain stationarity.
        """
        df = df.copy()
        
        if self.normalize_prices:
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    # Store original values
                    df[f'{col}_raw'] = df[col]
                    
                    # Calculate rolling statistics
                    rolling_mean = df[col].rolling(
                        window=self.normalization_window, min_periods=1
                    ).mean()
                    rolling_std = df[col].rolling(
                        window=self.normalization_window, min_periods=1
                    ).std()
                    
                    # Z-score normalization
                    df[f'{col}_normalized'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        if self.normalize_volume:
            if 'volume' in df.columns:
                # Store original
                df['volume_raw'] = df['volume']
                
                # Log transform + rolling z-score
                log_volume = np.log1p(df['volume'])
                rolling_mean = log_volume.rolling(
                    window=self.normalization_window, min_periods=1
                ).mean()
                rolling_std = log_volume.rolling(
                    window=self.normalization_window, min_periods=1
                ).std()
                
                df['volume_normalized'] = (log_volume - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def _calculate_quality_score(
        self,
        total_rows: int,
        missing_pct: float,
        duplicate_pct: float,
        gap_pct: float,
        invalid_pct: float,
        outlier_pct: float
    ) -> float:
        """Calculate an overall quality score (0-100)."""
        if total_rows < 100:
            return 0.0
        
        # Weights for each factor
        weights = {
            'missing': 0.3,
            'duplicate': 0.15,
            'gap': 0.25,
            'invalid': 0.2,
            'outlier': 0.1
        }
        
        # Calculate penalty for each factor (0-1, where 0 is best)
        penalties = {
            'missing': min(missing_pct / self.max_missing_pct, 1.0),
            'duplicate': min(duplicate_pct * 100, 1.0),
            'gap': min(gap_pct * 10, 1.0),
            'invalid': min(invalid_pct * 100, 1.0),
            'outlier': min(outlier_pct * 50, 1.0)
        }
        
        # Weighted penalty
        total_penalty = sum(
            weights[k] * penalties[k] for k in weights
        )
        
        # Convert to score (0-100)
        return max(0, 100 * (1 - total_penalty))
    
    def get_normalization_params(self, df: pd.DataFrame, col: str) -> Dict:
        """Get normalization parameters for inverse transformation."""
        return {
            'mean': df[f'{col}_raw'].iloc[-self.normalization_window:].mean(),
            'std': df[f'{col}_raw'].iloc[-self.normalization_window:].std()
        }
    
    def denormalize(
        self,
        values: np.ndarray,
        mean: float,
        std: float
    ) -> np.ndarray:
        """Reverse normalization."""
        return values * std + mean


def validate_ohlcv_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data structure and content.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")
    
    # Check for empty data
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    # Check for all NaN columns
    for col in required:
        if col in df.columns and df[col].isnull().all():
            errors.append(f"Column {col} contains only NaN values")
    
    # Check data types
    for col in required:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} must be numeric")
    
    return len(errors) == 0, errors
