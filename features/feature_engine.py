"""
Main Feature Engineering Engine.

Combines all feature generators into a unified pipeline.
Handles feature computation, caching, and incremental updates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import hashlib
import pickle
from pathlib import Path

from features.price_features import PriceFeatureGenerator
from features.trend_indicators import TrendIndicatorGenerator
from features.momentum_indicators import MomentumIndicatorGenerator
from features.volatility_indicators import VolatilityIndicatorGenerator
from features.volume_indicators import VolumeIndicatorGenerator
from features.regime_features import RegimeFeatureGenerator
from features.indicator_registry import get_indicator_registry, IndicatorSpec

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    
    # Price features
    log_return_periods: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    rolling_return_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 240])
    momentum_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Trend features
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    adx_period: int = 14
    
    # Momentum features
    rsi_periods: List[int] = field(default_factory=lambda: [6, 14, 21])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility features
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Volume features
    volume_ma_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Regime features
    regime_lookback: int = 100
    trend_threshold: float = 25.0
    adx_trend_threshold: float = 25.0  # Alias for trend_threshold


class FeatureEngine:
    """
    Main feature engineering engine.
    
    This class orchestrates all feature generators and provides:
    - Unified feature generation pipeline
    - Feature caching
    - Incremental feature updates
    - Feature selection and filtering
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature engine.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        
        # Initialize generators
        self.price_gen = PriceFeatureGenerator(
            log_return_periods=self.config.log_return_periods,
            rolling_return_windows=self.config.rolling_return_windows,
            momentum_periods=self.config.momentum_periods
        )
        
        self.trend_gen = TrendIndicatorGenerator(
            sma_windows=self.config.sma_windows,
            ema_windows=self.config.ema_windows,
            adx_period=self.config.adx_period
        )
        
        self.momentum_gen = MomentumIndicatorGenerator(
            rsi_periods=self.config.rsi_periods,
            macd_fast=self.config.macd_fast,
            macd_slow=self.config.macd_slow,
            macd_signal=self.config.macd_signal
        )
        
        self.volatility_gen = VolatilityIndicatorGenerator(
            atr_periods=self.config.atr_periods,
            bollinger_window=self.config.bollinger_window,
            bollinger_std=self.config.bollinger_std
        )
        
        self.volume_gen = VolumeIndicatorGenerator(
            volume_ma_windows=self.config.volume_ma_windows
        )
        
        self.regime_gen = RegimeFeatureGenerator(
            regime_lookback=self.config.regime_lookback,
            trend_threshold=self.config.adx_trend_threshold
        )

        # Registry of the full 52-indicator surface (implemented + planned)
        self.indicator_registry: List[IndicatorSpec] = get_indicator_registry()

        # Map indicator names (lowercase) to the column prefixes they emit.
        # This is used to filter feature sets when users request specific indicators.
        self._indicator_column_map = {
            "simple moving average (sma)": ["sma_", "price_sma_"],
            "exponential moving average (ema)": ["ema_", "price_ema_", "ema_20_for_slope", "ema_20_slope"],
            "moving average convergence divergence (macd)": ["macd"],
            "average directional index (adx)": ["adx", "plus_di", "minus_di", "dx", "trend_strength", "trend_direction", "di_crossover"],
            "relative strength index (rsi)": ["rsi_"],
            "stochastic rsi": ["stoch_rsi"],
            "williams %r": ["williams_r"],
            "commodity channel index (cci)": ["cci"],
            "rate of change (roc)": ["roc_"],
            "average true range (atr)": ["true_range", "atr_"],
            "bollinger bands": ["bb_"],
            "keltner channels": ["kc_"],
            "donchian channels": ["dc_"],
            "rolling standard deviation": ["volatility_"],
            "historical volatility (close-to-close)": ["volatility_"],
            "parkinson volatility": ["vol_parkinson"],
            "garman-klass volatility": ["vol_garman_klass"],
            "yang-zhang volatility": ["vol_yang_zhang"],
            "on-balance volume (obv)": ["obv"],
            "volume weighted average price (vwap)": ["vwap"],
            "money flow index (mfi)": ["mfi"],
            "accumulation/distribution line (adl)": ["ad_line", "mfm"],
            "trend lines (linear regression)": ["trend_"] ,
            "ema slope": ["ema_20_slope"],
        }
        
        # Feature cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_dir = Path("./cache/features")

    def list_indicators(self, implemented_only: bool = True) -> List[IndicatorSpec]:
        """Return indicator specifications, optionally filtering to implemented ones."""
        if implemented_only:
            return [spec for spec in self.indicator_registry if spec.implemented]
        return list(self.indicator_registry)
    
    def generate_features(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[List[str]] = None,
        include_indicators: Optional[List[str]] = None,
        drop_na: bool = True,
        use_cache: bool = False,
        cache_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate all features for the input data.
        
        Args:
            df: OHLCV DataFrame (must have open, high, low, close, volume)
            feature_groups: List of feature groups to generate (None = all)
                Options: 'price', 'trend', 'momentum', 'volatility', 'volume', 'regime'
            include_indicators: Optional list of indicator names (registry names) to include.
                If provided, only indicators in this list are emitted (plus OHLCV).
            drop_na: Whether to drop rows with NaN values
            use_cache: Whether to use cached features
            cache_key: Key for caching (auto-generated if None)
            
        Returns:
            DataFrame with all generated features
        """
        # Check cache
        if use_cache and cache_key:
            if cache_key in self._cache:
                logger.info(f"Using cached features for {cache_key}")
                return self._cache[cache_key]
        
        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Normalize indicator selection and derive groups to generate
        selected_specs: Optional[List[IndicatorSpec]] = None
        if include_indicators:
            requested = {name.strip().lower() for name in include_indicators}
            name_to_spec = {spec.name.lower(): spec for spec in self.list_indicators(implemented_only=True)}
            unknown = sorted(requested - set(name_to_spec.keys()))
            if unknown:
                raise ValueError(f"Unknown or unimplemented indicators: {unknown}")
            selected_specs = [name_to_spec[name] for name in requested]

        # Determine which groups to generate
        all_groups = ['price', 'trend', 'momentum', 'volatility', 'volume', 'regime']
        if selected_specs:
            groups_to_generate = sorted({spec.category for spec in selected_specs})
        else:
            groups_to_generate = feature_groups or all_groups

        logger.info(
            f"Generating features for groups: {groups_to_generate}" + (
                f" with indicators: {include_indicators}" if include_indicators else ""
            )
        )
        
        # Start with copy of input
        features = df.copy()
        
        # Generate features by group
        if 'price' in groups_to_generate:
            features = self.price_gen.generate(features)
        
        if 'trend' in groups_to_generate:
            features = self.trend_gen.generate(features)
        
        if 'momentum' in groups_to_generate:
            features = self.momentum_gen.generate(features)
        
        if 'volatility' in groups_to_generate:
            features = self.volatility_gen.generate(features)
        
        if 'volume' in groups_to_generate:
            features = self.volume_gen.generate(features)
        
        if 'regime' in groups_to_generate:
            features = self.regime_gen.generate(features)
        
        # Handle NaN values
        initial_len = len(features)
        if drop_na:
            features = features.dropna()
            dropped = initial_len - len(features)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NaN values")
        
        # If specific indicators were requested, filter down to their columns
        if selected_specs:
            features = self._filter_indicator_columns(features, selected_specs)

        # Cache results
        if use_cache and cache_key:
            self._cache[cache_key] = features

        logger.info(
            f"Generated {len(features.columns)} total features, "
            f"{len(features)} rows"
        )

        return features
    
    def generate_incremental(
        self,
        existing_features: pd.DataFrame,
        new_data: pd.DataFrame,
        lookback: int = 500
    ) -> pd.DataFrame:
        """
        Generate features incrementally for new data.
        
        Used for real-time updates where we only need to compute
        features for new candles.
        
        Args:
            existing_features: Previously computed features
            new_data: New OHLCV data to add
            lookback: Number of rows needed for feature calculation
            
        Returns:
            Updated features DataFrame
        """
        # Combine old data with new
        # We need some lookback from existing data for indicator calculations
        if len(existing_features) < lookback:
            # Not enough history, regenerate everything
            combined_ohlcv = pd.concat([
                existing_features[['open', 'high', 'low', 'close', 'volume']],
                new_data
            ])
            return self.generate_features(combined_ohlcv, drop_na=True)
        
        # Get last N rows of existing data + new data
        lookback_data = existing_features.iloc[-lookback:][
            ['open', 'high', 'low', 'close', 'volume']
        ]
        combined = pd.concat([lookback_data, new_data])
        
        # Generate features for combined data
        new_features = self.generate_features(combined, drop_na=False)
        
        # Keep only the truly new rows
        new_rows = new_features.iloc[lookback:]
        
        # Combine with existing
        result = pd.concat([
            existing_features.iloc[:-lookback] if len(existing_features) > lookback else pd.DataFrame(),
            new_features
        ])
        
        return result
    
    def get_feature_names(self, group: Optional[str] = None) -> List[str]:
        """
        Get list of all feature names.
        
        Args:
            group: Specific group to get names for (None = all)
            
        Returns:
            List of feature names
        """
        names = []
        
        generators = {
            'price': self.price_gen,
            'trend': self.trend_gen,
            'momentum': self.momentum_gen,
            'volatility': self.volatility_gen,
            'volume': self.volume_gen,
            'regime': self.regime_gen
        }
        
        if group:
            if group in generators:
                return generators[group].get_feature_names()
            else:
                raise ValueError(f"Unknown group: {group}")
        
        for gen in generators.values():
            names.extend(gen.get_feature_names())
        
        return names
    
    def get_feature_importance_for_strategy(
        self,
        strategy_type: str
    ) -> Dict[str, List[str]]:
        """
        Get recommended features for a specific strategy type.
        
        Args:
            strategy_type: Type of strategy ('rsi', 'ema_crossover', 'bollinger')
            
        Returns:
            Dictionary with 'primary' and 'secondary' feature lists
        """
        importance_map = {
            'rsi_mean_reversion': {
                'primary': [
                    'rsi_14', 'rsi_14_overbought', 'rsi_14_oversold',
                    'rsi_14_momentum', 'bb_percent_b', 'mr_zscore',
                    'regime_is_trending', 'vol_regime'
                ],
                'secondary': [
                    'stoch_rsi_k', 'mfi', 'price_position',
                    'volatility_20', 'volume_spike'
                ]
            },
            'ema_crossover': {
                'primary': [
                    'ema_9', 'ema_21', 'ema_9_21_cross',
                    'trend_slope', 'adx', 'macd', 'macd_signal'
                ],
                'secondary': [
                    'regime_trend', 'vol_ratio', 'momentum_20',
                    'trend_aligned_up', 'trend_aligned_down'
                ]
            },
            'bollinger_breakout': {
                'primary': [
                    'bb_upper', 'bb_lower', 'bb_width', 'bb_percent_b',
                    'bb_squeeze', 'bb_expansion', 'volatility_20',
                    'dc_breakout_up', 'dc_breakout_down'
                ],
                'secondary': [
                    'volume_spike', 'atr_14', 'regime_is_consolidating',
                    'vol_expanding', 'vol_contracting'
                ]
            }
        }
        
        return importance_map.get(strategy_type, {'primary': [], 'secondary': []})
    
    def select_features(
        self,
        df: pd.DataFrame,
        feature_list: List[str]
    ) -> pd.DataFrame:
        """
        Select specific features from the full feature set.
        
        Args:
            df: DataFrame with all features
            feature_list: List of feature names to select
            
        Returns:
            DataFrame with only selected features
        """
        # Always include OHLCV
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        cols_to_select = base_cols + [
            col for col in feature_list if col in df.columns
        ]
        
        missing = [col for col in feature_list if col not in df.columns]
        if missing:
            logger.warning(f"Missing features: {missing}")
        
        return df[cols_to_select]
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize feature values.
        
        Args:
            df: Features DataFrame
            method: Normalization method ('zscore', 'minmax', 'robust')
            exclude_cols: Columns to exclude from normalization
            
        Returns:
            Normalized DataFrame
        """
        exclude = exclude_cols or ['open', 'high', 'low', 'close', 'volume']
        
        result = df.copy()
        cols_to_normalize = [col for col in df.columns if col not in exclude]
        
        for col in cols_to_normalize:
            if df[col].dtype in ['object', 'category', 'bool']:
                continue
            
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                result[col] = (df[col] - mean) / (std + 1e-10)
            
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                result[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
            
            elif method == 'robust':
                median = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                result[col] = (df[col] - median) / (iqr + 1e-10)
        
        return result

    def _filter_indicator_columns(
        self,
        features: pd.DataFrame,
        selected_specs: List[IndicatorSpec]
    ) -> pd.DataFrame:
        """Keep only the columns tied to the selected indicators (plus OHLCV)."""
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        allowed_cols = set(base_cols)

        for spec in selected_specs:
            prefixes = self._indicator_column_map.get(spec.name.lower())
            if not prefixes:
                logger.warning(f"No column mapping for indicator '{spec.name}', keeping all columns")
                continue
            for col in features.columns:
                if col in base_cols:
                    continue
                if any(col.startswith(prefix) for prefix in prefixes):
                    allowed_cols.add(col)

        filtered = features[[col for col in features.columns if col in allowed_cols]]
        missing = [spec.name for spec in selected_specs if not self._indicator_column_map.get(spec.name.lower())]
        if missing:
            logger.warning(f"Indicators without mappings (columns kept unfiltered for them): {missing}")
        return filtered
    
    def save_cache(self, path: str) -> None:
        """Save feature cache to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        logger.info(f"Saved feature cache to {path}")
    
    def load_cache(self, path: str) -> None:
        """Load feature cache from disk."""
        if Path(path).exists():
            with open(path, 'rb') as f:
                self._cache = pickle.load(f)
            logger.info(f"Loaded feature cache from {path}")
    
    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._cache.clear()
        logger.info("Cleared feature cache")
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for all features.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        summary = df.describe().T
        summary['missing'] = df.isnull().sum()
        summary['missing_pct'] = df.isnull().sum() / len(df) * 100
        summary['unique'] = df.nunique()
        summary['dtype'] = df.dtypes
        
        return summary
