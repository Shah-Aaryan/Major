"""
Configuration Settings for ML Trading Research Project.

This module contains all configurable parameters for the research pipeline.
Settings are organized by component for easy modification and experimentation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from pathlib import Path


class TimeFrame(Enum):
    """Supported timeframes for analysis."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class StrategyType(Enum):
    """Available strategy types."""
    RSI_MEAN_REVERSION = "rsi_mean_reversion"
    EMA_CROSSOVER = "ema_crossover"
    BOLLINGER_BREAKOUT = "bollinger_breakout"


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    
    # Supported symbols
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"
    ])
    
    # Default timeframe
    default_timeframe: TimeFrame = TimeFrame.MINUTE_1
    
    # Resampling targets
    resample_timeframes: List[TimeFrame] = field(default_factory=lambda: [
        TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15
    ])
    
    # Data quality settings
    max_missing_pct: float = 0.05  # Maximum 5% missing values allowed
    forward_fill_limit: int = 5  # Max consecutive NaNs to forward fill
    
    # Normalization settings
    normalize_prices: bool = True
    normalize_volume: bool = True
    normalization_window: int = 1440  # 1 day of 1-minute candles
    
    # Date range (None for full range)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Price & Returns features
    log_return_periods: List[int] = field(default_factory=lambda: [1, 5, 15, 60])
    rolling_return_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 240])
    momentum_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # SMA settings
    sma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    
    # EMA settings
    ema_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    ema_slope_period: int = 5
    
    # RSI settings
    rsi_periods: List[int] = field(default_factory=lambda: [6, 14, 21])
    stoch_rsi_period: int = 14
    stoch_rsi_smooth: int = 3
    
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatility settings
    atr_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    rolling_vol_windows: List[int] = field(default_factory=lambda: [10, 20, 60])
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Volume settings
    volume_ma_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    volume_spike_threshold: float = 2.0  # 2x average volume
    
    # ADX settings
    adx_period: int = 14
    adx_trend_threshold: float = 25.0  # Above this = trending
    
    # Regime detection settings
    regime_lookback: int = 100
    volatility_regime_percentile: float = 0.7  # 70th percentile = high vol


@dataclass
class StrategyParameterBounds:
    """Defines safe bounds for strategy parameters (for ML optimization)."""
    
    # RSI bounds
    rsi_lookback: Tuple[int, int] = (5, 30)
    rsi_buy_threshold: Tuple[int, int] = (20, 40)
    rsi_sell_threshold: Tuple[int, int] = (60, 80)
    
    # EMA bounds
    ema_fast_period: Tuple[int, int] = (3, 20)
    ema_slow_period: Tuple[int, int] = (15, 100)
    
    # Bollinger bounds
    bollinger_window: Tuple[int, int] = (10, 50)
    bollinger_deviation: Tuple[float, float] = (1.5, 3.0)
    
    # Execution bounds
    entry_confirmation: Tuple[int, int] = (1, 5)
    exit_confirmation: Tuple[int, int] = (1, 5)
    cooldown_period: Tuple[int, int] = (1, 30)
    
    # Risk management bounds
    stop_loss_pct: Tuple[float, float] = (0.5, 5.0)
    take_profit_pct: Tuple[float, float] = (1.0, 10.0)
    trailing_stop_pct: Tuple[float, float] = (0.5, 3.0)
    max_loss_per_trade_pct: Tuple[float, float] = (0.5, 3.0)
    max_trades_per_day: Tuple[int, int] = (1, 20)
    position_size_pct: Tuple[float, float] = (1.0, 20.0)
    risk_per_trade_pct: Tuple[float, float] = (0.5, 5.0)
    
    # Time bounds
    max_holding_time: Tuple[int, int] = (5, 1440)  # 5 min to 1 day


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    
    # Strategy types to test
    active_strategies: List[StrategyType] = field(default_factory=lambda: [
        StrategyType.RSI_MEAN_REVERSION,
        StrategyType.EMA_CROSSOVER,
        StrategyType.BOLLINGER_BREAKOUT
    ])
    
    # Default human parameters (baseline)
    default_params: Dict[str, Any] = field(default_factory=lambda: {
        # RSI Mean Reversion defaults
        "rsi_lookback": 14,
        "rsi_buy_threshold": 30,
        "rsi_sell_threshold": 70,
        
        # EMA Crossover defaults
        "ema_fast_period": 9,
        "ema_slow_period": 21,
        
        # Bollinger defaults
        "bollinger_window": 20,
        "bollinger_deviation": 2.0,
        
        # Execution defaults
        "entry_confirmation": 1,
        "exit_confirmation": 1,
        "cooldown_period": 5,
        
        # Risk management defaults
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "trailing_stop_pct": 1.5,
        "max_loss_per_trade_pct": 2.0,
        "max_trades_per_day": 10,
        "position_size_pct": 10.0,
        "risk_per_trade_pct": 2.0,
        
        # Time defaults
        "trading_start_hour": 0,
        "trading_end_hour": 24,
        "max_holding_time": 240,  # 4 hours in minutes
    })
    
    # Parameter bounds for ML optimization
    parameter_bounds: StrategyParameterBounds = field(default_factory=StrategyParameterBounds)
    
    # Initial capital for backtesting
    initial_capital: float = 100000.0
    
    # Trading fees (0.1% = 0.001)
    trading_fee_pct: float = 0.001


@dataclass
class OptimizationConfig:
    """Configuration for ML parameter optimization."""
    
    # Optimization method
    methods: List[str] = field(default_factory=lambda: [
        "bayesian", "random_search", "evolutionary"
    ])

    # Preferred optimizer (overrides methods order when set)
    preferred_method: Optional[str] = None
    
    # Bayesian optimization settings
    bayesian_n_calls: int = 100
    bayesian_n_initial_points: int = 20
    bayesian_acq_func: str = "EI"  # Expected Improvement
    
    # Random search settings
    random_search_n_iter: int = 100
    random_search_strategy: str = "uniform"  # uniform | latin_hypercube | sobol
    grid_resolution: int = 5
    
    # Evolutionary strategy settings
    es_population_size: int = 50
    es_generations: int = 30
    es_mutation_rate: float = 0.2
    es_crossover_rate: float = 0.7

    # Differential evolution settings
    de_population_size: int = 15
    de_mutation: Tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7

    # Simulated annealing settings
    sa_maxiter: int = 200
    sa_initial_temp: float = 5230.0
    sa_restart_temp_ratio: float = 5e-4
    
    # Optimization objectives (weights)
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.3,  # Minimization (will be negated)
        "total_return": 0.2,
        "stability": 0.1
    })
    
    # Overfitting penalty
    overfitting_penalty_weight: float = 0.2
    
    # Parameter change constraints
    max_param_change_pct: float = 0.2  # Max 20% change per adjustment
    min_adjustment_interval: int = 60  # Minimum 60 candles between adjustments
    
    # ML confidence thresholds
    min_confidence_to_adjust: float = 0.6
    
    # Validation settings
    n_validation_folds: int = 5


@dataclass
class BacktestConfig:
    """Configuration for backtesting framework."""
    
    # Data splits
    train_pct: float = 0.6
    validation_pct: float = 0.2
    test_pct: float = 0.2
    
    # Walk-forward settings
    walk_forward_windows: int = 10
    walk_forward_train_size: int = 10000  # Number of candles
    walk_forward_test_size: int = 2000
    
    # Repeated backtest settings
    n_repeated_backtests: int = 30
    bootstrap_sample_pct: float = 0.8
    
    # Performance thresholds for analysis
    min_sharpe_ratio: float = 0.5
    max_acceptable_drawdown: float = 0.25  # 25%
    min_win_rate: float = 0.4
    
    # Slippage simulation
    slippage_pct: float = 0.0005  # 0.05%
    
    # Market impact (for larger positions)
    market_impact_factor: float = 0.0001


@dataclass
class RealTimeConfig:
    """Configuration for real-time data and paper trading."""
    
    # Binance WebSocket settings
    ws_reconnect_delay: int = 5  # seconds
    ws_ping_interval: int = 30  # seconds
    
    # Feature update settings
    feature_update_interval: int = 1  # Update every candle
    
    # Parameter adjustment settings
    real_time_adjustment_interval: int = 60  # Every 60 candles
    
    # Paper trading settings
    paper_trading_capital: float = 100000.0
    
    # Buffer settings
    candle_buffer_size: int = 1000  # Keep last 1000 candles in memory
    
    # API settings (loaded from environment)
    @property
    def api_key(self) -> Optional[str]:
        return os.getenv("BINANCE_API_KEY")
    
    @property
    def api_secret(self) -> Optional[str]:
        return os.getenv("BINANCE_API_SECRET")


@dataclass
class AuditConfig:
    """Configuration for audit logging and transparency."""
    
    # Output paths
    audit_log_dir: str = "./audit_logs"
    json_output_dir: str = "./output/json"
    
    # Logging detail level
    log_level: str = "INFO"
    log_to_console: bool = False
    
    # What to log
    log_parameter_changes: bool = True
    log_trades: bool = True
    log_market_conditions: bool = True
    log_ml_decisions: bool = True
    
    # JSON output settings
    output_json_pretty: bool = True
    include_timestamps: bool = True
    
    # Blockchain-ready format
    blockchain_audit_format: bool = True


@dataclass
class ResearchConfig:
    """Master configuration combining all settings."""
    
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    # Research experiment settings
    random_seed: int = 42
    n_parallel_jobs: int = -1  # Use all cores
    verbose: bool = True


def get_config() -> ResearchConfig:
    """Get the default research configuration."""
    return ResearchConfig()


def save_config(config: ResearchConfig, path: str) -> None:
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict
    
    # Convert enums to strings for JSON serialization
    def convert_enums(obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_enums(item) for item in obj]
        return obj
    
    config_dict = convert_enums(asdict(config))
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from JSON file."""
    import json
    
    with open(path, 'r') as f:
        return json.load(f)
