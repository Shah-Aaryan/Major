"""
Strategy Compatibility Checker (A3 - P1)

Analyzes market regime and recommends compatible strategies.
Prevents running unsuitable strategies and wasting time on optimization.

Key Functions:
1. Detect market regime (trending vs ranging)
2. Check strategy compatibility
3. Recommend alternative strategies if incompatible
4. Score each strategy for the current market
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from backtesting.backtest_engine import BacktestEngine, BacktestConfig
from strategies.strategy_engine import StrategyEngine, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Detected market regime characteristics."""
    regime_type: str  # "trending_up", "trending_down", "ranging", "high_volatility"
    adx: float  # Average Directional Index
    atr_pct: float  # Average True Range as % of price
    trend_strength: float  # 0-1, higher = stronger trend
    trend_slope_pct: float  # Signed slope per bar as % of price
    recent_return: float  # Signed return over analyzed lookback
    long_return: float  # Signed return over longer horizon
    bearish_bias: bool  # True when long-horizon context is bearish
    volatility: float  # 0-1, higher = more volatile
    range_ratio: float  # (high-low) / close, higher = wider range
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Regime: {self.regime_type} | "
            f"ADX: {self.adx:.1f} | "
            f"Trend Strength: {self.trend_strength:.1%} | "
            f"Recent Return: {self.recent_return:.1%} | "
            f"Long Return: {self.long_return:.1%} | "
            f"Volatility: {self.volatility:.1%}"
        )


@dataclass
class StrategyCompatibility:
    """Compatibility score for a strategy in current market."""
    strategy_name: str
    compatibility_score: float  # 0-1, higher = more compatible
    reasons: List[str]  # Why this score
    recommendation: str  # "highly_recommended", "compatible", "risky", "not_recommended"
    expected_return: str  # "positive", "neutral", "negative"
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.strategy_name}: "
            f"Score={self.compatibility_score:.1%} | "
            f"Rec={self.recommendation} | "
            f"Expected={self.expected_return}"
        )


class MarketRegimeAnalyzer:
    """Analyzes market regime from OHLCV data."""
    
    @staticmethod
    def analyze(df: pd.DataFrame, lookback: int = 50) -> MarketRegime:
        """
        Analyze market regime from recent data.
        
        Args:
            df: DataFrame with OHLCV
            lookback: Number of periods to analyze
            
        Returns:
            MarketRegime with detected characteristics
        """
        if len(df) < lookback:
            lookback = len(df)
        
        recent = df.tail(lookback).copy()
        long_lookback = min(len(df), max(lookback * 8, 200))
        long_recent = df.tail(long_lookback).copy()
        
        # Calculate metrics
        adx = MarketRegimeAnalyzer._calculate_adx(recent)
        atr_pct = MarketRegimeAnalyzer._calculate_atr_pct(recent)
        trend_slope_pct = MarketRegimeAnalyzer._calculate_trend_slope_pct(recent)
        trend_strength = MarketRegimeAnalyzer._calculate_trend_strength(recent)
        recent_return = MarketRegimeAnalyzer._calculate_recent_return(recent)
        long_return = MarketRegimeAnalyzer._calculate_recent_return(long_recent)
        bearish_bias = MarketRegimeAnalyzer._calculate_bearish_bias(df)
        volatility = MarketRegimeAnalyzer._calculate_volatility(recent)
        range_ratio = MarketRegimeAnalyzer._calculate_range_ratio(recent)
        
        # Detect regime type
        regime_type = MarketRegimeAnalyzer._classify_regime(
            adx, atr_pct, trend_strength, trend_slope_pct, recent_return, volatility
        )

        if bearish_bias and regime_type == "ranging":
            regime_type = "trending_down"
        
        return MarketRegime(
            regime_type=regime_type,
            adx=adx,
            atr_pct=atr_pct,
            trend_strength=trend_strength,
            trend_slope_pct=trend_slope_pct,
            recent_return=recent_return,
            long_return=long_return,
            bearish_bias=bearish_bias,
            volatility=volatility,
            range_ratio=range_ratio
        )

    @staticmethod
    def _calculate_bearish_bias(df: pd.DataFrame) -> bool:
        """Detect broad bearish context using MA structure and long return."""
        close = df['close']
        if len(close) < 200:
            return False

        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        last_close = close.iloc[-1]

        if pd.isna(ma50) or pd.isna(ma200) or pd.isna(last_close):
            return False

        long_window = min(len(close), 500)
        long_return = (close.iloc[-1] / close.iloc[-long_window]) - 1.0 if close.iloc[-long_window] != 0 else 0.0

        return (last_close < ma200 and ma50 < ma200) or long_return <= -0.15

    @staticmethod
    def _calculate_trend_slope_pct(df: pd.DataFrame) -> float:
        """Calculate signed linear-regression slope as percentage of average close."""
        close = df['close'].values
        x = np.arange(len(close))
        slope = np.polyfit(x, close, 1)[0]

        avg_close = np.mean(close)
        if avg_close <= 0:
            return 0.0
        return slope / avg_close
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame) -> float:
        """Calculate Average Directional Index."""
        if 'adx' in df.columns:
            return df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 25.0
        
        # Fallback: simple trend strength
        close = df['close'].values
        up_days = sum(1 for i in range(1, len(close)) if close[i] > close[i-1])
        down_days = len(close) - 1 - up_days
        
        trend_strength = abs(up_days - down_days) / len(close)
        adx = trend_strength * 70  # Scale to roughly 0-70
        return max(10, min(70, adx))
    
    @staticmethod
    def _calculate_atr_pct(df: pd.DataFrame) -> float:
        """Calculate Average True Range as percentage."""
        if 'atr' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            if close > 0:
                return atr / close
        
        # Fallback: simple volatility
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns)
        return volatility
    
    @staticmethod
    def _calculate_trend_strength(df: pd.DataFrame) -> float:
        """Calculate trend strength 0-1."""
        slope_pct = MarketRegimeAnalyzer._calculate_trend_slope_pct(df)
        
        # Convert to 0-1 scale (±2% = ±1.0 in this scale)
        trend_strength = abs(slope_pct) / 0.02
        return min(1.0, max(0.0, trend_strength))

    @staticmethod
    def _calculate_recent_return(df: pd.DataFrame) -> float:
        """Calculate signed return over analyzed window."""
        close = df['close'].values
        if len(close) < 2 or close[0] == 0:
            return 0.0
        return (close[-1] / close[0]) - 1.0
    
    @staticmethod
    def _calculate_volatility(df: pd.DataFrame) -> float:
        """Calculate volatility 0-1."""
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns) * 100  # In percentage
        
        # Scale: 1% std = 0.2, 5% std = 1.0
        scaled = volatility / 5.0
        return min(1.0, max(0.0, scaled))
    
    @staticmethod
    def _calculate_range_ratio(df: pd.DataFrame) -> float:
        """Calculate (high - low) / close ratio."""
        high = df['high'].values[-30:] if len(df) >= 30 else df['high'].values
        low = df['low'].values[-30:] if len(df) >= 30 else df['low'].values
        close = df['close'].values[-30:] if len(df) >= 30 else df['close'].values
        
        ranges = (high - low) / close
        return np.mean(ranges)
    
    @staticmethod
    def _classify_regime(
        adx: float,
        atr_pct: float,
        trend_strength: float,
        trend_slope_pct: float,
        recent_return: float,
        volatility: float,
    ) -> str:
        """Classify market regime."""
        if volatility > 0.7 and adx < 30:
            return "high_volatility"

        bearish_drift = (recent_return <= -0.03) or (trend_slope_pct <= -0.0004)
        bullish_drift = (recent_return >= 0.03) or (trend_slope_pct >= 0.0004)

        # Moderate ADX plus sustained directional drift should still count as trend.
        if adx >= 25 and trend_strength >= 0.25:
            if bearish_drift:
                return "trending_down"
            if bullish_drift:
                return "trending_up"

        # Strong directional move over lookback can dominate low-ADX noise.
        if recent_return <= -0.06:
            return "trending_down"
        if recent_return >= 0.06:
            return "trending_up"

        if adx < 20:
            return "ranging"

        if bearish_drift:
            return "trending_down"
        if bullish_drift:
            return "trending_up"

        return "ranging"


class StrategyCompatibilityChecker:
    """Checks if a strategy is compatible with current market regime."""

    # When OHLCV+features are available, prefer an empirical check over heuristics.
    EMPIRICAL_LOOKBACK_BARS = 2000
    
    # Strategy compatibility matrix
    COMPATIBILITY_MATRIX = {
        "rsi_mean_reversion": {
            "trending_up": {"score": 0.2, "reason": "Mean reversion fails in uptrends"},
            "trending_down": {"score": 0.1, "reason": "Mean reversion fails in downtrends"},
            "ranging": {"score": 0.85, "reason": "Mean reversion thrives in ranging markets"},
            "high_volatility": {"score": 0.3, "reason": "Mean reversion whipsawed by volatility"},
        },
        "ema_crossover": {
            "trending_up": {"score": 0.9, "reason": "EMA crossover captures uptrends well"},
            "trending_down": {"score": 0.85, "reason": "EMA crossover captures downtrends well"},
            "ranging": {"score": 0.4, "reason": "EMA crossover generates false signals in ranges"},
            "high_volatility": {"score": 0.5, "reason": "EMA crossover prone to whipsaws"},
        },
        "bollinger_breakout": {
            "trending_up": {"score": 0.8, "reason": "Breakout strategy works on up moves"},
            "trending_down": {"score": 0.75, "reason": "Breakout strategy works on down moves"},
            "ranging": {"score": 0.3, "reason": "False breakouts common in ranges"},
            "high_volatility": {"score": 0.6, "reason": "Breakouts can be legitimate in volatility"},
        },
    }
    
    @staticmethod
    def check_compatibility(
        strategy_name: str,
        market_regime: MarketRegime,
        df: Optional[pd.DataFrame] = None,
        empirical: bool = True,
        allow_shorting: bool = True,
        slippage_pct: float = 0.0005
    ) -> StrategyCompatibility:
        """
        Check if strategy is compatible with market regime.
        
        Args:
            strategy_name: Name of strategy
            market_regime: Detected market regime
            
        Returns:
            StrategyCompatibility with score and reasons
        """
        # If we have data, prefer a quick empirical backtest-based score.
        if empirical and df is not None:
            empirical_result = StrategyCompatibilityChecker._empirical_compatibility(
                strategy_name=strategy_name,
                data=df,
                allow_shorting=allow_shorting,
                slippage_pct=slippage_pct,
            )
            if empirical_result is not None:
                return empirical_result

        # Fallback: heuristic matrix score
        if strategy_name not in StrategyCompatibilityChecker.COMPATIBILITY_MATRIX:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return StrategyCompatibility(
                strategy_name=strategy_name,
                compatibility_score=0.5,
                reasons=["Unknown strategy"],
                recommendation="unknown",
                expected_return="neutral"
            )
        
        strategy_compat = StrategyCompatibilityChecker.COMPATIBILITY_MATRIX[strategy_name]
        regime_data = strategy_compat.get(market_regime.regime_type, {})
        
        score = regime_data.get("score", 0.5)
        reason = regime_data.get("reason", "")

        # Contextual adjustments: base matrix is regime-level, but drift can materially
        # change expected outcome for mean-reversion strategies.
        if strategy_name == "rsi_mean_reversion":
            if market_regime.bearish_bias:
                score = min(score, 0.25)
                reason = (
                    f"{reason}; long-horizon bearish bias detected "
                    f"(long return {market_regime.long_return:.1%})"
                )
            if market_regime.recent_return <= -0.03:
                score = min(score, 0.35)
                reason = (
                    f"{reason}; bearish drift ({market_regime.recent_return:.1%}) "
                    "reduces long mean-reversion edge"
                )
            elif market_regime.recent_return <= -0.015:
                score = min(score, 0.55)
                reason = (
                    f"{reason}; mild bearish drift ({market_regime.recent_return:.1%}) "
                    "makes reversions less reliable"
                )
        
        # Determine recommendation
        if score > 0.75:
            recommendation = "highly_recommended"
            expected_return = "positive"
        elif score > 0.5:
            recommendation = "compatible"
            expected_return = "neutral"
        elif score > 0.25:
            recommendation = "risky"
            expected_return = "negative"
        else:
            recommendation = "not_recommended"
            expected_return = "negative"
        
        reasons = [
            reason,
            f"Market: {market_regime.regime_type}",
            f"ADX: {market_regime.adx:.1f}",
            f"Trend Strength: {market_regime.trend_strength:.1%}",
            f"Trend Slope (per bar): {market_regime.trend_slope_pct:.3%}",
            f"Recent Return: {market_regime.recent_return:.1%}",
            f"Long Return: {market_regime.long_return:.1%}",
            f"Bearish Bias: {market_regime.bearish_bias}",
            f"Volatility: {market_regime.volatility:.1%}",
        ]
        
        return StrategyCompatibility(
            strategy_name=strategy_name,
            compatibility_score=score,
            reasons=reasons,
            recommendation=recommendation,
            expected_return=expected_return
        )

    @staticmethod
    def _strategy_name_to_type(strategy_name: str) -> Optional[StrategyType]:
        """Map a user-facing string strategy name to StrategyType."""
        normalized = (strategy_name or "").strip().lower()
        for st in StrategyType:
            if st.value == normalized:
                return st
        return None

    @staticmethod
    def _empirical_compatibility(
        *,
        strategy_name: str,
        data: pd.DataFrame,
        allow_shorting: bool,
        slippage_pct: float,
    ) -> Optional[StrategyCompatibility]:
        """Empirically score a strategy by running a fast backtest on recent bars.

        Returns None if empirical scoring cannot be computed (e.g. missing columns).
        """
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if any(c not in data.columns for c in required_cols):
                return None

            stype = StrategyCompatibilityChecker._strategy_name_to_type(strategy_name)
            if stype is None:
                return None

            # Use a recent window for a quick-but-grounded score.
            window = min(len(data), StrategyCompatibilityChecker.EMPIRICAL_LOOKBACK_BARS)
            if window < 200:
                return None

            recent = data.tail(window).copy()
            recent = recent.ffill().bfill()
            recent = recent.dropna(subset=required_cols)
            if len(recent) < 200:
                return None

            # Create strategy with the same baseline capital/fees used elsewhere.
            engine = StrategyEngine(initial_capital=100000.0, trading_fee_pct=0.001)
            engine.register_strategy(stype)
            strat = engine.get_strategy(stype.value)
            if strat is None:
                return None

            bt_config = BacktestConfig(
                initial_capital=100000.0,
                commission_pct=0.001,
                slippage_pct=slippage_pct,
                allow_shorting=allow_shorting,
            )
            bt = BacktestEngine(config=bt_config)
            result = bt.run(strat, recent)
            m = result.metrics

            # Build a bounded 0-1 score using return/risk/trade-sufficiency.
            total_return = float(getattr(m, 'total_return', 0.0) or 0.0)
            sharpe = float(getattr(m, 'sharpe_ratio', 0.0) or 0.0)
            max_dd = float(getattr(m, 'max_drawdown', 0.0) or 0.0)  # typically negative
            trades = int(getattr(m, 'total_trades', 0) or 0)
            max_dd_abs = abs(max_dd)

            # Core scoring (tanh keeps things stable across extremes)
            score = 0.50
            score += 0.18 * np.tanh(sharpe / 2.0)
            score += 0.22 * np.tanh(total_return / 0.03)
            score -= 0.30 * np.tanh(max_dd_abs / 0.04)

            # Penalize too-few-trades (low confidence in the estimate)
            if trades < 5:
                confidence = trades / 5.0
                score = 0.50 + (score - 0.50) * confidence

            score = float(np.clip(score, 0.0, 1.0))

            # Recommendation thresholds
            if score > 0.75:
                recommendation = "highly_recommended"
            elif score > 0.55:
                recommendation = "compatible"
            elif score > 0.40:
                recommendation = "risky"
            else:
                recommendation = "not_recommended"

            # Expected return label is based on realized metrics.
            if total_return > 0 and sharpe > 0:
                expected_return = "positive"
            elif total_return > -0.01:
                expected_return = "neutral"
            else:
                expected_return = "negative"

            primary_reason = (
                f"Empirical check (last {len(recent)} bars): "
                f"Sharpe={sharpe:.2f}, Return={total_return:.1%}, "
                f"MaxDD={max_dd_abs:.1%}, Trades={trades}"
            )

            reasons = [
                primary_reason,
                f"Empirical Window Bars: {len(recent)}",
                f"Allow Shorting: {allow_shorting}",
            ]

            return StrategyCompatibility(
                strategy_name=strategy_name,
                compatibility_score=score,
                reasons=reasons,
                recommendation=recommendation,
                expected_return=expected_return,
            )
        except Exception as e:
            logger.debug(f"Empirical compatibility failed for {strategy_name}: {e}")
            return None
    
    @staticmethod
    def rank_all_strategies(
        market_regime: MarketRegime,
        df: Optional[pd.DataFrame],
        available_strategies: List[str]
    ) -> List[StrategyCompatibility]:
        """
        Rank all available strategies for current market.
        
        Args:
            market_regime: Detected market regime
            available_strategies: List of available strategy names
            
        Returns:
            List of StrategyCompatibility sorted by score (highest first)
        """
        results = []
        for strategy in available_strategies:
            compat = StrategyCompatibilityChecker.check_compatibility(strategy, market_regime, df=df)
            results.append(compat)
        
        # Sort by score descending
        return sorted(results, key=lambda x: x.compatibility_score, reverse=True)


def early_warning_check(
    df: pd.DataFrame,
    strategy_name: str,
    available_strategies: Optional[List[str]] = None
) -> Tuple[bool, str, Optional[StrategyCompatibility]]:
    """
    Check if strategy is compatible with market and recommend if not.
    
    Args:
        df: DataFrame with OHLCV
        strategy_name: Strategy to check
        available_strategies: Alternative strategies available
        
    Returns:
        Tuple of (is_compatible, warning_message, best_alternative)
    """
    if available_strategies is None:
        available_strategies = ["rsi_mean_reversion", "ema_crossover", "bollinger_breakout"]
    else:
        # If we need to recommend an *alternative*, never recommend the current strategy.
        available_strategies = [s for s in available_strategies if s != strategy_name]
    
    # Analyze market
    market_regime = MarketRegimeAnalyzer.analyze(df)
    
    # Check compatibility
    compat = StrategyCompatibilityChecker.check_compatibility(strategy_name, market_regime, df=df)
    
    # Prepare message
    is_compatible = compat.recommendation not in ["risky", "not_recommended"]
    
    message = f"\n{'='*70}\n"
    message += f"STRATEGY COMPATIBILITY CHECK\n"
    message += f"{'='*70}\n\n"
    message += f"Market Regime: {market_regime.summary()}\n\n"
    message += f"Current Strategy: {strategy_name}\n"
    message += f"Compatibility: {compat.summary()}\n\n"
    if compat.reasons:
        message += f"Details: {compat.reasons[0]}\n\n"
    
    if not is_compatible:
        message += f"WARNING: {strategy_name} is NOT recommended for {market_regime.regime_type}\n"
        message += "Risk: Empirical/heuristic checks suggest poor performance or instability in this regime.\n\n"
        
        # Recommend alternatives
        ranked = StrategyCompatibilityChecker.rank_all_strategies(market_regime, df, available_strategies)
        if ranked:
            message += "RECOMMENDED STRATEGIES:\n"
            best = ranked[0]
            for i, alt in enumerate(ranked[:3], 1):
                message += f"{i}. {alt.strategy_name}: {alt.recommendation} (Score: {alt.compatibility_score:.1%})\n"
                message += f"   → {alt.reasons[0]}\n"

            # Only propose an alternative if it is actually usable.
            if best.recommendation in {"compatible", "highly_recommended"}:
                message += f"\nSUGGESTION: Use '{best.strategy_name}' instead\n"
                return is_compatible, message, best

            message += "\nSUGGESTION: No strategy is recommended based on the current checks. "
            message += "Try a different timeframe, reduce --sample-rows bias, or run --all-strategies research mode.\n"
            return is_compatible, message, None
    else:
        message += f"OK: {strategy_name} is suitable for current market conditions\n"
    
    message += f"{'='*70}\n"
    return is_compatible, message, None


def strategy_watchdog(
    df: pd.DataFrame,
    strategy_name: str,
    available_strategies: List[str],
    fail_if_incompatible: bool = False
) -> Tuple[bool, str]:
    """
    Watchdog function to prevent running incompatible strategies.
    
    Args:
        df: DataFrame with OHLCV
        strategy_name: Strategy to use
        available_strategies: Available alternatives
        fail_if_incompatible: If True, raise error instead of warning
        
    Returns:
        Tuple of (should_continue, message)
    """
    is_compat, message, alt = early_warning_check(df, strategy_name, available_strategies)
    
    logger.info(message)
    
    if not is_compat and fail_if_incompatible:
        if alt:
            raise ValueError(
                f"Strategy {strategy_name} incompatible with market regime. "
                f"Use {alt.strategy_name} instead (compatibility: {alt.compatibility_score:.1%})"
            )
        else:
            raise ValueError(f"Strategy {strategy_name} incompatible with market regime")
    
    return is_compat, message
