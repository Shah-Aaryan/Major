"""
Custom Strategy - Executes user-defined trading algorithms.

Allows users to define trading logic as a string with conditions like:
  EMA20 < EMA50 AND price < EMA50 AND RSI crosses down from 45-60

The system parses and evaluates these conditions on each bar.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
import logging

from strategies.base_strategy import BaseStrategy, StrategySignal, SignalType, StrategyParameters

logger = logging.getLogger(__name__)


class CustomStrategyParams(StrategyParameters):
    """Parameters for custom strategies (minimal - conditions come from algorithm string)."""
    
    def __init__(self, algorithm: str = ""):
        """
        Initialize custom strategy parameters.
        
        Args:
            algorithm: User-defined algorithm string
        """
        self.algorithm = algorithm
        super().__init__()
    
    def to_dict(self) -> Dict:
        return {'algorithm': self.algorithm}
    
    def from_dict(self, data: Dict):
        self.algorithm = data.get('algorithm', '')


class ConditionEvaluator:
    """Evaluates trading conditions based on indicator values."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize evaluator with feature data.
        
        Args:
            df: DataFrame with OHLCV and indicators
        """
        self.df = df
        self.current_idx = None
    
    def _find_column(self, signal_name: str) -> Optional[str]:
        """
        Smart column resolution: handles case-insensitive, snake_case conversion, and aliases.
        
        Args:
            signal_name: Human-readable indicator name (e.g., 'EMA20', 'lower Bollinger Band')
            
        Returns:
            Actual column name or None if not found
        """
        signal_lower = signal_name.lower().strip()
        available_cols = set(self.df.columns)
        cols_lower = {col.lower(): col for col in available_cols}
        
        # 1. Direct exact match (case-insensitive)
        if signal_lower in cols_lower:
            return cols_lower[signal_lower]
        
        # 2. Common alias mappings
        aliases = {
            'rsi': 'rsi_14',  # Most common
            'lower bollinger band': 'bb_lower',
            'upper bollinger band': 'bb_upper',
            'middle bollinger band': 'bb_middle',
            'bb_lower': 'bb_lower',
            'price': 'close',
        }
        if signal_lower in aliases:
            alias_col = aliases[signal_lower]
            if alias_col in available_cols:
                return alias_col
            # Try case-insensitive match of alias
            if alias_col.lower() in cols_lower:
                return cols_lower[alias_col.lower()]
        
        # 3. Handle EMA/SMA patterns (e.g., "EMA20" -> "ema_20")
        match = re.match(r'(ema|sma)(\d+)', signal_lower)
        if match:
            indicator = match.group(1)  # 'ema' or 'sma'
            period = match.group(2)      # '20', '50', etc.
            
            # Try exact snake_case
            candidate = f"{indicator}_{period}"
            if candidate in available_cols:
                return candidate
            if candidate.lower() in cols_lower:
                return cols_lower[candidate.lower()]
            
            # Try finding any matching column with this pattern
            pattern = f"^{indicator}_{period}(_|$)"
            for col in available_cols:
                if re.match(pattern, col, re.IGNORECASE):
                    return col
        
        # 4. Handle RSI patterns (e.g., "RSI" -> "rsi_14")
        if signal_lower.startswith('rsi'):
            # Try rsi_14 (most common), then rsi_6, rsi_21
            for period in ['14', '6', '21']:
                candidate = f"rsi_{period}"
                if candidate in available_cols:
                    return candidate
                if candidate.lower() in cols_lower:
                    return cols_lower[candidate.lower()]
        
        # 5. Handle volume patterns (e.g., "volume EMA20" -> "volume_ma_20")
        if 'volume' in signal_lower and ('ema' in signal_lower or 'ma' in signal_lower):
            # Try "volume_ma_20", "volume_ema_20"
            match = re.search(r'volume\s+(ema|ma)(\d+)', signal_lower)
            if match:
                freq_type = match.group(1)  # 'ema' or 'ma'
                period = match.group(2)      # '20', '50', etc.
                
                for variant in [f"volume_{freq_type}_{period}", f"volume_ma_{period}"]:
                    if variant in available_cols:
                        return variant
                    if variant.lower() in cols_lower:
                        return cols_lower[variant.lower()]
        
        # 6. Last resort: fuzzy match (find column containing the core pattern)
        core_name = re.sub(r'\d+', '', signal_lower).strip()
        if core_name:
            for col in available_cols:
                col_core = re.sub(r'\d+', '', col.lower()).strip()
                if col_core == core_name and col_core:  # Avoid empty strings
                    return col
        
        return None
    
    def get_value(self, signal_name: str, lookback: int = 0) -> Optional[float]:
        """
        Get indicator/price value with optional lookback.
        
        Args:
            signal_name: Name of signal (e.g., 'EMA20', 'RSI14', 'price', 'volume')
            lookback: How many bars back (0 = current)
            
        Returns:
            Value or None if not found
        """
        if self.current_idx is None:
            return None
        
        idx = self.current_idx - lookback
        if idx < 0 or idx >= len(self.df):
            return None
        
        signal_lower = signal_name.lower().strip()
        
        # Price columns (direct lookup)
        if signal_lower == 'price' or signal_lower == 'close':
            return self.df['close'].iloc[idx]
        elif signal_lower == 'open':
            return self.df['open'].iloc[idx]
        elif signal_lower == 'high':
            return self.df['high'].iloc[idx]
        elif signal_lower == 'low':
            return self.df['low'].iloc[idx]
        elif signal_lower == 'volume':
            return self.df['volume'].iloc[idx]
        
        # Smart column resolution
        col = self._find_column(signal_name)
        if col:
            try:
                val = self.df[col].iloc[idx]
                return float(val) if pd.notna(val) else None
            except (KeyError, IndexError, ValueError):
                return None
        
        # Not found
        logger.warning(f"Indicator/column '{signal_name}' not found in data (tried smart matching)")
        return None
    
    def evaluate_comparison(self, condition: str) -> bool:
        """
        Evaluate a single comparison like 'EMA20 < EMA50' or 'price > 100'.
        
        Args:
            condition: Comparison string
            
        Returns:
            Boolean result
        """
        condition = condition.strip()
        
        # Try different operators
        operators = ['<=', '>=', '<', '>', '==', '!=']
        for op in operators:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) != 2:
                    continue
                
                left_str = parts[0].strip()
                right_str = parts[1].strip()
                
                try:
                    left = self.parse_expression(left_str)
                    right = self.parse_expression(right_str)
                except Exception as e:
                    logger.warning(f"Error parsing expression in '{condition}': {e}")
                    return False
                
                if left is None or right is None:
                    return False
                
                if op == '<':
                    return left < right
                elif op == '>':
                    return left > right
                elif op == '<=':
                    return left <= right
                elif op == '>=':
                    return left >= right
                elif op == '==':
                    return abs(left - right) < 1e-6
                elif op == '!=':
                    return abs(left - right) >= 1e-6
        
        logger.warning(f"Could not parse condition: {condition}")
        return False
    
    def parse_expression(self, expr: str) -> Optional[float]:
        """
        Parse an expression that could be a number, indicator, or calculation.
        
        Args:
            expr: Expression like 'EMA20', '100', 'EMA20 - EMA50', etc.
            
        Returns:
            Numeric result or None
        """
        expr = expr.strip()
        
        # Try to convert to float directly
        try:
            return float(expr)
        except ValueError:
            pass
        
        # Check for arithmetic operations
        if any(op in expr for op in ['+', '-', '*', '/', '(', ')']):
            # Simple arithmetic (be careful here)
            # Replace indicator names with their values
            expr_eval = expr
            
            # Find all words (potential indicators)
            words = re.findall(r'\b[A-Za-z_]\w*\b', expr)
            for word in words:
                val = self.get_value(word)
                if val is not None:
                    expr_eval = expr_eval.replace(word, str(val))
            
            try:
                # Safely evaluate if all variables were replaced
                if not any(c.isalpha() for c in expr_eval):
                    result = eval(expr_eval)
                    return float(result)
            except Exception as e:
                logger.warning(f"Could not evaluate: {expr}: {e}")
                return None
        
        # Single indicator
        return self.get_value(expr)
    
    def evaluate_crosses(self, condition: str) -> bool:
        """
        Evaluate crossing conditions like 'RSI crosses down from 45-60'.
        
        Args:
            condition: Condition string with 'crosses' keyword
            
        Returns:
            Boolean result
        """
        if 'crosses down' in condition.lower():
            # e.g., "RSI crosses down from 45-60"
            match = re.search(r'(\w+)\s+crosses\s+down\s+from\s+([\d.]+)\s*-\s*([\d.]+)', condition, re.IGNORECASE)
            if match:
                indicator = match.group(1)
                lower_bound = float(match.group(2))
                upper_bound = float(match.group(3))
                
                current = self.get_value(indicator)
                previous = self.get_value(indicator, lookback=1)
                
                if current is None or previous is None:
                    return False
                
                # Was above range, now below
                return previous >= upper_bound and current < lower_bound
        
        elif 'crosses up' in condition.lower():
            # e.g., "RSI crosses up from 40-50"
            match = re.search(r'(\w+)\s+crosses\s+up\s+from\s+([\d.]+)\s*-\s*([\d.]+)', condition, re.IGNORECASE)
            if match:
                indicator = match.group(1)
                lower_bound = float(match.group(2))
                upper_bound = float(match.group(3))
                
                current = self.get_value(indicator)
                previous = self.get_value(indicator, lookback=1)
                
                if current is None or previous is None:
                    return False
                
                # Was below range, now above
                return previous <= lower_bound and current >= upper_bound
        
        return False
    
    def evaluate_breaks(self, condition: str) -> bool:
        """
        Evaluate break conditions like 'price breaks below lower Bollinger Band'.
        
        Args:
            condition: Condition string with 'breaks' keyword
            
        Returns:
            Boolean result
        """
        if 'breaks below' in condition.lower():
            match = re.search(r'(\w+)\s+breaks\s+below\s+(.+)', condition, re.IGNORECASE)
            if match:
                indicator1 = match.group(1)
                indicator2 = match.group(2)
                
                val1_current = self.get_value(indicator1)
                val1_prev = self.get_value(indicator1, lookback=1)
                val2 = self.get_value(indicator2)
                
                if any(v is None for v in [val1_current, val1_prev, val2]):
                    return False
                
                return val1_prev >= val2 and val1_current < val2
        
        elif 'breaks above' in condition.lower():
            match = re.search(r'(\w+)\s+breaks\s+above\s+(.+)', condition, re.IGNORECASE)
            if match:
                indicator1 = match.group(1)
                indicator2 = match.group(2)
                
                val1_current = self.get_value(indicator1)
                val1_prev = self.get_value(indicator1, lookback=1)
                val2 = self.get_value(indicator2)
                
                if any(v is None for v in [val1_current, val1_prev, val2]):
                    return False
                
                return val1_prev <= val2 and val1_current > val2
        
        return False
    
    def evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a single condition (could be comparison, cross, or break).
        
        Args:
            condition: Condition string
            
        Returns:
            Boolean result
        """
        condition = condition.strip()
        
        # Check for special keywords first
        if 'crosses' in condition.lower():
            return self.evaluate_crosses(condition)
        elif 'breaks' in condition.lower():
            return self.evaluate_breaks(condition)
        else:
            return self.evaluate_comparison(condition)


class CustomStrategy(BaseStrategy):
    """
    Strategy that executes user-defined algorithm.
    
    Algorithm is specified as a string with conditions joined by AND/OR.
    Example:
        EMA20 < EMA50 AND price < EMA50 AND RSI crosses down from 45-60
    """
    
    def __init__(self, algorithm: str = "", initial_capital: float = 100000.0, trading_fee_pct: float = 0.001):
        """
        Initialize custom strategy.
        
        Args:
            algorithm: User-defined algorithm string
            initial_capital: Starting capital
            trading_fee_pct: Trading fee percentage
        """
        super().__init__(
            name="Custom_Algorithm",
            parameters=CustomStrategyParams(algorithm=algorithm),
            initial_capital=initial_capital,
            trading_fee_pct=trading_fee_pct
        )
        self.algorithm = algorithm
        
        # Parse algorithm for buy/sell conditions
        self.buy_conditions = []
        self.sell_conditions = []
        self._parse_algorithm()
    
    def _parse_algorithm(self):
        """Parse algorithm string into buy and sell conditions."""
        if not self.algorithm:
            logger.warning("No algorithm specified for custom strategy")
            return
        
        # Simple split: assume first part is buy conditions, after "SELL:" is sell conditions
        parts = self.algorithm.split('SELL:', 1)
        
        if len(parts) == 2:
            buy_algo = parts[0].strip()
            sell_algo = parts[1].strip()
        else:
            # If no explicit SELL:, use same conditions for reverse signals
            buy_algo = self.algorithm
            sell_algo = None
        
        # Split by AND/OR (for now, treat all as AND)
        self.buy_conditions = [c.strip() for c in re.split(r'\bAND\b|\bOR\b', buy_algo, flags=re.IGNORECASE) if c.strip()]
        self.sell_conditions = [c.strip() for c in re.split(r'\bAND\b|\bOR\b', sell_algo, flags=re.IGNORECASE) if c.strip()] if sell_algo else []
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters and re-parse algorithm if changed."""
        super().update_parameters(params)
        if 'algorithm' in params:
            self.algorithm = str(params['algorithm'])
            self._parse_algorithm()
    
    def _ensure_indicators(self, df: pd.DataFrame) -> None:
        """
        Ensure required indicators exist in dataframe.
        
        For custom strategies, we assume all required indicators are already
        generated during feature engineering. This method is a no-op.
        """
        pass
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int) -> StrategySignal:
        """
        Generate trading signal by evaluating algorithm conditions.
        
        Args:
            df: Feature-augmented OHLCV DataFrame
            current_idx: Current bar index
            
        Returns:
            StrategySignal with BUY, SELL, or HOLD
        """
        if not self.buy_conditions:
            return StrategySignal(
                timestamp=pd.Timestamp(df.index[current_idx]),
                signal_type=SignalType.HOLD,
                price=df['close'].iloc[current_idx],
                reason="No algorithm conditions"
            )
        
        try:
            evaluator = ConditionEvaluator(df)
            evaluator.current_idx = current_idx
            
            # Get current price for signal
            current_price = df['close'].iloc[current_idx]
            current_timestamp = pd.Timestamp(df.index[current_idx])
            
            # Evaluate buy conditions (all must be true = AND logic)
            buy_signal = all(evaluator.evaluate_condition(cond) for cond in self.buy_conditions)
            
            if buy_signal:
                return StrategySignal(
                    timestamp=current_timestamp,
                    signal_type=SignalType.LONG,
                    price=current_price,
                    reason=f"All buy conditions met: {self.algorithm}"
                )
            
            # Evaluate sell conditions if specified
            if self.sell_conditions:
                sell_signal = all(evaluator.evaluate_condition(cond) for cond in self.sell_conditions)
                if sell_signal:
                    return StrategySignal(
                        timestamp=current_timestamp,
                        signal_type=SignalType.SHORT,
                        price=current_price,
                        reason=f"All sell conditions met: {self.algorithm}"
                    )
            
            return StrategySignal(
                timestamp=current_timestamp,
                signal_type=SignalType.HOLD,
                price=current_price,
                reason="Conditions not met"
            )
        except Exception as e:
            logger.error(f"Error evaluating custom algorithm at index {current_idx}: {e}")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"Algorithm: {self.algorithm}")
            # Return HOLD on error with proper timestamp and price
            return StrategySignal(
                timestamp=pd.Timestamp(df.index[current_idx]),
                signal_type=SignalType.HOLD,
                price=df['close'].iloc[current_idx],
                reason=f"Algorithm evaluation error: {str(e)}"
            )
    
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds (empty for custom strategy)."""
        return {}
    
    def get_strategy_specific_params(self) -> Dict[str, Any]:
        """Get custom strategy parameters (algorithm string)."""
        return {
            'algorithm': self.algorithm
        }
    
    def set_strategy_specific_params(self, params: Dict[str, Any]) -> None:
        """Set custom strategy parameters."""
        if 'algorithm' in params:
            self.algorithm = str(params['algorithm'])
            self._parse_algorithm()
