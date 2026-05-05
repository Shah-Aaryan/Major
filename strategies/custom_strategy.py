"""
Custom Strategy - Executes user-defined trading algorithms.

Allows users to define trading logic as a string with conditions like:
  EMA20 < EMA50 AND price < EMA50 AND RSI crosses down from 45-60

The system parses and evaluates these conditions on each bar.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
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
        
        # Parse signal name
        signal_lower = signal_name.lower().strip()
        
        # Price columns
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
        
        # Try to find column directly
        cols_lower = {col.lower(): col for col in self.df.columns}
        if signal_lower in cols_lower:
            col = cols_lower[signal_lower]
            return self.df[col].iloc[idx]
        
        # Not found
        logger.warning(f"Indicator/column '{signal_name}' not found in data")
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
                
                left = self.parse_expression(left_str)
                right = self.parse_expression(right_str)
                
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
    
    def __init__(self, algorithm: str = ""):
        """
        Initialize custom strategy.
        
        Args:
            algorithm: User-defined algorithm string
        """
        self.algorithm = algorithm
        self.parameters = CustomStrategyParams(algorithm=algorithm)
        
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
            return StrategySignal(SignalType.HOLD, reason="No algorithm conditions")
        
        evaluator = ConditionEvaluator(df)
        evaluator.current_idx = current_idx
        
        # Evaluate buy conditions (all must be true = AND logic)
        buy_signal = all(evaluator.evaluate_condition(cond) for cond in self.buy_conditions)
        
        if buy_signal:
            return StrategySignal(
                SignalType.LONG,
                reason=f"All buy conditions met: {self.algorithm}"
            )
        
        # Evaluate sell conditions if specified
        if self.sell_conditions:
            sell_signal = all(evaluator.evaluate_condition(cond) for cond in self.sell_conditions)
            if sell_signal:
                return StrategySignal(
                    SignalType.SHORT,
                    reason=f"All sell conditions met: {self.algorithm}"
                )
        
        return StrategySignal(SignalType.HOLD, reason="Conditions not met")
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter bounds (empty for custom strategy)."""
        return {}
