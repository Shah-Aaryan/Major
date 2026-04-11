"""
Strategy Text Parser (A1 - CRITICAL)

Converts user-entered strategy text into a constrained parameter space.
Implements a lightweight DSL to parse trading rules.

Example Input:
    "Buy when RSI > 50, Sell when RSI <= 50"

Output:
    StrategySpec with indicators_used, tunable parameters, and fixed logic.

RULES:
- User can ONLY use indicators that already exist
- ML can ONLY adjust parameters explicitly mentioned by user
- Strategy LOGIC remains unchanged
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAMMAR DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Strategy Grammar (BNF-like):
#
# <strategy>      ::= <rule> ("," <rule>)*
# <rule>          ::= <action> "when" <condition>
# <action>        ::= "buy" | "sell" | "exit" | "hold"
# <condition>     ::= <comparison> (("and" | "or") <comparison>)*
# <comparison>    ::= <indicator> <operator> <number>
#                   | <indicator> "crosses" ("above" | "below") <indicator_or_num>
# <indicator>     ::= <indicator_name> ["(" <number> ")"]
# <indicator_name>::= "RSI" | "MACD" | "EMA" | "SMA" | "ATR" | "ADX" | "BB" ...
# <operator>      ::= ">" | ">=" | "<" | "<=" | "==" | "!="
# <number>        ::= integer | float
#
# ═══════════════════════════════════════════════════════════════════════════════


class ActionType(Enum):
    """Trading action types."""
    BUY = "buy"
    SELL = "sell"
    EXIT = "exit"
    HOLD = "hold"


class ComparisonOp(Enum):
    """Comparison operators."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


# Supported indicators and their default parameter bounds
SUPPORTED_INDICATORS = {
    "rsi": {
        "params": ["period"],
        "default_period": 14,
        "period_bounds": (5, 50),
        "value_bounds": (0, 100),
        "feature_pattern": "rsi_{period}"
    },
    "macd": {
        "params": ["fast", "slow", "signal"],
        "default_fast": 12,
        "default_slow": 26,
        "default_signal": 9,
        "feature_pattern": "macd_line"
    },
    "macd_signal": {
        "params": [],
        "feature_pattern": "macd_signal"
    },
    "macd_histogram": {
        "params": [],
        "feature_pattern": "macd_histogram"
    },
    "ema": {
        "params": ["period"],
        "default_period": 20,
        "period_bounds": (5, 200),
        "feature_pattern": "ema_{period}"
    },
    "sma": {
        "params": ["period"],
        "default_period": 20,
        "period_bounds": (5, 200),
        "feature_pattern": "sma_{period}"
    },
    "atr": {
        "params": ["period"],
        "default_period": 14,
        "period_bounds": (5, 50),
        "feature_pattern": "atr_{period}"
    },
    "adx": {
        "params": ["period"],
        "default_period": 14,
        "period_bounds": (5, 50),
        "value_bounds": (0, 100),
        "feature_pattern": "adx"
    },
    "bb_upper": {
        "params": ["period", "std"],
        "default_period": 20,
        "default_std": 2.0,
        "feature_pattern": "bb_upper"
    },
    "bb_lower": {
        "params": ["period", "std"],
        "default_period": 20,
        "default_std": 2.0,
        "feature_pattern": "bb_lower"
    },
    "bb_middle": {
        "params": ["period"],
        "default_period": 20,
        "feature_pattern": "bb_middle"
    },
    "bb_percent": {
        "params": [],
        "value_bounds": (0, 1),
        "feature_pattern": "bb_percent_b"
    },
    "stoch_k": {
        "params": ["period"],
        "default_period": 14,
        "value_bounds": (0, 100),
        "feature_pattern": "stoch_k"
    },
    "stoch_d": {
        "params": ["period"],
        "default_period": 14,
        "value_bounds": (0, 100),
        "feature_pattern": "stoch_d"
    },
    "volume_sma": {
        "params": ["period"],
        "default_period": 20,
        "feature_pattern": "volume_sma_{period}"
    },
    "stoch_rsi": {
        "params": ["period"],
        "default_period": 14,
        "feature_pattern": "stoch_rsi"
    },
    "williams_r": {
        "params": ["period"],
        "default_period": 14,
        "feature_pattern": "williams_r"
    },
    "cci": {
        "params": ["period"],
        "default_period": 20,
        "feature_pattern": "cci"
    },
    "roc": {
        "params": ["period"],
        "default_period": 12,
        "feature_pattern": "roc_{period}"
    },
    "obv": {
        "params": [],
        "feature_pattern": "obv"
    },
    "vwap": {
        "params": [],
        "feature_pattern": "vwap"
    },
    "mfi": {
        "params": ["period"],
        "default_period": 14,
        "feature_pattern": "mfi"
    },
    "price": {
        "params": [],
        "feature_pattern": "close"
    },
    "close": {
        "params": [],
        "feature_pattern": "close"
    }
}



@dataclass
class IndicatorRef:
    """Reference to an indicator with optional parameters."""
    name: str
    period: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_feature_name(self) -> str:
        """Get the DataFrame column name for this indicator."""
        spec = SUPPORTED_INDICATORS.get(self.name.lower(), {})
        pattern = spec.get("feature_pattern", self.name)
        if self.period and "{period}" in pattern:
            return pattern.format(period=self.period)
        return pattern


@dataclass
class Condition:
    """A single condition in a trading rule."""
    indicator: IndicatorRef
    operator: ComparisonOp
    value: Any  # Can be a number or another IndicatorRef
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator": self.indicator.name,
            "indicator_period": self.indicator.period,
            "operator": self.operator.value,
            "value": self.value if not isinstance(self.value, IndicatorRef) else f"{self.value.name}({self.value.period})"
        }


@dataclass
class Rule:
    """A trading rule with action and conditions."""
    action: ActionType
    conditions: List[Condition]
    logic_operator: str = "and"  # "and" or "or" between conditions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "logic_operator": self.logic_operator
        }


@dataclass
class TunableParameter:
    """A parameter that ML can adjust."""
    name: str
    value: Any
    tunable: bool
    bounds: Tuple[float, float]
    param_type: str = "float"  # "float", "int"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "tunable": self.tunable,
            "bounds": list(self.bounds),
            "type": self.param_type,
            "description": self.description
        }


@dataclass
class StrategySpec:
    """
    Structured specification of a parsed strategy.
    
    This is the output of the parser - contains everything needed
    to run the strategy and optimize its parameters.
    """
    raw_text: str
    indicators_used: List[str]
    parameters: Dict[str, TunableParameter]
    rules: List[Rule]
    logic: str = "FIXED"  # Always FIXED - ML cannot change logic
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "indicators_used": self.indicators_used,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "rules": [r.to_dict() for r in self.rules],
            "logic": self.logic,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for all tunable parameters."""
        return {
            name: param.bounds 
            for name, param in self.parameters.items() 
            if param.tunable
        }
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """Get current values of all parameters."""
        return {name: param.value for name, param in self.parameters.items()}


class StrategyParseError(Exception):
    """Error during strategy parsing."""
    pass


class StrategyParser:
    """
    Parser for user-defined trading strategies.
    
    Converts natural language-like strategy descriptions into
    structured StrategySpec objects that can be executed and optimized.
    
    Usage:
        parser = StrategyParser()
        spec = parser.parse("Buy when RSI > 30, Sell when RSI < 70")
        
        if spec.valid:
            print(spec.parameters)  # Parameters ML can tune
            print(spec.rules)       # Fixed trading rules
    """
    
    # Token patterns
    PATTERNS = {
        "action": r"\b(buy|sell|exit|hold)\b",
        "when": r"\bwhen\b",
        "and": r"\band\b",
        "or": r"\bor\b",
        "crosses": r"\bcrosses\s+(above|below)\b",
        "operator": r"(>=|<=|>|<|==|!=)",
        "number": r"-?\d+\.?\d*",
        "indicator": r"\b([a-zA-Z][a-zA-Z0-9_%]*)(?:\s*\(\s*(\d+)\s*\))?",

    }
    
    def __init__(self, custom_indicators: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser.
        
        Args:
            custom_indicators: Additional indicator definitions
        """
        self.indicators = SUPPORTED_INDICATORS.copy()
        if custom_indicators:
            self.indicators.update(custom_indicators)
    
    def parse(self, strategy_text: str) -> StrategySpec:
        """
        Parse strategy text into a StrategySpec.
        
        Args:
            strategy_text: User-entered strategy description
            
        Returns:
            StrategySpec with parsed rules and parameters
        """
        errors = []
        warnings = []
        rules = []
        parameters = {}
        indicators_used = set()
        
        # Normalize text
        text = strategy_text.lower().strip()
        
        # Split into rules by comma (allowing for variations)
        rule_texts = re.split(r'[,;]\s*', text)
        
        for rule_text in rule_texts:
            rule_text = rule_text.strip()
            if not rule_text:
                continue
            
            try:
                rule, rule_params, rule_indicators = self._parse_rule(rule_text)
                rules.append(rule)
                parameters.update(rule_params)
                indicators_used.update(rule_indicators)
            except StrategyParseError as e:
                errors.append(str(e))
        
        # Validate indicators exist
        for ind in indicators_used:
            if ind.lower() not in self.indicators:
                errors.append(f"Unknown indicator: {ind}")
        
        # Check we have at least one buy and one sell rule
        actions = [r.action for r in rules]
        if ActionType.BUY not in actions:
            warnings.append("No BUY rule defined")
        if ActionType.SELL not in actions:
            warnings.append("No SELL rule defined")
        
        return StrategySpec(
            raw_text=strategy_text,
            indicators_used=list(indicators_used),
            parameters=parameters,
            rules=rules,
            logic="FIXED",
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _parse_rule(
        self, 
        rule_text: str
    ) -> Tuple[Rule, Dict[str, TunableParameter], Set[str]]:
        """
        Parse a single rule from text.
        
        Args:
            rule_text: Text of one rule (e.g., "buy when rsi > 30")
            
        Returns:
            Tuple of (Rule, parameters dict, indicators set)
        """
        parameters = {}
        indicators = set()
        
        # Extract action
        action_match = re.search(self.PATTERNS["action"], rule_text, re.IGNORECASE)
        if not action_match:
            raise StrategyParseError(f"No action found in rule: '{rule_text}'")
        
        action_str = action_match.group(1).lower()
        action = ActionType(action_str)
        
        # Find "when" and get conditions part
        when_match = re.search(self.PATTERNS["when"], rule_text, re.IGNORECASE)
        if not when_match:
            raise StrategyParseError(f"Missing 'when' in rule: '{rule_text}'")
        
        conditions_text = rule_text[when_match.end():].strip()
        
        # Determine logic operator (and/or)
        logic_op = "and"
        if re.search(self.PATTERNS["or"], conditions_text, re.IGNORECASE):
            logic_op = "or"
        
        # Split conditions by and/or
        condition_parts = re.split(r'\s+(?:and|or)\s+', conditions_text, flags=re.IGNORECASE)
        
        conditions = []
        for cond_text in condition_parts:
            cond_text = cond_text.strip()
            if not cond_text:
                continue
            
            cond, cond_params, cond_indicators = self._parse_condition(cond_text, action)
            conditions.append(cond)
            parameters.update(cond_params)
            indicators.update(cond_indicators)
        
        if not conditions:
            raise StrategyParseError(f"No conditions found in rule: '{rule_text}'")
        
        return Rule(action=action, conditions=conditions, logic_operator=logic_op), parameters, indicators
    
    def _parse_condition(
        self, 
        cond_text: str,
        action: ActionType
    ) -> Tuple[Condition, Dict[str, TunableParameter], Set[str]]:
        """
        Parse a single condition.
        
        Args:
            cond_text: Condition text (e.g., "rsi > 30" or "ema(20) crosses above sma(50)")
            action: The action this condition belongs to
            
        Returns:
            Tuple of (Condition, parameters, indicators)
        """
        parameters = {}
        indicators = set()
        
        # Check for crosses pattern
        crosses_match = re.search(self.PATTERNS["crosses"], cond_text, re.IGNORECASE)
        if crosses_match:
            return self._parse_crosses_condition(cond_text, crosses_match, action)
        
        # Standard comparison: indicator op value
        # Pattern: indicator_name[(period)] operator number
        pattern = r'(\w+)(?:\s*\(\s*(\d+)\s*\))?\s*(>=|<=|>|<|==|!=)\s*(-?\d+\.?\d*)'
        match = re.search(pattern, cond_text, re.IGNORECASE)
        
        if not match:
            raise StrategyParseError(f"Cannot parse condition: '{cond_text}'")
        
        ind_name = match.group(1).lower()
        ind_period = int(match.group(2)) if match.group(2) else None
        op_str = match.group(3)
        value = float(match.group(4))
        
        # Validate indicator
        if ind_name not in self.indicators:
            raise StrategyParseError(f"Unknown indicator: {ind_name}")
        
        ind_spec = self.indicators[ind_name]
        
        # Use default period if not specified
        if ind_period is None and "default_period" in ind_spec:
            ind_period = ind_spec["default_period"]
        
        # Create indicator reference
        indicator = IndicatorRef(name=ind_name, period=ind_period)
        indicators.add(ind_name)
        
        # Map operator
        op_map = {
            ">": ComparisonOp.GT,
            ">=": ComparisonOp.GTE,
            "<": ComparisonOp.LT,
            "<=": ComparisonOp.LTE,
            "==": ComparisonOp.EQ,
            "!=": ComparisonOp.NEQ
        }
        operator = op_map[op_str]
        
        # Create tunable parameter for the threshold
        param_name = f"{ind_name}_threshold_{action.value}"
        if ind_period:
            param_name = f"{ind_name}_{ind_period}_threshold_{action.value}"
        
        # Get bounds for this indicator's value
        value_bounds = ind_spec.get("value_bounds", (value * 0.5, value * 1.5))
        
        # Adjust bounds based on typical usage
        if ind_name == "rsi":
            if action == ActionType.BUY and operator in (ComparisonOp.LT, ComparisonOp.LTE):
                value_bounds = (15, 50)  # Oversold buy
            elif action == ActionType.SELL and operator in (ComparisonOp.GT, ComparisonOp.GTE):
                value_bounds = (50, 85)  # Overbought sell
            else:
                value_bounds = (20, 80)
        
        parameters[param_name] = TunableParameter(
            name=param_name,
            value=value,
            tunable=True,
            bounds=value_bounds,
            param_type="float" if "." in str(value) else "int",
            description=f"{ind_name} threshold for {action.value} signal"
        )
        
        # Also create period parameter if applicable
        if ind_period and "period_bounds" in ind_spec:
            period_param_name = f"{ind_name}_period"
            parameters[period_param_name] = TunableParameter(
                name=period_param_name,
                value=ind_period,
                tunable=True,
                bounds=ind_spec["period_bounds"],
                param_type="int",
                description=f"{ind_name} lookback period"
            )
        
        return Condition(indicator=indicator, operator=operator, value=value), parameters, indicators
    
    def _parse_crosses_condition(
        self,
        cond_text: str,
        crosses_match: re.Match,
        action: ActionType
    ) -> Tuple[Condition, Dict[str, TunableParameter], Set[str]]:
        """Parse a 'crosses above/below' condition."""
        parameters = {}
        indicators = set()
        
        direction = crosses_match.group(1).lower()
        
        # Split by "crosses"
        parts = re.split(r'\s+crosses\s+', cond_text, flags=re.IGNORECASE)
        if len(parts) != 2:
            raise StrategyParseError(f"Invalid crosses syntax: '{cond_text}'")
        
        # Parse left indicator
        left_text = parts[0].strip()
        left_match = re.match(r'(\w+)(?:\s*\(\s*(\d+)\s*\))?', left_text)
        if not left_match:
            raise StrategyParseError(f"Cannot parse indicator: '{left_text}'")
        
        left_name = left_match.group(1).lower()
        left_period = int(left_match.group(2)) if left_match.group(2) else None
        
        if left_name not in self.indicators:
            raise StrategyParseError(f"Unknown indicator: {left_name}")
        
        indicators.add(left_name)
        left_ind = IndicatorRef(name=left_name, period=left_period)
        
        # Parse right side (after "above"/"below")
        right_text = re.sub(r'^(above|below)\s+', '', parts[1].strip(), flags=re.IGNORECASE)
        
        # Check if right side is a number or indicator
        if re.match(r'^-?\d+\.?\d*$', right_text):
            value = float(right_text)
            param_name = f"{left_name}_cross_level_{action.value}"
            parameters[param_name] = TunableParameter(
                name=param_name,
                value=value,
                tunable=True,
                bounds=(value * 0.5, value * 1.5),
                param_type="float"
            )
        else:
            # Right side is another indicator
            right_match = re.match(r'(\w+)(?:\s*\(\s*(\d+)\s*\))?', right_text)
            if not right_match:
                raise StrategyParseError(f"Cannot parse: '{right_text}'")
            
            right_name = right_match.group(1).lower()
            right_period = int(right_match.group(2)) if right_match.group(2) else None
            
            if right_name not in self.indicators:
                raise StrategyParseError(f"Unknown indicator: {right_name}")
            
            indicators.add(right_name)
            value = IndicatorRef(name=right_name, period=right_period)
            
            # Add period parameters for both indicators
            for ind_name, period in [(left_name, left_period), (right_name, right_period)]:
                if period:
                    ind_spec = self.indicators[ind_name]
                    if "period_bounds" in ind_spec:
                        param_name = f"{ind_name}_period"
                        if param_name not in parameters:
                            parameters[param_name] = TunableParameter(
                                name=param_name,
                                value=period,
                                tunable=True,
                                bounds=ind_spec["period_bounds"],
                                param_type="int"
                            )
        
        operator = ComparisonOp.CROSSES_ABOVE if direction == "above" else ComparisonOp.CROSSES_BELOW
        
        return Condition(indicator=left_ind, operator=operator, value=value), parameters, indicators
    
    def validate_features(
        self, 
        spec: StrategySpec, 
        available_features: List[str]
    ) -> List[str]:
        """
        Validate that required features are available.
        
        Args:
            spec: Parsed strategy specification
            available_features: List of available feature column names
            
        Returns:
            List of missing features (empty if all present)
        """
        missing = []
        
        for rule in spec.rules:
            for cond in rule.conditions:
                feature_name = cond.indicator.get_feature_name()
                if feature_name not in available_features:
                    missing.append(feature_name)
                
                # Check if value is also an indicator
                if isinstance(cond.value, IndicatorRef):
                    val_feature = cond.value.get_feature_name()
                    if val_feature not in available_features:
                        missing.append(val_feature)
        
        return list(set(missing))


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_parser():
    """Unit tests for the strategy parser."""
    parser = StrategyParser()
    
    # Test 1: Simple RSI strategy
    print("Test 1: Simple RSI strategy")
    spec = parser.parse("Buy when RSI > 30, Sell when RSI < 70")
    print(f"  Valid: {spec.valid}")
    print(f"  Indicators: {spec.indicators_used}")
    print(f"  Parameters: {list(spec.parameters.keys())}")
    print(f"  Rules: {len(spec.rules)}")
    assert spec.valid
    assert "rsi" in spec.indicators_used
    print("  ✓ PASSED\n")
    
    # Test 2: RSI with period
    print("Test 2: RSI with custom period")
    spec = parser.parse("Buy when RSI(21) < 25, Sell when RSI(21) > 75")
    print(f"  Valid: {spec.valid}")
    print(f"  Parameters: {spec.parameters}")
    assert spec.valid
    assert "rsi_period" in spec.parameters
    print("  ✓ PASSED\n")
    
    # Test 3: Multiple indicators
    print("Test 3: Multiple indicators")
    spec = parser.parse("Buy when RSI < 30 and ADX > 25, Sell when RSI > 70")
    print(f"  Valid: {spec.valid}")
    print(f"  Indicators: {spec.indicators_used}")
    assert spec.valid
    assert "rsi" in spec.indicators_used
    assert "adx" in spec.indicators_used
    print("  ✓ PASSED\n")
    
    # Test 4: EMA crossover
    print("Test 4: EMA crossover")
    spec = parser.parse("Buy when EMA(20) crosses above SMA(50), Sell when EMA(20) crosses below SMA(50)")
    print(f"  Valid: {spec.valid}")
    print(f"  Indicators: {spec.indicators_used}")
    print(f"  Parameters: {list(spec.parameters.keys())}")
    assert spec.valid
    assert "ema" in spec.indicators_used
    assert "sma" in spec.indicators_used
    print("  ✓ PASSED\n")
    
    # Test 5: Invalid indicator
    print("Test 5: Invalid indicator (should fail)")
    spec = parser.parse("Buy when FOOBAR > 50")
    print(f"  Valid: {spec.valid}")
    print(f"  Errors: {spec.errors}")
    assert not spec.valid
    print("  ✓ PASSED\n")
    
    # Test 6: Complex multi-condition
    print("Test 6: Complex multi-condition")
    spec = parser.parse("Buy when RSI < 30 and BB_percent < 0.2, Sell when RSI > 70 or BB_percent > 0.8")
    print(f"  Valid: {spec.valid}")
    print(f"  Indicators: {spec.indicators_used}")
    print(f"  Rules: {[r.to_dict() for r in spec.rules]}")
    assert spec.valid
    print("  ✓ PASSED\n")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_parser()
