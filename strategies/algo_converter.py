"""
Algorithm Converter Utility.

Converts natural language trading descriptions into structured, 
variable-based strategy specifications.

Generic Format:
    [Action] [Symbol] if [Condition] [Logic] [Condition], ...
    
Example:
    "buy btc if rsi < 30 and adx > 25, sell btc if rsi > 70"
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from strategies.strategy_parser import StrategyParser, StrategySpec, ActionType

logger = logging.getLogger(__name__)

class AlgoConverter:
    """
    High-level converter for natural language trading algorithms.
    
    Building on top of StrategyParser, this adds symbol detection,
    more flexible syntax (if/when), and better noise filtering.
    """
    
    def __init__(self):
        self.parser = StrategyParser()
        
    def convert_to_variables(self, text: str) -> Dict[str, Any]:
        """
        Convert natural language text to a structured variable format.
        
        Args:
            text: Natural language strategy string.
            
        Returns:
            Dictionary containing the variable-based specification.
        """
        # 1. Pre-process and normalize
        cleaned_text = self._preprocess(text)
        
        # 2. Extract symbols (if any)
        symbols = self._extract_symbols(text)
        
        # 3. Parse conditions and parameters using the underlying DSL parser
        spec = self.parser.parse(cleaned_text)
        
        # 4. Format into final variable structure
        result = {
            "original_input": text,
            "normalized_logic": cleaned_text,
            "detected_symbols": symbols,
            "valid": spec.valid,
            "errors": spec.errors,
            "warnings": spec.warnings,
            "variable_format": {
                "indicators": spec.indicators_used,
                "parameters": {
                    name: {
                        "value": param.value,
                        "bounds": param.bounds,
                        "type": param.param_type,
                        "tunable": param.tunable
                    }
                    for name, param in spec.parameters.items()
                },
                "execution_rules": [
                    {
                        "action": rule.action.value,
                        "logic": rule.logic_operator,
                        "conditions": [
                            {
                                "indicator": cond.indicator.name,
                                "indicator_period": cond.indicator.period,
                                "operator": cond.operator.value,
                                "threshold_variable": f"{cond.indicator.name}_{cond.indicator.period if cond.indicator.period else 'default'}_threshold_{rule.action.value}"
                            }
                            for cond in rule.conditions
                        ]
                    }
                    for rule in spec.rules
                ]
            }
        }
        
        return result

    def _preprocess(self, text: str) -> str:
        """Clean and normalize the input text for the DSL parser."""
        # Replace 'if' with 'when'
        processed = re.sub(r'\bif\b', 'when', text, flags=re.IGNORECASE)
        
        # Handle 'and' or 'or' between rules (e.g., "buy ... and sell ...")
        # We want to change it to "buy ..., sell ..." so the parser sees separate rules
        processed = re.sub(r'\band\s+(buy|sell|exit|hold)\b', r', \1', processed, flags=re.IGNORECASE)
        
        # Remove common noise words (symbol names etc)
        # We'll remove specific coins if they appear directly after buy/sell
        processed = re.sub(r'\b(buy|sell|exit|hold)\s+[a-z]{2,10}\s+when\b', r'\1 when', processed, flags=re.IGNORECASE)
        
        return processed

    def _extract_symbols(self, text: str) -> List[str]:
        """Attempt to extract likely trade symbols from the text."""
        # Look for 3-5 letter uppercase words or common coins
        common_coins = r'\b(btc|eth|sol|bnb|xrp|ada|doge|dot|matic|ltc)\b'
        matches = re.findall(common_coins, text, flags=re.IGNORECASE)
        
        # Also look for uppercase symbols like BTCUSDT
        matches += re.findall(r'\b[A-Z]{3,10}\b', text)
        
        return sorted(list(set(m.upper() for m in matches)))

def main():
    """Demo the converter."""
    converter = AlgoConverter()
    
    examples = [
        "buy btc if rsi < 30 and adx > 25, sell btc if rsi > 70",
        "buy eth when ema(50) crosses above sma(200)",
        "buy when macd > 0 and rsi < 50 and sell when rsi > 80"
    ]
    
    for ex in examples:
        print(f"\nExample: {ex}")
        result = converter.convert_to_variables(ex)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
