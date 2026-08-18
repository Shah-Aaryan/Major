"""
Autonomous ML pipeline: candle -> rolling window -> features -> optimization
-> parallel human/ML execution -> portfolio/evaluation -> charts -> audit.
"""

from pipeline.autonomous_loop import STRATEGY_FACTORY, AutonomousPipeline, CycleResult

__all__ = ["AutonomousPipeline", "CycleResult", "STRATEGY_FACTORY"]
