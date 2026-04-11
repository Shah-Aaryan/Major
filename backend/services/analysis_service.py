"""Service layer for analysis operations (condition, failure, comparison, explainability, leakage)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.models.request_models import (
    AnalysisRequest,
    ExplainabilityRequest,
    LeakageCheckRequest,
)
from backend.models.response_models import (
    ComparisonReportResponse,
    ConditionAnalysisResponse,
    ExplainabilityResponse,
    FailureDetectionResponse,
    LeakageCheckResponse,
)
from backend.utils.session_files import get_research_output_dir, load_session_file
from analysis.condition_analyzer import ConditionAnalyzer
from analysis.failure_detector import FailureDetector
from analysis.comparison_report import generate_full_report
from analysis.explainability import ParameterExplainer
from analysis.leakage_checker import LeakageChecker
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.resampler import DataResampler
from features.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class AnalysisService:
    """Bridges the analysis module with the FastAPI backend."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or get_research_output_dir()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_session(self, session_id: str) -> dict[str, Any]:
        data = load_session_file(session_id, self.output_dir)
        if data is None:
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        return data

    def _extract_results_pair(
        self, session_data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract first human_result / ml_result pair from a session file."""
        results = session_data.get("results", [])
        if not results:
            raise ValueError("Session has no experiment results.")
        first = results[0]
        human = first.get("human_result", {})
        ml = first.get("best_ml_result", first.get("ml_result", {}))
        return human, ml

    # ──────────────────────────────────────────────────────────────────────────
    # Condition Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def run_condition_analysis(self, payload: AnalysisRequest) -> ConditionAnalysisResponse:
        try:
            session_data = self._load_session(payload.session_id)
            human, ml = self._extract_results_pair(session_data)

            # Build condition_results list from the stored results
            results = session_data.get("results", [])
            condition_results: list = []
            for r in results:
                condition_dict = r.get("condition", {})
                from analysis.condition_analyzer import MarketCondition, TrendRegime, VolatilityRegime, VolumeRegime
                try:
                    cond = MarketCondition(
                        trend=TrendRegime(condition_dict.get("trend", "ranging")),
                        volatility=VolatilityRegime(condition_dict.get("volatility", "normal")),
                        volume=VolumeRegime(condition_dict.get("volume", "normal")),
                    )
                except Exception:
                    from analysis.condition_analyzer import TrendRegime, VolatilityRegime, VolumeRegime
                    cond = MarketCondition(
                        trend=TrendRegime.RANGING,
                        volatility=VolatilityRegime.NORMAL,
                        volume=VolumeRegime.NORMAL,
                    )

                condition_results.append((
                    cond,
                    {
                        "human_metrics": r.get("human_result", {}).get("metrics", {}),
                        "ml_metrics": r.get("best_ml_result", {}).get("metrics", {}),
                        "ml_params": r.get("best_ml_params", {}),
                    },
                ))

            analyzer = ConditionAnalyzer()
            if condition_results:
                analysis = analyzer.analyze_ml_effectiveness(condition_results)
            else:
                analysis = {"message": "No condition data in session results."}

            return ConditionAnalysisResponse(
                session_id=payload.session_id,
                status="completed",
                analysis=analysis,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Condition analysis failed for session %s", payload.session_id)
            return ConditionAnalysisResponse(
                session_id=payload.session_id,
                status="failed",
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Failure Detection
    # ──────────────────────────────────────────────────────────────────────────

    def run_failure_detection(self, payload: AnalysisRequest) -> FailureDetectionResponse:
        try:
            session_data = self._load_session(payload.session_id)
            human, ml = self._extract_results_pair(session_data)

            train_metrics = human.get("train_metrics", human.get("metrics", {}))
            test_metrics = ml.get("metrics", {})
            baseline_metrics = human.get("metrics", {})
            ml_params = session_data.get("results", [{}])[0].get("best_ml_params", {})

            detector = FailureDetector()
            failures = detector.detect_failures(
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                baseline_metrics=baseline_metrics,
                ml_params=ml_params,
                strategy_name=session_data.get("results", [{}])[0].get("strategy_name", ""),
            )

            should_use, reason = detector.should_use_baseline()
            summary = detector.get_failure_summary()

            return FailureDetectionResponse(
                session_id=payload.session_id,
                status="completed",
                failures=[f.to_dict() for f in failures],
                failure_summary=summary,
                should_use_baseline=should_use,
                reason=reason,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Failure detection failed for session %s", payload.session_id)
            return FailureDetectionResponse(
                session_id=payload.session_id,
                status="failed",
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Comparison Report
    # ──────────────────────────────────────────────────────────────────────────

    def generate_comparison_report(self, payload: AnalysisRequest) -> ComparisonReportResponse:
        try:
            session_data = self._load_session(payload.session_id)
            results = session_data.get("results", [])
            if not results:
                raise ValueError("Session has no experiment results.")

            first = results[0]
            strategy_name = first.get("strategy_name", "unknown")
            human_result = first.get("human_result", {})
            ml_result = first.get("best_ml_result", {})
            human_params = first.get("human_params", {})
            ml_params = first.get("best_ml_params", {})

            report = generate_full_report(
                strategy_name=strategy_name,
                human_results=human_result,
                ml_results=ml_result,
                human_params=human_params,
                ml_params=ml_params,
            )

            return ComparisonReportResponse(
                session_id=payload.session_id,
                status="completed",
                report=report.to_dict(),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Comparison report failed for session %s", payload.session_id)
            return ComparisonReportResponse(
                session_id=payload.session_id,
                status="failed",
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Explainability
    # ──────────────────────────────────────────────────────────────────────────

    def generate_explainability_report(
        self, payload: ExplainabilityRequest
    ) -> ExplainabilityResponse:
        try:
            session_data = self._load_session(payload.session_id)
            results = session_data.get("results", [])

            explainer = ParameterExplainer()
            for r in results:
                human_params = r.get("human_params", {})
                ml_params = r.get("best_ml_params", {})
                human_sharpe = r.get("human_result", {}).get("metrics", {}).get("sharpe_ratio", 0.0)
                ml_sharpe = r.get("best_ml_result", {}).get("metrics", {}).get("sharpe_ratio", 0.0)

                if human_params and ml_params:
                    explainer.log_optimization_result(
                        human_params=human_params,
                        ml_params=ml_params,
                        human_objective=human_sharpe,
                        ml_objective=ml_sharpe,
                        optimization_method=r.get("optimizer", ""),
                        market_condition=str(r.get("condition", "")),
                    )

            report = explainer.generate_report(payload.strategy)

            return ExplainabilityResponse(
                session_id=payload.session_id,
                strategy=payload.strategy,
                status="completed",
                report=report.to_dict(),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Explainability report failed for session %s", payload.session_id)
            return ExplainabilityResponse(
                session_id=payload.session_id,
                strategy=payload.strategy,
                status="failed",
                error=str(exc),
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Leakage Check
    # ──────────────────────────────────────────────────────────────────────────

    def check_leakage(self, payload: LeakageCheckRequest) -> LeakageCheckResponse:
        try:
            data_path = Path(payload.data_path)
            if not data_path.is_absolute():
                data_path = Path.cwd() / data_path
            data_path = data_path.resolve()

            if not data_path.exists():
                raise FileNotFoundError(f"Data path '{data_path}' does not exist.")

            loader = DataLoader()
            preprocessor = DataPreprocessor(normalize_prices=False, normalize_volume=False)

            raw_data = loader.load_csv(str(data_path))
            df, _ = preprocessor.preprocess(raw_data)

            engine = FeatureEngine()
            features = engine.generate_features(df, drop_na=False)

            train_end = int(len(features) * payload.train_ratio)
            checker = LeakageChecker()
            report = checker.check_dataframe(features, train_end_idx=train_end)

            return LeakageCheckResponse(
                data_path=str(data_path),
                status="completed",
                passed=report.overall_passed,
                critical_issues=report.critical_issues,
                warnings_count=report.warnings_count,
                report=report.to_dict(),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise
        except Exception as exc:
            logger.exception("Leakage check failed for %s", payload.data_path)
            return LeakageCheckResponse(
                data_path=payload.data_path,
                status="failed",
                error=str(exc),
            )


analysis_service = AnalysisService()

