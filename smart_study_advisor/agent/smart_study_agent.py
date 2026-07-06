"""Smart Study Advisor — Sense → Think → Act → Learn orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agent.act import recommend
from agent.learn import FeedbackStore
from agent.sense import collect_from_mapping, collect_student_input
from agent.think import PredictionService
from agent.types import FeedbackRecord, PredictionResult, StrategyRecommendation, StudentContext

logger = logging.getLogger(__name__)


@dataclass
class AgentStepResult:
    context: StudentContext
    prediction: PredictionResult
    recommendation: StrategyRecommendation
    feedback_record: FeedbackRecord | None = None


class SmartStudyAgent:
    """
    Intelligent ML agent that recommends a study strategy for students.

    Workflow:
        Sense  → collect and validate student input
        Think  → load saved ML artifacts and predict RecommendedStrategy
        Act    → return strategy, confidence, explanation, and advice
        Learn  → store prediction and optional user feedback
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        self.prediction_service = PredictionService(models_dir=models_dir)
        self.feedback_store = feedback_store or FeedbackStore()
        logger.info("SmartStudyAgent initialized")

    def run(
        self,
        context: StudentContext,
        *,
        save_for_learning: bool = True,
        user_feedback: str | None = None,
    ) -> AgentStepResult:
        """Execute one full Sense → Think → Act → Learn cycle."""
        logger.info("Agent cycle started")

        # Sense (context already validated if built via collect_student_input)
        logger.info("Sense: percept ready")

        # Think
        prediction = self.prediction_service.predict(context)

        # Act
        recommendation = recommend(context, prediction)

        # Learn
        feedback_record = None
        if save_for_learning:
            feedback_record = self.feedback_store.save_prediction(
                context,
                recommendation,
                user_feedback=user_feedback,
            )

        logger.info("Agent cycle completed — strategy=%s", recommendation.strategy)
        return AgentStepResult(
            context=context,
            prediction=prediction,
            recommendation=recommendation,
            feedback_record=feedback_record,
        )

    def run_from_form(self, form_data: dict, **kwargs) -> AgentStepResult:
        """Convenience entry point for UI forms."""
        context = collect_from_mapping(form_data)
        return self.run(context, **kwargs)

    def submit_feedback(self, record_id: str, user_feedback: str) -> bool:
        """Learn: attach user feedback to a prior prediction."""
        return self.feedback_store.add_feedback(record_id, user_feedback)

    @staticmethod
    def sense(**fields) -> StudentContext:
        return collect_student_input(**fields)
