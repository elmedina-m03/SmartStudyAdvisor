"""Act phase — format prediction, confidence, explanation, and advice."""

from __future__ import annotations

import logging

from agent.explanation import build_advice, build_explanation
from agent.types import PredictionResult, StrategyRecommendation, StudentContext

logger = logging.getLogger(__name__)


def recommend(
    context: StudentContext,
    prediction: PredictionResult,
) -> StrategyRecommendation:
    """Turn a model prediction into a user-facing recommendation."""
    strategy = prediction.recommended_strategy
    explanation = build_explanation(context, prediction)
    advice = build_advice(strategy)

    logger.info("Act: returning recommendation %s (confidence=%.4f)", strategy, prediction.confidence)
    return StrategyRecommendation(
        strategy=strategy,
        confidence=prediction.confidence,
        explanation=explanation,
        advice=advice,
    )
