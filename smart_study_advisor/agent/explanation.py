"""Human-readable explanations for study strategy predictions."""

from __future__ import annotations

from agent.types import PredictionResult, StudentContext


def _sleep_phrase(context: StudentContext) -> str:
    quality = context.sleep_quality.lower()
    if quality == "good":
        return "good sleep quality"
    if quality == "poor":
        return "poor sleep quality"
    return "average sleep quality"


def _stress_phrase(level: str) -> str:
    return f"{level.lower()} stress level"


def _fatigue_phrase(level: str) -> str:
    return f"{level.lower()} fatigue level"


def _exam_phrase(days: int) -> str:
    if days <= 2:
        return "the exam is very soon"
    if days <= 7:
        return "there is limited time before the exam"
    if days <= 14:
        return "there is a moderate amount of time before the exam"
    return "there is sufficient time before the exam"


def build_explanation(context: StudentContext, prediction: PredictionResult) -> str:
    """Create a short explanation referencing the student's input values."""
    strategy = prediction.recommended_strategy
    sleep = _sleep_phrase(context)
    stress = _stress_phrase(context.stress_level)
    fatigue = _fatigue_phrase(context.fatigue_level)
    exam = _exam_phrase(context.days_until_exam)

    if strategy == "Rest":
        return (
            f"{fatigue.capitalize()}, {stress}, {sleep}, and {exam} "
            "suggest that recovery should take priority over additional studying."
        )
    if strategy == "BalancedStudy":
        return (
            f"{sleep.capitalize()}, {stress}, {fatigue}, and {exam} "
            "indicate that a balanced study schedule is the best option."
        )
    if strategy == "IntensiveStudy":
        return (
            f"High study load ({context.hours_studied:.1f} h/day), {exam}, "
            f"and {stress} support an intensive short-term study push."
        )
    if strategy == "LongTermPlan":
        return (
            f"{exam.capitalize()}, {stress}, {fatigue}, and {sleep} "
            "support building a structured long-term study plan."
        )
    return (
        f"Based on {sleep}, {stress}, {fatigue}, and {exam}, "
        f"{strategy} is the recommended approach."
    )


def build_advice(strategy: str) -> str:
    """Return practical study advice for the predicted strategy."""
    advice_map = {
        "Rest": (
            "Take a recovery day: aim for 7-9 hours of sleep, reduce screen time before bed, "
            "and postpone heavy study until fatigue and stress decrease."
        ),
        "BalancedStudy": (
            "Study in focused 45-60 minute blocks with short breaks. Cover your highest-priority "
            "subjects first and keep a sustainable daily schedule."
        ),
        "IntensiveStudy": (
            "Prioritize the most exam-relevant topics, use active recall and practice questions, "
            "and schedule brief breaks to maintain focus during the final days."
        ),
        "LongTermPlan": (
            "Create a weekly plan that spreads revision across all subjects, set milestone goals, "
            "and review progress regularly while maintaining consistent sleep."
        ),
    }
    return advice_map.get(
        strategy,
        "Follow a study routine that matches your current energy, stress, and exam timeline.",
    )
