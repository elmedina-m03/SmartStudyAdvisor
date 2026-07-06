"""Sense phase — collect and validate student input."""

from __future__ import annotations

import logging
from typing import Any

from agent.types import StudentContext

logger = logging.getLogger(__name__)

ALLOWED = {
    "StressLevel": {"Low", "Medium", "High"},
    "FatigueLevel": {"Low", "Medium", "High"},
    "SleepQuality": {"Poor", "Average", "Good"},
    "PreviousFeedback": {"None", "Positive", "Negative", "Mixed"},
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "HoursStudied": (0, 12),
    "NumberOfSubjects": (1, 10),
    "DaysUntilExam": (0, 60),
    "SleepHours": (0, 12),
}


def _validate_choice(name: str, value: str, allowed: set[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got '{value}'")
    return value


def _validate_numeric(name: str, value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


def collect_student_input(
    *,
    hours_studied: float,
    number_of_subjects: int,
    days_until_exam: int,
    stress_level: str,
    fatigue_level: str,
    sleep_hours: float,
    sleep_quality: str,
    previous_feedback: str,
) -> StudentContext:
    """Validate raw user input and return a StudentContext percept."""
    logger.info("Sense: collecting student input")

    context = StudentContext(
        hours_studied=_validate_numeric("HoursStudied", float(hours_studied), NUMERIC_RANGES["HoursStudied"]),
        number_of_subjects=int(
            _validate_numeric("NumberOfSubjects", float(number_of_subjects), NUMERIC_RANGES["NumberOfSubjects"])
        ),
        days_until_exam=int(
            _validate_numeric("DaysUntilExam", float(days_until_exam), NUMERIC_RANGES["DaysUntilExam"])
        ),
        stress_level=_validate_choice("StressLevel", stress_level, ALLOWED["StressLevel"]),
        fatigue_level=_validate_choice("FatigueLevel", fatigue_level, ALLOWED["FatigueLevel"]),
        sleep_hours=_validate_numeric("SleepHours", float(sleep_hours), NUMERIC_RANGES["SleepHours"]),
        sleep_quality=_validate_choice("SleepQuality", sleep_quality, ALLOWED["SleepQuality"]),
        previous_feedback=_validate_choice("PreviousFeedback", previous_feedback, ALLOWED["PreviousFeedback"]),
    )
    logger.debug("Sense: validated context %s", context.to_feature_dict())
    return context


def collect_from_mapping(data: dict[str, Any]) -> StudentContext:
    """Build a StudentContext from a dictionary (e.g. Streamlit form)."""
    return collect_student_input(
        hours_studied=data["HoursStudied"],
        number_of_subjects=data["NumberOfSubjects"],
        days_until_exam=data["DaysUntilExam"],
        stress_level=data["StressLevel"],
        fatigue_level=data["FatigueLevel"],
        sleep_hours=data["SleepHours"],
        sleep_quality=data["SleepQuality"],
        previous_feedback=data["PreviousFeedback"],
    )
