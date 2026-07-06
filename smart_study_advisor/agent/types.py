"""Shared types for the Smart Study Advisor agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class StudentContext:
    """Student input collected during the Sense phase."""

    hours_studied: float
    number_of_subjects: int
    days_until_exam: int
    stress_level: str
    fatigue_level: str
    sleep_hours: float
    sleep_quality: str
    previous_feedback: str

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "HoursStudied": self.hours_studied,
            "NumberOfSubjects": self.number_of_subjects,
            "DaysUntilExam": self.days_until_exam,
            "StressLevel": self.stress_level,
            "FatigueLevel": self.fatigue_level,
            "SleepHours": self.sleep_hours,
            "SleepQuality": self.sleep_quality,
            "PreviousFeedback": self.previous_feedback,
        }


@dataclass(frozen=True)
class PredictionResult:
    """Output of the Think phase."""

    recommended_strategy: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class StrategyRecommendation:
    """Output of the Act phase."""

    strategy: str
    confidence: float
    explanation: str
    advice: str


@dataclass
class FeedbackRecord:
    """Stored during the Learn phase."""

    timestamp: datetime
    context: StudentContext
    predicted_strategy: str
    confidence: float
    explanation: str
    advice: str
    user_feedback: str | None = None
    feedback_timestamp: datetime | None = None
    record_id: str = field(default="")
