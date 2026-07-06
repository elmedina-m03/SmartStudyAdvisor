"""Learn phase — persist predictions and user feedback for future retraining."""

from __future__ import annotations

import csv
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent.types import FeedbackRecord, StrategyRecommendation, StudentContext

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
DEFAULT_CSV_PATH = DEFAULT_FEEDBACK_DIR / "prediction_history.csv"
DEFAULT_DB_PATH = DEFAULT_FEEDBACK_DIR / "feedback.db"

CSV_COLUMNS = [
    "record_id",
    "timestamp",
    "HoursStudied",
    "NumberOfSubjects",
    "DaysUntilExam",
    "StressLevel",
    "FatigueLevel",
    "SleepHours",
    "SleepQuality",
    "PreviousFeedback",
    "PredictedStrategy",
    "Confidence",
    "Explanation",
    "Advice",
    "UserFeedback",
    "FeedbackTimestamp",
]


class FeedbackStore:
    """Stores prediction history in CSV and SQLite for future retraining."""

    def __init__(
        self,
        feedback_dir: Path | None = None,
        *,
        use_sqlite: bool = True,
    ) -> None:
        self.feedback_dir = (feedback_dir or DEFAULT_FEEDBACK_DIR).resolve()
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.feedback_dir / "prediction_history.csv"
        self.db_path = self.feedback_dir / "feedback.db"
        self.use_sqlite = use_sqlite
        self._ensure_csv_header()
        if self.use_sqlite:
            self._ensure_sqlite_schema()

    def _ensure_csv_header(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info("Learn: created feedback CSV at %s", self.csv_path)

    def _ensure_sqlite_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    hours_studied REAL,
                    number_of_subjects INTEGER,
                    days_until_exam INTEGER,
                    stress_level TEXT,
                    fatigue_level TEXT,
                    sleep_hours REAL,
                    sleep_quality TEXT,
                    previous_feedback TEXT,
                    predicted_strategy TEXT,
                    confidence REAL,
                    explanation TEXT,
                    advice TEXT,
                    user_feedback TEXT,
                    feedback_timestamp TEXT
                )
                """
            )
        logger.info("Learn: SQLite feedback database ready at %s", self.db_path)

    def save_prediction(
        self,
        context: StudentContext,
        recommendation: StrategyRecommendation,
        *,
        user_feedback: str | None = None,
    ) -> FeedbackRecord:
        """Persist a prediction (and optional immediate feedback)."""
        now = datetime.now(timezone.utc)
        record = FeedbackRecord(
            timestamp=now,
            context=context,
            predicted_strategy=recommendation.strategy,
            confidence=recommendation.confidence,
            explanation=recommendation.explanation,
            advice=recommendation.advice,
            user_feedback=user_feedback,
            feedback_timestamp=now if user_feedback else None,
            record_id=str(uuid.uuid4()),
        )
        self._append_csv(record)
        if self.use_sqlite:
            self._insert_sqlite(record)
        logger.info("Learn: saved prediction record %s", record.record_id)
        return record

    def add_feedback(self, record_id: str, user_feedback: str) -> bool:
        """Attach user feedback to an existing prediction record."""
        if self.use_sqlite:
            updated = self._update_sqlite_feedback(record_id, user_feedback)
            if updated:
                self._sync_csv_from_sqlite()
                logger.info("Learn: feedback stored for record %s", record_id)
                return True
        logger.warning("Learn: record %s not found for feedback update", record_id)
        return False

    def _append_csv(self, record: FeedbackRecord) -> None:
        features = record.context.to_feature_dict()
        row = {
            "record_id": record.record_id,
            "timestamp": record.timestamp.isoformat(),
            **features,
            "PredictedStrategy": record.predicted_strategy,
            "Confidence": record.confidence,
            "Explanation": record.explanation,
            "Advice": record.advice,
            "UserFeedback": record.user_feedback or "",
            "FeedbackTimestamp": record.feedback_timestamp.isoformat() if record.feedback_timestamp else "",
        }
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def _insert_sqlite(self, record: FeedbackRecord) -> None:
        context = record.context
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO prediction_history (
                    record_id, timestamp, hours_studied, number_of_subjects, days_until_exam,
                    stress_level, fatigue_level, sleep_hours, sleep_quality, previous_feedback,
                    predicted_strategy, confidence, explanation, advice,
                    user_feedback, feedback_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.timestamp.isoformat(),
                    context.hours_studied,
                    context.number_of_subjects,
                    context.days_until_exam,
                    context.stress_level,
                    context.fatigue_level,
                    context.sleep_hours,
                    context.sleep_quality,
                    context.previous_feedback,
                    record.predicted_strategy,
                    record.confidence,
                    record.explanation,
                    record.advice,
                    record.user_feedback,
                    record.feedback_timestamp.isoformat() if record.feedback_timestamp else None,
                ),
            )

    def _update_sqlite_feedback(self, record_id: str, user_feedback: str) -> bool:
        feedback_time = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE prediction_history
                SET user_feedback = ?, feedback_timestamp = ?
                WHERE record_id = ?
                """,
                (user_feedback, feedback_time, record_id),
            )
            return cursor.rowcount > 0

    def _sync_csv_from_sqlite(self) -> None:
        """Rewrite CSV from SQLite so both stores stay aligned after feedback updates."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM prediction_history ORDER BY timestamp"
            ).fetchall()

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "record_id": row["record_id"],
                        "timestamp": row["timestamp"],
                        "HoursStudied": row["hours_studied"],
                        "NumberOfSubjects": row["number_of_subjects"],
                        "DaysUntilExam": row["days_until_exam"],
                        "StressLevel": row["stress_level"],
                        "FatigueLevel": row["fatigue_level"],
                        "SleepHours": row["sleep_hours"],
                        "SleepQuality": row["sleep_quality"],
                        "PreviousFeedback": row["previous_feedback"],
                        "PredictedStrategy": row["predicted_strategy"],
                        "Confidence": row["confidence"],
                        "Explanation": row["explanation"],
                        "Advice": row["advice"],
                        "UserFeedback": row["user_feedback"] or "",
                        "FeedbackTimestamp": row["feedback_timestamp"] or "",
                    }
                )

    def recent_records(self, limit: int = 20) -> list[dict]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM prediction_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
