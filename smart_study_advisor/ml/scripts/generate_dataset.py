"""Generate a realistic, balanced Smart Study Advisor dataset (~400 rows)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import ALL_COLUMNS, TARGET_COLUMN, load_schema, validate_dataset

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 100
STRATEGIES = ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"]

LEVELS = ["Low", "Medium", "High"]


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    labels = list(weights.keys())
    probs = np.array([weights[label] for label in labels], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(labels, p=probs))


def _sleep_quality_from_hours(rng: np.random.Generator, hours: float, bias: str) -> str:
    if bias == "good":
        if hours >= 7.5:
            return _pick(rng, {"Good": 0.7, "Average": 0.3})
        return _pick(rng, {"Good": 0.4, "Average": 0.6})
    if bias == "poor":
        if hours <= 5.0:
            return _pick(rng, {"Poor": 0.75, "Average": 0.25})
        return _pick(rng, {"Poor": 0.45, "Average": 0.55})
    if hours >= 7.5:
        return _pick(rng, {"Good": 0.5, "Average": 0.5})
    if hours <= 5.5:
        return _pick(rng, {"Poor": 0.35, "Average": 0.65})
    return _pick(rng, {"Poor": 0.2, "Average": 0.55, "Good": 0.25})


def _feedback_for_strategy(rng: np.random.Generator, strategy: str) -> str:
    profiles: dict[str, dict[str, float]] = {
        "Rest": {"Negative": 0.40, "Mixed": 0.30, "None": 0.20, "Positive": 0.10},
        "BalancedStudy": {"None": 0.35, "Positive": 0.30, "Mixed": 0.20, "Negative": 0.15},
        "IntensiveStudy": {"Positive": 0.40, "Mixed": 0.30, "None": 0.25, "Negative": 0.05},
        "LongTermPlan": {"None": 0.45, "Positive": 0.35, "Mixed": 0.12, "Negative": 0.08},
    }
    return _pick(rng, profiles[strategy])


def _generate_rest(rng: np.random.Generator) -> dict:
    sleep_hours = round(float(rng.uniform(3.5, 5.8)), 1)
    return {
        "HoursStudied": round(float(rng.uniform(0.0, 2.5)), 1),
        "NumberOfSubjects": int(rng.integers(1, 6)),
        "DaysUntilExam": int(rng.integers(0, 4)),
        "StressLevel": _pick(rng, {"High": 0.75, "Medium": 0.20, "Low": 0.05}),
        "FatigueLevel": _pick(rng, {"High": 0.80, "Medium": 0.15, "Low": 0.05}),
        "SleepHours": sleep_hours,
        "SleepQuality": _sleep_quality_from_hours(rng, sleep_hours, "poor"),
        "PreviousFeedback": _feedback_for_strategy(rng, "Rest"),
        TARGET_COLUMN: "Rest",
    }


def _generate_balanced(rng: np.random.Generator, *, negative_feedback_shift: bool = False) -> dict:
    if negative_feedback_shift:
        sleep_hours = round(float(rng.uniform(5.5, 7.0)), 1)
        return {
            "HoursStudied": round(float(rng.uniform(4.0, 7.5)), 1),
            "NumberOfSubjects": int(rng.integers(5, 9)),
            "DaysUntilExam": int(rng.integers(1, 6)),
            "StressLevel": _pick(rng, {"Medium": 0.45, "High": 0.45, "Low": 0.10}),
            "FatigueLevel": _pick(rng, {"Low": 0.35, "Medium": 0.55, "High": 0.10}),
            "SleepHours": sleep_hours,
            "SleepQuality": _sleep_quality_from_hours(rng, sleep_hours, "neutral"),
            "PreviousFeedback": "Negative",
            TARGET_COLUMN: "BalancedStudy",
        }

    sleep_hours = round(float(rng.uniform(6.0, 8.2)), 1)
    return {
        "HoursStudied": round(float(rng.uniform(3.0, 7.0)), 1),
        "NumberOfSubjects": int(rng.integers(3, 7)),
        "DaysUntilExam": int(rng.integers(5, 15)),
        "StressLevel": _pick(rng, {"Low": 0.35, "Medium": 0.50, "High": 0.15}),
        "FatigueLevel": _pick(rng, {"Low": 0.40, "Medium": 0.45, "High": 0.15}),
        "SleepHours": sleep_hours,
        "SleepQuality": _sleep_quality_from_hours(rng, sleep_hours, "neutral"),
        "PreviousFeedback": _feedback_for_strategy(rng, "BalancedStudy"),
        TARGET_COLUMN: "BalancedStudy",
    }


def _generate_intensive(rng: np.random.Generator) -> dict:
    sleep_hours = round(float(rng.uniform(5.2, 7.2)), 1)
    return {
        "HoursStudied": round(float(rng.uniform(5.5, 10.0)), 1),
        "NumberOfSubjects": int(rng.integers(5, 10)),
        "DaysUntilExam": int(rng.integers(1, 5)),
        "StressLevel": _pick(rng, {"Medium": 0.35, "High": 0.55, "Low": 0.10}),
        "FatigueLevel": _pick(rng, {"Low": 0.40, "Medium": 0.45, "High": 0.15}),
        "SleepHours": sleep_hours,
        "SleepQuality": _sleep_quality_from_hours(rng, sleep_hours, "neutral"),
        "PreviousFeedback": _feedback_for_strategy(rng, "IntensiveStudy"),
        TARGET_COLUMN: "IntensiveStudy",
    }


def _generate_long_term(rng: np.random.Generator) -> dict:
    sleep_hours = round(float(rng.uniform(7.0, 9.0)), 1)
    return {
        "HoursStudied": round(float(rng.uniform(1.0, 5.0)), 1),
        "NumberOfSubjects": int(rng.integers(2, 7)),
        "DaysUntilExam": int(rng.integers(14, 31)),
        "StressLevel": _pick(rng, {"Low": 0.65, "Medium": 0.30, "High": 0.05}),
        "FatigueLevel": _pick(rng, {"Low": 0.70, "Medium": 0.25, "High": 0.05}),
        "SleepHours": sleep_hours,
        "SleepQuality": _sleep_quality_from_hours(rng, sleep_hours, "good"),
        "PreviousFeedback": _feedback_for_strategy(rng, "LongTermPlan"),
        TARGET_COLUMN: "LongTermPlan",
    }


def _row_key(row: dict) -> tuple:
    return tuple(row[col] for col in ALL_COLUMNS)


def _generate_class_rows(
    rng: np.random.Generator,
    strategy: str,
    count: int,
    generator,
    seen: set[tuple],
    max_attempts: int = 200,
) -> list[dict]:
    rows: list[dict] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * max_attempts:
            raise RuntimeError(f"Could not generate enough unique rows for {strategy}")

        row = generator(rng)
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def generate_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    rows: list[dict] = []

    rows.extend(_generate_class_rows(rng, "Rest", samples_per_class, _generate_rest, seen))

    balanced_count = samples_per_class - 25
    rows.extend(
        _generate_class_rows(
            rng, "BalancedStudy", balanced_count,
            lambda r: _generate_balanced(r), seen,
        )
    )
    rows.extend(
        _generate_class_rows(
            rng, "BalancedStudy", 25,
            lambda r: _generate_balanced(r, negative_feedback_shift=True), seen,
        )
    )

    rows.extend(
        _generate_class_rows(rng, "IntensiveStudy", samples_per_class, _generate_intensive, seen)
    )
    rows.extend(
        _generate_class_rows(rng, "LongTermPlan", samples_per_class, _generate_long_term, seen)
    )

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_dataset(df: pd.DataFrame) -> tuple[Path, Path]:
    schema = load_schema()
    validate_dataset(df, schema)

    schema.source_path.parent.mkdir(parents=True, exist_ok=True)
    schema.raw_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(schema.source_path, index=False)
    df.to_csv(schema.raw_path, index=False)
    return schema.source_path, schema.raw_path


def main() -> None:
    df = generate_dataset()
    source_path, raw_path = save_dataset(df)

    print(f"Generated {len(df)} rows")
    print(f"Saved: {source_path}")
    print(f"Copied: {raw_path}")
    print()
    print("Class distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())
    print()
    print("StressLevel distribution:")
    print(df["StressLevel"].value_counts().to_string())
    print()
    print("Summary statistics (numeric features):")
    print(df[["HoursStudied", "NumberOfSubjects", "DaysUntilExam", "SleepHours"]].describe().T.to_string())


if __name__ == "__main__":
    main()
