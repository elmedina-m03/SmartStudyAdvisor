"""Bosnian display labels for dataset columns and categorical values."""

from __future__ import annotations

import pandas as pd

from app.components.constants import (
    FEEDBACK_TYPE_LABELS_BS,
    LEVEL_LABELS_BS,
    SLEEP_QUALITY_LABELS_BS,
    STRATEGY_LABELS,
)

FEATURE_LABELS_BS: dict[str, str] = {
    "HoursStudied": "Sati učenja",
    "NumberOfSubjects": "Broj predmeta",
    "DaysUntilExam": "Dana do ispita",
    "StressLevel": "Nivo stresa",
    "FatigueLevel": "Nivo umora",
    "SleepHours": "Sati sna",
    "SleepQuality": "Kvalitet sna",
    "PreviousFeedback": "Prethodna povratna informacija",
    "RecommendedStrategy": "Preporučena strategija",
}

CATEGORICAL_VALUE_MAPS: dict[str, dict[str, str]] = {
    "StressLevel": LEVEL_LABELS_BS,
    "FatigueLevel": LEVEL_LABELS_BS,
    "SleepQuality": SLEEP_QUALITY_LABELS_BS,
    "PreviousFeedback": FEEDBACK_TYPE_LABELS_BS,
    "RecommendedStrategy": STRATEGY_LABELS,
}


def strategy_label(value: str) -> str:
    return STRATEGY_LABELS.get(value, value)


def feature_label(value: str) -> str:
    return FEATURE_LABELS_BS.get(value, value)


def categorical_value_label(column: str, value: str) -> str:
    mapping = CATEGORICAL_VALUE_MAPS.get(column, {})
    return mapping.get(str(value), str(value))


def rename_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [feature_label(str(col)) for col in out.columns]
    return out


def rename_strategy_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.name in (None, "RecommendedStrategy"):
        out.index = [strategy_label(str(idx)) for idx in out.index]
        out.index.name = "Strategija"
    return out


def translate_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        column = str(row["Feature"])
        value = str(row["Vrijednost"])
        rows.append(
            {
                "Značajka": feature_label(column),
                "Vrijednost": categorical_value_label(column, value),
                "Broj": int(row["Broj"]),
            }
        )
    return pd.DataFrame(rows)
