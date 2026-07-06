"""Cached data loading for Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.paths import DATA_RAW, ML_ROOT, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from app.components.labels import (  # noqa: E402
    feature_label,
    rename_feature_columns,
    rename_strategy_index,
    strategy_label,
    translate_categorical_summary,
)
from src.data.schema import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def _dataset_mtime() -> float:
    if not DATA_RAW.exists():
        return 0.0
    return DATA_RAW.stat().st_mtime


@st.cache_data(show_spinner=False)
def _cached_study_dataset(_file_version: float) -> pd.DataFrame:
    return pd.read_csv(DATA_RAW, keep_default_na=False)


def get_study_dataset() -> pd.DataFrame:
    """Učitaj dataset; keš se poništava kad se CSV promijeni."""
    return _cached_study_dataset(_dataset_mtime())


def load_study_dataset() -> pd.DataFrame:
    """Javni API za učitavanje dataseta (kompatibilno sa starim pozivima)."""
    return get_study_dataset()


def clear_dataset_cache() -> None:
    """Invalidate cached dataset reads (e.g. after file replacement)."""
    _cached_study_dataset.clear()


def dataset_provenance() -> dict[str, str | int | bool | None]:
    """Tačan izvor broja zapisa — za prikaz u UI."""
    if not DATA_RAW.exists():
        return {
            "exists": False,
            "rows": None,
            "filename": None,
            "path_display": "data/raw/student_study_strategy.csv",
        }
    df = get_study_dataset()
    return {
        "exists": True,
        "rows": len(df),
        "filename": DATA_RAW.name,
        "path_display": "data/raw/student_study_strategy.csv",
    }


DESC_STATS_COLUMNS = {
    "count": "Broj",
    "mean": "Sredina",
    "std": "Std. dev.",
    "min": "Min",
    "25%": "Q1",
    "50%": "Medijan",
    "75%": "Q3",
    "max": "Max",
}

DESC_STATS_FORMAT = {
    "Broj": "{:.0f}",
    "Sredina": "{:.2f}",
    "Std. dev.": "{:.2f}",
    "Min": "{:.2f}",
    "Q1": "{:.2f}",
    "Medijan": "{:.2f}",
    "Q3": "{:.2f}",
    "Max": "{:.2f}",
}


@st.cache_data(show_spinner=False)
def descriptive_statistics() -> pd.DataFrame:
    df = get_study_dataset()
    stats = df[NUMERIC_FEATURES].describe().T
    stats = stats.rename(columns=DESC_STATS_COLUMNS)
    stats.index = [feature_label(str(idx)) for idx in stats.index]
    stats.index.name = "Značajka"
    return stats.round(2)


def styled_descriptive_statistics():
    """Deskriptivna statistika s čitljivim zaokruživanjem (2 decimale, Broj bez decimale)."""
    return (
        descriptive_statistics()
        .style.format(DESC_STATS_FORMAT)
        .background_gradient(cmap="Blues", axis=None)
    )


@st.cache_data(show_spinner=False)
def strategy_means() -> pd.DataFrame:
    df = get_study_dataset()
    order = ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"]
    means = df.groupby(TARGET_COLUMN)[NUMERIC_FEATURES].mean().reindex(order).round(2)
    return rename_strategy_index(rename_feature_columns(means))


def styled_strategy_means():
    return strategy_means().style.format("{:.2f}")


@st.cache_data(show_spinner=False)
def class_balance() -> pd.DataFrame:
    df = get_study_dataset()
    order = ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"]
    counts = df[TARGET_COLUMN].value_counts().reindex(order)
    result = pd.DataFrame(
        {
            "Broj": counts.astype(int),
            "Postotak (%)": (counts / len(df) * 100).round(1),
        }
    )
    result.index = [strategy_label(str(idx)) for idx in result.index]
    result.index.name = "Strategija"
    return result


@st.cache_data(show_spinner=False)
def categorical_summary() -> pd.DataFrame:
    df = get_study_dataset()
    rows = []
    for col in CATEGORICAL_FEATURES:
        for value, count in df[col].value_counts().items():
            rows.append({"Feature": col, "Vrijednost": value, "Broj": int(count)})
    return translate_categorical_summary(pd.DataFrame(rows))


def iqr_outlier_count() -> int | None:
    """Ukupan broj IQR outliera na numeričkim feature-ima."""
    if not DATA_RAW.exists():
        return None
    try:
        df = get_study_dataset()
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    flagged: set[int] = set()
    for column in NUMERIC_FEATURES:
        series = df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for idx in series[(series < lower) | (series > upper)].index:
            flagged.add(int(idx))
    return len(flagged)


def _base_overview() -> dict[str, int | str | None]:
    return {
        "records": None,
        "features": len(FEATURE_COLUMNS),
        "numeric": len(NUMERIC_FEATURES),
        "categorical": len(CATEGORICAL_FEATURES),
        "missing": None,
        "duplicates": None,
        "target": TARGET_COLUMN,
        "classes": None,
        "source_file": None,
    }


def dataset_overview() -> dict[str, int | str | None]:
    """Dataset summary — broj zapisa se uvijek čita iz CSV fajla."""
    overview = _base_overview()
    prov = dataset_provenance()
    if not prov["exists"]:
        return overview
    overview["records"] = prov["rows"]
    overview["source_file"] = prov["filename"]
    try:
        df = get_study_dataset()
    except (OSError, pd.errors.ParserError, ValueError):
        return overview
    overview.update(
        {
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "classes": int(df[TARGET_COLUMN].nunique()) if TARGET_COLUMN in df.columns else None,
        }
    )
    return overview
