"""Read project metrics from real files — no hardcoded values in the UI."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.paths import (
    CV_DIR,
    DATA_RAW,
    EVALUATION_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
)

MODEL_LABELS = {
    "random_forest": "Nasumična šuma",
    "logistic_regression": "Logistička regresija",
    "decision_tree": "Stablo odlučivanja",
    "gradient_boosting": "Gradijentni boosting",
}

ML_PIPELINE_STEPS = 8


@dataclass
class EvaluationMetrics:
    best_model_key: str | None
    best_model_label: str | None
    holdout_f1: float | None
    model_count: int | None
    cv_mean_f1: float | None
    cv_std_f1: float | None


def fmt_percent(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def fmt_number(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


@st.cache_data(show_spinner=False)
def load_cv_fold_results() -> pd.DataFrame | None:
    path = CV_DIR / "cross_validation_results.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, skiprows=1, nrows=5)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> EvaluationMetrics:
    comparison_path = EVALUATION_DIR / "model_comparison.csv"
    best_model_key = None
    best_model_label = None
    holdout_f1 = None
    model_count = None

    if comparison_path.exists():
        try:
            df = pd.read_csv(comparison_path)
            model_count = len(df)
            if "F1_macro" in df.columns and not df.empty:
                best_row = df.loc[df["F1_macro"].idxmax()]
                best_model_key = str(best_row["Model"])
                best_model_label = MODEL_LABELS.get(best_model_key, best_model_key)
                holdout_f1 = float(best_row["F1_macro"])
        except (OSError, pd.errors.ParserError, ValueError, KeyError):
            pass

    cv_df = load_cv_fold_results()
    cv_mean_f1 = None
    cv_std_f1 = None
    if cv_df is not None and "f1_macro" in cv_df.columns:
        cv_mean_f1 = float(cv_df["f1_macro"].mean())
        cv_std_f1 = float(cv_df["f1_macro"].std())

    return EvaluationMetrics(
        best_model_key=best_model_key,
        best_model_label=best_model_label,
        holdout_f1=holdout_f1,
        model_count=model_count,
        cv_mean_f1=cv_mean_f1,
        cv_std_f1=cv_std_f1,
    )


@st.cache_data(show_spinner=False)
def _feedback_record_count_cached(_db_mtime: float, _csv_mtime: float) -> int | None:
    db_path = PROJECT_ROOT / "data" / "feedback" / "feedback.db"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error:
            pass

    csv_path = PROJECT_ROOT / "data" / "feedback" / "prediction_history.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return len(df)
        except (OSError, pd.errors.ParserError):
            pass

    return None


def feedback_record_count() -> int | None:
    """Ukupan broj predikcija u feedback bazi — osvježava se kad dodate preporuku."""
    db_path = PROJECT_ROOT / "data" / "feedback" / "feedback.db"
    csv_path = PROJECT_ROOT / "data" / "feedback" / "prediction_history.csv"
    db_mtime = db_path.stat().st_mtime if db_path.exists() else 0.0
    csv_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0.0
    return _feedback_record_count_cached(db_mtime, csv_mtime)


@st.cache_data(show_spinner=False)
def eda_figure_count() -> int | None:
    if not FIGURES_DIR.exists():
        return None
    count = len(list(FIGURES_DIR.glob("*.png")))
    return count if count > 0 else None


@st.cache_data(show_spinner=False)
def target_class_count() -> int | None:
    if not DATA_RAW.exists():
        return None
    try:
        df = pd.read_csv(DATA_RAW, keep_default_na=False)
        if "RecommendedStrategy" not in df.columns:
            return None
        return int(df["RecommendedStrategy"].nunique())
    except (OSError, pd.errors.ParserError):
        return None


def models_available() -> bool:
    required = ("best_model.pkl", "preprocessor.pkl", "label_encoder.pkl")
    return all((MODELS_DIR / name).exists() for name in required)
