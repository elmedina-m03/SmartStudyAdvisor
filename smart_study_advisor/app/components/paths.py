"""Central path definitions for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
ML_ROOT = PROJECT_ROOT / "ml"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "student_study_strategy.csv"

ML_OUTPUT = ML_ROOT / "output"
FIGURES_DIR = ML_OUTPUT / "figures"
EVALUATION_DIR = ML_OUTPUT / "evaluation"
FEATURE_IMPORTANCE_DIR = ML_OUTPUT / "feature_importance"
ABLATION_DIR = ML_OUTPUT / "ablation"
CV_DIR = ML_OUTPUT / "cross_validation"
EXPERIMENTS_DIR = ML_OUTPUT / "experiments"
CORRELATION_PAIRS_NUMERIC = ML_OUTPUT / "correlation_pairs_numeric.csv"
CORRELATION_PAIRS_ENCODED = ML_OUTPUT / "correlation_pairs_encoded.csv"
REDUNDANCY_REPORT = ML_OUTPUT / "redundancy_analysis.md"
FINAL_REPORT = PROJECT_ROOT / "reports" / "final_ml_report.md"
