"""Shared constants and display labels for the Streamlit app."""

from __future__ import annotations

# Internal values (ML model / dataset contract — English)
STRATEGIES = ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"]
STRESS_LEVELS = ["Low", "Medium", "High"]
FATIGUE_LEVELS = ["Low", "Medium", "High"]
SLEEP_QUALITY = ["Poor", "Average", "Good"]
FEEDBACK_TYPES = ["None", "Positive", "Negative", "Mixed"]
# Zastarjelo — koristite FEEDBACK_OPTIONS_BS u UI-u
FEEDBACK_OPTIONS = STRATEGIES + ["Drugo / Nisam siguran/na"]

# Bosnian display labels for categorical inputs
LEVEL_LABELS_BS: dict[str, str] = {
    "Low": "Nizak",
    "Medium": "Srednji",
    "High": "Visok",
}

SLEEP_QUALITY_LABELS_BS: dict[str, str] = {
    "Poor": "Loš",
    "Average": "Prosječan",
    "Good": "Dobar",
}

FEEDBACK_TYPE_LABELS_BS: dict[str, str] = {
    "None": "Nema",
    "Positive": "Pozitivan",
    "Negative": "Negativan",
    "Mixed": "Miješan",
}

STRATEGY_LABELS = {
    "Rest": "Odmor",
    "BalancedStudy": "Uravnoteženo učenje",
    "IntensiveStudy": "Intenzivno učenje",
    "LongTermPlan": "Dugoročni plan",
}

# Opcije povratne informacije — prikaz na bosanskom
FEEDBACK_OPTIONS_BS = [STRATEGY_LABELS[s] for s in STRATEGIES] + ["Drugo / Nisam siguran/na"]

PROJECT_NAME = "Smart Study Savjetnik"

STRATEGY_COLORS = {
    "Rest": "#E74C3C",
    "BalancedStudy": "#3498DB",
    "IntensiveStudy": "#F39C12",
    "LongTermPlan": "#27AE60",
}

AGENT_PHASES = [
    ("Percepcija", "Prikupljanje i validacija unosa", "#6366F1"),
    ("Predikcija", "ML predikcija strategije", "#8B5CF6"),
    ("Preporuka", "Objašnjenje i savjeti", "#06B6D4"),
    ("Učenje", "Pohrana povratne informacije", "#10B981"),
]

AGENT_CYCLE_LABEL = "Percepcija → Predikcija → Preporuka → Učenje"

PROJECT_AUTHORS = "Elmedina Marić, Aldina Kurtović"
PROJECT_COURSE = "Mašinsko učenje"
