"""Reusable stat-card builders tied to real data sources."""

from __future__ import annotations

from app.components.data_loader import dataset_overview
from app.components.metrics import fmt_number


def dataset_records_card() -> dict:
    """Kartica broja zapisa — vrijednost iz CSV, ne hardkodirano."""
    overview = dataset_overview()
    rows = overview.get("records")
    return {
        "label": "Zapisa u datasetu",
        "value": fmt_number(rows),
        "icon": "📋",
        "delta": "ML dataset · ne raste pri korištenju aplikacije",
        "help_text": "Fiksni trening skup iz data/raw/student_study_strategy.csv. "
        "Novi unosi iz aplikacije idu u feedback bazu, ne ovdje.",
        "color": "#6366F1",
    }
