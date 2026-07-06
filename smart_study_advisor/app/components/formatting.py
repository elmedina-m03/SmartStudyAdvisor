"""Consistent number formatting across the UI."""

from __future__ import annotations

import pandas as pd


def fmt_score(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_decimal(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


_SKIP_FORMAT = {
    "Model",
    "Eksperiment",
    "Opis",
    "Značajka",
    "Feature",
    "Osnovni feature",
    "Uklonjen feature",
    "Kategorija",
    "Outlieri",
    "Uklonjeni feature-i",
    "Strategija",
}


def style_score_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    format_map: dict[str, str] = {}
    for col in df.columns:
        if col in _SKIP_FORMAT:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if col in ("Važnost", "Δ F1", "Doprinos"):
            format_map[col] = "{:.4f}"
        elif col in ("Broj", "Rang", "Rank", "Fold", "Train zapisa", "Test zapisa", "Broj feature-a"):
            format_map[col] = "{:.0f}"
        elif col in ("Train omjer", "Test omjer"):
            format_map[col] = "{:.0%}"
        else:
            format_map[col] = "{:.2%}"
    return df.style.format(format_map)
